import numpy as np
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import math

from process_evosuite_data import ProcessEvoSuiteData
from feature_selection_extraction import select_features

data = ProcessEvoSuiteData()

labels_dict_gl = dict()
labels_dict_gl[60] = {'branch_60':0, 'default_60':1, 'weak_60':2}
labels_dict_gl[180] = {'branch_180':0, 'default_180':1, 'weak_180':2}
labels_dict_gl[300] = {'branch_300':0, 'default_300':1, 'weak_300':2}

how_many = 0

def get_label(search_budget, class_name, medians, columns_to_group, score_metric, labels_dict=None):
    """
    Get the label of a specific sample based on the medians of the score metric.
    """
    
    if (not labels_dict):
        labels_dict = labels_dict_gl[search_budget]

    global how_many
    class_column = columns_to_group[0]
    config_column = columns_to_group[1]

    # Get matching rows
    selected_rows = medians.loc[medians[class_column]==class_name]
    assert selected_rows.shape[0] == 3, f"Expected 3 selected rows, but got {selected_rows[0]}"
    
    # Get the maximum branch covreage of the three data points
    max_coverage = selected_rows.max(numeric_only=True)[score_metric]
    
    # Select the configuration_id by the maximum branch coverage
    # TODO: decide on a policy for equality
    # A lot of the datapoints have equal values, so this is an extremely important decision
    # Currently: just pick the last one
    max_rows = selected_rows.loc[selected_rows[score_metric]==max_coverage]
    
    # Count the number of labels that have non-unique maximums
    if max_rows.shape[0] > 1:
        how_many += 1
        
    # Select the first row of the maximum ones
    max_config_id = max_rows.iloc[-1][config_column]
    
    assert max_config_id in labels_dict, f"Expected configuration id to be one of {labels_dict.keys()}, but got {max_config_id}"

    return labels_dict[max_config_id]

def get_X_y(search_budget, columns_to_group, score_metric, score_metric_filename, labels_dict=None):
    """
    Gets the dataset and the taget labels to further train the ML models.
    """

    medians_columns = [columns_to_group[0], columns_to_group[1], score_metric]
    class_column = columns_to_group[0]
    metrics_budget = 'class_metrics' + str(search_budget)

    signif_matched_metrics = data.get_significant_classes_matching_metrics(
        search_budget,
        columns_to_group,
        score_metric,
        score_metric_filename)

    medians = data.calculate_medians(search_budget, columns_to_group, score_metric, score_metric_filename)[medians_columns]

    X = []
    classes = []
    features = signif_matched_metrics[metrics_budget].keys()[2:]

    # Loop through class names
    for cls in signif_matched_metrics[metrics_budget]['class'].items():
        # Select only those classes for which all 3 configuration data points are available
        if medians.loc[medians[class_column]==cls[1]].shape[0] != 3:
            continue
        X.append([])
        
        # Keep class names for later use
        # We need them to determine the proper labels
        classes.append(cls[1])
        
        # [2:] to skip the name and type
        for feature in signif_matched_metrics[metrics_budget].keys()[2:]:
            
            # X[-1] is the last (current) entry (class)
            # cls[0] is the id of the entry (class)
            # dict[class_metrics60][feature][cls[0]] is the specific feature
            # Of the current class
            feat = signif_matched_metrics[metrics_budget][feature][cls[0]]
            X[-1].append(feat if not math.isnan(feat) else 0)

    # All features should have the same length
    assert all(map(lambda x: len(x) == len(X[0]), X))

    y = []
    for cl in classes:
        y.append(get_label(search_budget, cl, medians, columns_to_group, score_metric, labels_dict))
    # y = list(map(get_label(classes)))
    
    # print(f"Non-unique maximums for {how_many} out of {len(X)} entries")
    # print(y)
    assert len(y) == len(X), f"X and y should have the same number of entries, but they have {len(X)} and {len(y)}, respectively."

    return X, y, features


def run_KFold(search_budgets=[60,180,300], columns_to_group=['TARGET_CLASS', 'configuration_id', 'project.id'], score_metric='BranchCoverage', score_metric_filename='res_data/results-'):
    """
    Run KFold cross validation using X, y and the chosen models.
    """

    f1s = dict()
    for search_budget in search_budgets:

        # Get samples and their labels
        X, y, features = get_X_y(search_budget, columns_to_group, score_metric, score_metric_filename, labels_dict=None)
        X = np.array(X)
        y = np.array(y)

        # Apply dataset balancing techniques
        over_sampler = RandomOverSampler(random_state=42)
        X_res, y_res = over_sampler.fit_resample(X, y)
        # under_sampler = RandomUnderSampler(random_state=42)
        # X_res, y_res = under_sampler.fit_resample(X, y)

        kf = KFold(n_splits=10, random_state=42, shuffle=True)
        f1s_svc = []
        f1s_dt = []
        f1s_rf = []
        f1s_lr = []

        for train_index, test_index in kf.split(X_res):
            # Split the data
            X_train, X_test = X_res[train_index], X_res[test_index]
            y_train, y_test = y_res[train_index], y_res[test_index]            

            # Feature selection
            X_train, X_test = select_features(X_train, X_test, features)
            
            # Train SVC
            clf = SVC()
            clf.set_params(kernel='rbf').fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            f1s_svc.append(f1_score(y_pred, y_test, average='weighted'))
            
            # Train Decision Tree
            decision_tree = tree.DecisionTreeClassifier(random_state=456).fit(X_train, y_train)
            y_pred = decision_tree.predict(X_test)
            f1s_dt.append(f1_score(y_pred, y_test, average='weighted'))

            # Train Random Forest classifier
            random_forest = RandomForestClassifier(n_estimators = 100).fit(X_train, y_train)
            y_pred = random_forest.predict(X_test)
            f1s_rf.append(f1_score(y_pred, y_test, average='weighted'))

            # Train a Logistic Regression classifier
            logistic_regression = LogisticRegression(random_state=0, max_iter=300).fit(X_train, y_train)
            y_pred = logistic_regression.predict(X_test)
            f1s_lr.append(f1_score(y_pred, y_test, average='weighted'))

        f1s[search_budget] = (f1s_dt, f1s_svc, f1s_rf, f1s_lr)

    return f1s

if __name__ == '__main__':
    f1s_coverage = run_KFold(search_budgets=[60, 180, 300], columns_to_group=['TARGET_CLASS', 'configuration_id', 'project.id'], score_metric='BranchCoverage', score_metric_filename='res_data/results-')
    print( 'DT: ' + str(np.average(f1s_coverage[60][0])) + ' SVC: ' + str(np.average(f1s_coverage[60][1])) + ' RF: ' + str(np.average(f1s_coverage[60][2])) + ' LR: ' + str(np.average(f1s_coverage[60][3])))
    print( 'DT: ' + str(np.average(f1s_coverage[180][0])) + ' SVC: ' + str(np.average(f1s_coverage[180][1])) + ' RF: ' + str(np.average(f1s_coverage[180][2])) + ' LR: ' + str(np.average(f1s_coverage[180][3])))
    print( 'DT: ' + str(np.average(f1s_coverage[300][0])) + ' SVC: ' + str(np.average(f1s_coverage[300][1])) + ' RF: ' + str(np.average(f1s_coverage[300][2])) + ' LR: ' + str(np.average(f1s_coverage[300][3])))

    f1s_mutation_score = run_KFold(search_budgets=[60], columns_to_group=['class', 'configuration', 'project'], score_metric='mutation_score_percent', score_metric_filename='res_data/mutation_scores.csv')
    print( 'DT_mutation: ' + str(np.average(f1s_mutation_score[60][0])) + ' SVC_mutation: ' + str(np.average(f1s_mutation_score[60][1])) + ' RF: ' + str(np.average(f1s_mutation_score[60][2])) + ' LR: ' + str(np.average(f1s_mutation_score[60][3])))

