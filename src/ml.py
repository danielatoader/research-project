import numpy as np
from sklearn import tree
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
import math

from ProcessEvoSuiteData import ProcessEvoSuiteData

data = ProcessEvoSuiteData()

labels_dict_gl = dict()
labels_dict_gl[60] = {'branch_60':0, 'default_60':1, 'weak_60':2}
labels_dict_gl[180] = {'branch_180':0, 'default_180':1, 'weak_180':2}
labels_dict_gl[300] = {'branch_300':0, 'default_300':1, 'weak_300':2}

how_many = 0

def get_significant_classes_matching_metrics(search_budget):
    """ Get significant classes and their corresponding matching metrics. """

    significant_classes_stats = {}
    key_budget = 'stats' + str(search_budget)

    # Contains the coverage results per class from EvoSuite (e.g. coverage_res_filename = "res_data/results-60.csv")
    coverage_res_filename = 'res_data/results-' + str(search_budget) + '.csv'

    # Contains statistically significant classes at [0] and all classes at [1]
    # res_dict['stats60'][0].items() contains pairs of type (class_name, p-value)
    significant_classes_stats[str(key_budget + '_branch')] = data.get_significant_classes_stats(coverage_res_filename, search_budget, 'branch')
    significant_classes_stats[str(key_budget + '_default')] = data.get_significant_classes_stats(coverage_res_filename, search_budget, 'default')

    matching_classes_metrics = {}
    signif_matched_metrics = {}
    key_class_metrics = 'class_metrics' + str(search_budget)

    # Contains the dictionary with matched metrics (for classes from EvoSuite output and CK tool)
    # Should probably be reindexed (or ignore index)
    matching_classes_metrics[key_class_metrics] = data.get_matching_classes_metrics(coverage_res_filename, search_budget)

    # Contains names of statistically significant classes
    significant_classes = list(set(list(significant_classes_stats[str(key_budget + '_branch')][0].keys()) + list(significant_classes_stats[str(key_budget + '_branch')][0].keys())))

    # Matched metrics for the significant classes
    matched_metrics = matching_classes_metrics[key_class_metrics]
    signif_matched_metrics[key_class_metrics] = matched_metrics[matched_metrics.apply(lambda row : row['class'] in significant_classes, axis=1)]

    return signif_matched_metrics

def get_X_y(search_budget):
    """ Gets the dataset and the taget labels to further train the ML models. """

    metrics_budget = 'class_metrics' + str(search_budget)
    signif_matched_metrics = get_significant_classes_matching_metrics(search_budget)
    medians = data.calculate_medians(search_budget)[['TARGET_CLASS', 'configuration_id', 'BranchCoverage']]

    X = []
    classes = []

    # Loop through class names
    for cls in signif_matched_metrics[metrics_budget]['class'].items():
        # Select only those classes for which all 3 configuration data points are available
        if medians.loc[medians['TARGET_CLASS']==cls[1]].shape[0] != 3:
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
        y.append(get_label(search_budget, cl, medians))
    # y = list(map(get_label(classes)))
    
    # print(f"Non-unique maximums for {how_many} out of {len(X)} entries")
    # print(y)
    assert len(y) == len(X), f"X and y should have the same number of entries, but they have {len(X)} and {len(y)}, respectively."

    return X, y

def get_label(search_budget, class_name, medians, labels_dict=None):
    if (not labels_dict):
        labels_dict = labels_dict_gl[search_budget]

    global how_many
    # Get matching rows
    selected_rows = medians.loc[medians['TARGET_CLASS']==class_name]
    assert selected_rows.shape[0] == 3, f"Expected 3 selected rows, but got {selected_rows[0]}"
    
    # Get the maximum branch covreage of the three data points
    max_coverage = selected_rows.max(numeric_only=True)['BranchCoverage']
    
    # Select the configuration_id by the maximum branch coverage
    # TODO: decide on a policy for equality
    # A lot of the datapoints have equal values, so this is an extremely important decision
    # Currently: just pick the last one
    max_rows = selected_rows.loc[selected_rows['BranchCoverage']==max_coverage]
    
    # Count the number of labels that have non-unique maximums
    if max_rows.shape[0] > 1:
        how_many += 1
        
    # Select the first row of the maximum ones
    max_config_id = max_rows.iloc[-1]['configuration_id']
    
    assert max_config_id in labels_dict, f"Expected configuration id to be one of {labels_dict.keys()}, but got {max_config_id}"
    return labels_dict[max_config_id]


def run_KFold(search_budgets):
    """ Run KFold cross validation using X, y and the chosen models. """

    f1s = dict()
    for search_budget in search_budgets:
        X, y = get_X_y(search_budget)
        X = np.array(X)
        y = np.array(y)

        kf = KFold(n_splits=20, random_state=42, shuffle=True)
        f1s_dt = []
        f1s_svc = []
        for train_index, test_index in kf.split(X):
            # Split the data
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # Train svc
            clf = SVC()
            clf.set_params(kernel='rbf').fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            f1s_svc.append(f1_score(y_pred, y_test, average='weighted'))
            
            # Train decision tree
            decision_tree = tree.DecisionTreeClassifier(random_state=456).fit(X_train, y_train)
            y_pred = decision_tree.predict(X_test)
            f1s_dt.append(f1_score(y_pred, y_test, average='weighted'))

        f1s[search_budget] = (f1s_dt, f1s_svc)

    return f1s

if __name__ == '__main__':
    f1s = run_KFold([60, 180, 300])
    print( 'DT: ' + str(np.average(f1s[60][0])) + ' SVC: ' + str(np.average(f1s[60][1])))
    print( 'DT: ' + str(np.average(f1s[180][0])) + ' SVC: ' + str(np.average(f1s[180][1])))
    print( 'DT: ' + str(np.average(f1s[300][0])) + ' SVC: ' + str(np.average(f1s[300][1])))
