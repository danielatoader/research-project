import numpy as np
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier

from feature_selection_extraction import select_features
from make_dataset import get_X_y, get_balanced_X_y


def run_KFold(search_budgets=[60,180,300], columns_to_group=['TARGET_CLASS', 'configuration_id', 'project.id'], score_metric='BranchCoverage', score_metric_filename='res_data/results-'):
    """
    Run KFold cross validation using X, y, and the chosen models.
    """

    f1s = dict()
    for search_budget in search_budgets:

        # Get samples and their labels
        # Also balance data
        X_res, y_res, features = get_balanced_X_y(search_budget, columns_to_group, score_metric, score_metric_filename, labels_dict=None)
        
        # X = np.array(X)
        # y = np.array(y)

        # Apply dataset balancing techniques
        # over_sampler = RandomOverSampler(random_state=42)
        # X_res, y_res = over_sampler.fit_resample(X, y)
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
            X_train, X_test, selected_features = select_features(X_train, X_test, features)
            
            # Train SVC
            clf = SVC()
            clf.set_params(kernel='rbf').fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            f1s_svc.append(f1_score(y_pred, y_test, average='weighted'))
            
            # Train Decision Tree
            decision_tree = tree.DecisionTreeClassifier(criterion="entropy", max_depth=15).fit(X_train, y_train)
            y_pred = decision_tree.predict(X_test)
            f1s_dt.append(f1_score(y_pred, y_test, average='weighted'))
            tree.plot_tree(decision_tree)

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

