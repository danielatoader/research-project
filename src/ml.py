import numpy as np
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from numpy import mean
from numpy import std
# from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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
            f1s_svc.append(f1_score(y_test, y_pred, average='weighted'))
            
            # Train Decision Tree
            decision_tree = tree.DecisionTreeClassifier(criterion="entropy", max_depth=15).fit(X_train, y_train)
            y_pred = decision_tree.predict(X_test)
            f1s_dt.append(f1_score(y_test, y_pred, average='weighted'))
            tree.plot_tree(decision_tree)

            # Train Random Forest classifier
            random_forest = RandomForestClassifier(n_estimators = 100).fit(X_train, y_train)
            y_pred = random_forest.predict(X_test)
            f1s_rf.append(f1_score(y_test, y_pred, average='weighted'))

            # Train a Logistic Regression classifier
            logistic_regression = LogisticRegression(random_state=0, max_iter=300).fit(X_train, y_train)
            y_pred = logistic_regression.predict(X_test)
            f1s_lr.append(f1_score(y_test, y_pred, average='weighted'))

        f1s[search_budget] = (f1s_dt, f1s_svc, f1s_rf, f1s_lr)

    return f1s

def double_K_fold(search_budget=60, columns_to_group=['TARGET_CLASS', 'configuration_id', 'project.id'], score_metric='BranchCoverage', score_metric_filename='res_data/results-'):
    # create dataset
    X, y, features = get_balanced_X_y(search_budget, columns_to_group, score_metric, score_metric_filename, labels_dict=None)
    # configure the cross-validation procedure
    cv_outer = KFold(n_splits=10, shuffle=True, random_state=1)
    # enumerate splits
    outer_results = list()
    for train_ix, test_ix in cv_outer.split(X):
        # split data
        X_train, X_test = X[train_ix, :], X[test_ix, :]
        y_train, y_test = y[train_ix], y[test_ix]
        
        # Feature selection
        X_train, X_test, selected_features = select_features(X_train, X_test, features)

        # configure the cross-validation procedure
        cv_inner = KFold(n_splits=3, shuffle=True, random_state=1)
        # define the model
        model = RandomForestClassifier(random_state=1)
        # define search space
        space = dict()
        space['n_estimators'] = [10, 100, 500]
        space['max_features'] = [2, 4, 6, 10, 20]
        # define search
        search = GridSearchCV(model, space, scoring='f1_weighted', cv=cv_inner, refit=True)
        # execute search
        result = search.fit(X_train, y_train)
        # get the best performing model fit on the whole training set
        best_model = result.best_estimator_
        # evaluate model on the hold out dataset
        yhat = best_model.predict(X_test)
        # evaluate the model
        acc = f1_score(y_test, yhat, average='weighted')
        # store the result
        outer_results.append(acc)
        # report progress
        print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_))
        # summarize the estimated performance of the model
    print('F1-score: %.3f (%.3f)' % (mean(outer_results), std(outer_results)))


def auto_KFold(model, space, search_budget=60, columns_to_group=['TARGET_CLASS', 'configuration_id', 'project.id'], score_metric='BranchCoverage', score_metric_filename='res_data/results-'):
    X, y, features = get_balanced_X_y(search_budget, columns_to_group, score_metric, score_metric_filename, labels_dict=None)
    # Feature selection
    X, _, selected_features = select_features(X, X, features)
    # configure the cross-validation procedure
    cv_inner = KFold(n_splits=3, shuffle=True, random_state=1)

    # define the model
    # model = RandomForestClassifier(random_state=1)
    # define search space
    # space = dict()
    # space['n_estimators'] = [10, 100, 500]
    # space['max_features'] = [2, 4, 6]

    # define search
    search = GridSearchCV(model, space, scoring='f1_weighted', n_jobs=1, cv=cv_inner, refit=True)
    # configure the cross-validation procedure
    cv_outer = KFold(n_splits=10, shuffle=True, random_state=1)
    # execute the nested cross-validation
    scores = cross_val_score(search, X, y, scoring='f1_weighted', cv=cv_outer, n_jobs=-1)
    # report performance
    print('F1-score: %.3f (%.3f)' % (mean(scores), std(scores)))

def run_KFold_Grid_all_models(search_budget=60, columns_to_group=['TARGET_CLASS', 'configuration_id', 'project.id'], score_metric='BranchCoverage', score_metric_filename='res_data/results-'):
    X, y, features = get_balanced_X_y(search_budget, columns_to_group, score_metric, score_metric_filename, labels_dict=None)
    # Feature selection
    X, _, selected_features = select_features(X, X, features)

    models = []

    models.append(("LogisticRegression",LogisticRegression()))
    models.append(("SVC",SVC()))
    # models.append(("LinearSVC",LinearSVC()))
    # models.append(("KNeighbors",KNeighborsClassifier()))
    models.append(("DecisionTree",tree.DecisionTreeClassifier()))
    models.append(("RandomForest",RandomForestClassifier()))
    # rf2 = RandomForestClassifier(n_estimators=100, criterion='gini',
                                    # max_depth=10, random_state=0, max_features=None)
    # models.append(("RandomForest2",rf2))
    # models.append(("MLPClassifier",MLPClassifier(solver='lbfgs', random_state=0)))

    space = dict()
    for (name, model) in models:
        space[name] = {}

    space['RandomForest']['n_estimators'] = [10, 100, 500]
    space['RandomForest']['max_features'] = [2, 4, 6]
    space['SVC']['kernel'] = ['rbf', 'linear']
    space['DecisionTree']['criterion'] = ['entropy', 'gini']
    space['LogisticRegression']['random_state'] = [0]
    
    results = []
    names = []
    for (name, model) in models:
        grid = space[name]
        cv_inner = KFold(n_splits=3, shuffle=True, random_state=1)
        # define search
        search = GridSearchCV(model, grid, scoring='f1_weighted', n_jobs=1, cv=cv_inner, refit=True)
        # configure the cross-validation procedure
        cv_outer = KFold(n_splits=10, shuffle=True, random_state=1)
        # execute the nested cross-validation
        scores = cross_val_score(search, X, y, scoring='f1_weighted', cv=cv_outer, n_jobs=-1)
        # report performance
        # print('F1-score: %.3f (%.3f)' % (mean(scores), std(scores)))

        result = mean(scores)
        names.append(name)
        results.append(result)

    for i in range(len(names)):
        print(names[i],results[i].mean())


if __name__ == '__main__':
    run_KFold_Grid_all_models()
    # auto_KFold()
    # double_K_fold()

    # f1s_coverage = run_KFold(search_budgets=[60, 180, 300], columns_to_group=['TARGET_CLASS', 'configuration_id', 'project.id'], score_metric='BranchCoverage', score_metric_filename='res_data/results-')
    # print( 'DT: ' + str(np.average(f1s_coverage[60][0])) + ' SVC: ' + str(np.average(f1s_coverage[60][1])) + ' RF: ' + str(np.average(f1s_coverage[60][2])) + ' LR: ' + str(np.average(f1s_coverage[60][3])))
    # print( 'DT: ' + str(np.average(f1s_coverage[180][0])) + ' SVC: ' + str(np.average(f1s_coverage[180][1])) + ' RF: ' + str(np.average(f1s_coverage[180][2])) + ' LR: ' + str(np.average(f1s_coverage[180][3])))
    # print( 'DT: ' + str(np.average(f1s_coverage[300][0])) + ' SVC: ' + str(np.average(f1s_coverage[300][1])) + ' RF: ' + str(np.average(f1s_coverage[300][2])) + ' LR: ' + str(np.average(f1s_coverage[300][3])))

    # f1s_mutation_score = run_KFold(search_budgets=[60], columns_to_group=['class', 'configuration', 'project'], score_metric='mutation_score_percent', score_metric_filename='res_data/mutation_scores.csv')
    # print( 'DT_mutation: ' + str(np.average(f1s_mutation_score[60][0])) + ' SVC_mutation: ' + str(np.average(f1s_mutation_score[60][1])) + ' RF: ' + str(np.average(f1s_mutation_score[60][2])) + ' LR: ' + str(np.average(f1s_mutation_score[60][3])))

