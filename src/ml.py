from six import StringIO
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
import json
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
import sklearn.feature_selection as fs
from sklearn.feature_selection import RFECV
from numpy import mean
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
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

        # bk = fs.SelectKBest()
        # bk.fit(X_res, y_res)
        # X_transf = bk.transform(X_res)

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
            # X_train, X_test, selected_features = select_features(X_train, X_test, features)
            
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

def double_K_fold(
    search_budget=60,
    columns_to_group=['TARGET_CLASS', 'configuration_id', 'project.id'],
    score_metric='BranchCoverage',
    score_metric_filename='res_data/results-',
    k=10
):
    models = []

    models.append(("RandomForest",RandomForestClassifier()))
    models.append(("SVC",SVC()))
    models.append(("DecisionTree",tree.DecisionTreeClassifier()))
    models.append(("LogisticRegression",LogisticRegression()))
    # models.append(("LinearSVC",LinearSVC()))
    # models.append(("KNeighbors",KNeighborsClassifier()))

    # create dataset
    X, y, features = get_balanced_X_y(search_budget, columns_to_group, score_metric, score_metric_filename, labels_dict=None)
    
    # Feature selection
    bk = fs.SelectKBest(k=k)
    bk.fit(X, y)
    X_transf = bk.transform(X)
    features = [column[0]  for column in zip(features,bk.get_support()) if column[1]]
    print(features)
    # features = bk.get_support(indices=True)
    
    space = dict()
    for (name, model) in models:
        space[name] = {}

    space['RandomForest']['n_estimators'] = [10, 50, 100, 200, 500]
    space['RandomForest']['max_features'] = [2, 4, 6, 8]
    space['SVC']['kernel'] = ['rbf', 'linear']
    space['SVC']['C'] = [1,10,100,1000]
    space['SVC']['gamma'] = [1,0.1,0.001]
    space['DecisionTree']['criterion'] = ['entropy', 'gini']
    space['DecisionTree']['max_depth'] = [5, 10, 15, 20]
    space['DecisionTree']['max_leaf_nodes'] = [5, 10, 20, 30, 40]
    space['LogisticRegression']['penalty'] = ['none', 'l1', 'l2', 'elasticnet']
    space['LogisticRegression']['C'] = [1e-5, 1e-3, 1e-1, 10, 100]
    space['LogisticRegression']['solver'] = ['newton-cg', 'lbfgs', 'liblinear']
    
    results = []
    names = []
    for (name, model) in models:
        grid = space[name]
        # configure the cross-validation procedure
        cv_outer = KFold(n_splits=10, shuffle=True, random_state=1)
        # enumerate splits
        outer_results = list()
        for train_ix, test_ix in cv_outer.split(X_transf):
            # split data
            X_train, X_test = X_transf[train_ix, :], X_transf[test_ix, :]
            y_train, y_test = y[train_ix], y[test_ix]

            # configure the cross-validation procedure
            cv_inner = KFold(n_splits=3, shuffle=True, random_state=1)

            # define search
            search = GridSearchCV(model, grid, scoring='f1_weighted', cv=cv_inner, refit=True)
            # execute search
            result = search.fit(X_train, y_train)
            # get the best performing model fit on the whole training set
            best_model = result.best_estimator_
            # evaluate model on the hold out dataset
            yhat = best_model.predict(X_test)
            # evaluate the model
            f1 = f1_score(y_test, yhat, average='weighted')
            # store the result
            outer_results.append(f1)
            # report progress
            print('>f1=%.3f, est=%.3f, cfg=%s' % (f1, result.best_score_, result.best_params_))
            # summarize the estimated performance of the model

        if ('DecisionTree' in name):
            dot_data = StringIO()
            tree.export_graphviz(best_model, out_file=dot_data,  
                            filled=True, rounded=True,
                            special_characters=True, feature_names = features,class_names=list(set(yhat)))
            graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
            graph.write_png('dec_tree' + score_metric + '_' + str(search_budget) + '.png')
            Image(graph.create_png())

        # print('F1-score: %.3f (%.3f)' % (mean(outer_results), std(outer_results)))

        result = mean(outer_results)
        names.append(name)
        results.append(result)

    setting = score_metric + ' : ' + 'BUDGET ' + str(search_budget)
    print(setting + ' :' + str(results))

    results_dict = dict()

    for i in range(len(names)):
        results_dict[names[i]] = results[i].mean()

    with open(score_metric + str(search_budget)  + '_' + str(k) + '_features'+ '.txt', 'w') as file:
        file.write(json.dumps(results_dict)) 

    return results

if __name__ == '__main__':
    """
    Uses auto KFold with Grid Search for selected models.
    """
    for k in [3, 5, 10, 15, 20, 49]:
        # Branch coverage
        for search_budget in [60, 180, 300]:
            # Feature selection in CV
            double_K_fold(search_budget, columns_to_group=['TARGET_CLASS', 'configuration_id', 'project.id'], score_metric='BranchCoverage', score_metric_filename='res_data/results-', k=k)
            
            # Feature selection before CV
            # run_KFold_Grid_all_models(search_budget, columns_to_group=['TARGET_CLASS', 'configuration_id', 'project.id'], score_metric='BranchCoverage', score_metric_filename='res_data/results-')
        
        # Mutation score
        # Feature selection in CV
        double_K_fold(search_budget=60, columns_to_group=['class', 'configuration', 'project'], score_metric='mutation_score_percent', score_metric_filename='res_data/mutation_scores.csv', k=k)
    
    # Feature selection before CV
    # run_KFold_Grid_all_models(search_budget=60, columns_to_group=['class', 'configuration', 'project'], score_metric='mutation_score_percent', score_metric_filename='res_data/mutation_scores.csv')


    # KFold with no Grid Search
    # f1s_coverage = run_KFold(search_budgets=[60, 180, 300], columns_to_group=['TARGET_CLASS', 'configuration_id', 'project.id'], score_metric='BranchCoverage', score_metric_filename='res_data/results-')
    # print( 'DT: ' + str(np.average(f1s_coverage[60][0])) + ' SVC: ' + str(np.average(f1s_coverage[60][1])) + ' RF: ' + str(np.average(f1s_coverage[60][2])) + ' LR: ' + str(np.average(f1s_coverage[60][3])))
    # print( 'DT: ' + str(np.average(f1s_coverage[180][0])) + ' SVC: ' + str(np.average(f1s_coverage[180][1])) + ' RF: ' + str(np.average(f1s_coverage[180][2])) + ' LR: ' + str(np.average(f1s_coverage[180][3])))
    # print( 'DT: ' + str(np.average(f1s_coverage[300][0])) + ' SVC: ' + str(np.average(f1s_coverage[300][1])) + ' RF: ' + str(np.average(f1s_coverage[300][2])) + ' LR: ' + str(np.average(f1s_coverage[300][3])))

    # f1s_mutation_score = run_KFold(search_budgets=[60], columns_to_group=['class', 'configuration', 'project'], score_metric='mutation_score_percent', score_metric_filename='res_data/mutation_scores.csv')
    # print( 'DT_mutation: ' + str(np.average(f1s_mutation_score[60][0])) + ' SVC_mutation: ' + str(np.average(f1s_mutation_score[60][1])) + ' RF: ' + str(np.average(f1s_mutation_score[60][2])) + ' LR: ' + str(np.average(f1s_mutation_score[60][3])))
