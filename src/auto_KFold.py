import json
from numpy import mean
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.svm import SVC
import sklearn.feature_selection as fs
from make_dataset import get_balanced_X_y


def run_KFold_Grid_all_models(
    search_budget=60,
    columns_to_group=['TARGET_CLASS', 'configuration_id', 'project.id'],
    score_metric='BranchCoverage',
    score_metric_filename='res_data/results-'
):

    X, y, features = get_balanced_X_y(search_budget, columns_to_group, score_metric, score_metric_filename, labels_dict=None)
    # Feature selection
    # X = select_features(X, y, features)

    bk = fs.SelectKBest()
    bk.fit(X, y)
    X_transf = bk.transform(X)
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
        scores = cross_val_score(search, X_transf, y, scoring='f1_weighted', cv=cv_outer, n_jobs=-1)
        # report performance
        # print('F1-score: %.3f (%.3f)' % (mean(scores), std(scores)))

        result = mean(scores)
        names.append(name)
        results.append(result)

    setting = score_metric + ' : ' + 'BUDGET ' + str(search_budget)
    print(setting + ' :' + str(results))

    results_dict = dict()

    for i in range(len(names)):
        results_dict[names[i]] = results[i].mean()
    
    with open(score_metric + str(search_budget) + '.txt', 'w') as file:
        file.write(json.dumps(results_dict)) 

    return results