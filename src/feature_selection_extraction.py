import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline

def select_features(X_train, y_train, features):
    """
    Perform feature selection based on correlation coefficients.
    """
    # X = pd.DataFrame(X_train, columns = features)
    # y = pd.DataFrame(y_train)

    # # Create and fit selector
    # selector = SelectKBest(f_regression, k=5)
    # selector.fit(X, y)
    # X_selected_features = selector.transform(X)

    # # Get columns to keep and create new dataframe with those only
    # cols = selector.get_support(indices=True)
    # # X_selected_features= X.iloc[:,cols]
    # print(cols)

    df_train = pd.DataFrame(X_train, columns = features)
    df_y = pd.DataFrame(y_train)
    
    corr_features = correlation(df_train, 0.3)
    selected_features = list(set(features) - set(corr_features))
    print('Number of selected features: ' + str(len(set(selected_features))))
    print(selected_features)

    X_selected_features = (df_train.drop(corr_features,axis=1)).to_numpy()
    # y_selected_features = (df_y.drop(corr_features,axis=1)).to_numpy()

    assert (len(selected_features) == (len(features) - len(corr_features)))


    return X_selected_features
    

# with the following function we can select highly correlated features
# it will remove the first feature that is correlated with anything other feature

def correlation(dataset, threshold):

    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr


def select_f():
    cv_estimator = RandomForestClassifier(random_state =42)
    X_train,X_test,Y_train,Y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    cv_estimator.fit(X_train, Y_train)
    cv_selector = RFECV(cv_estimator,cv= 5, step=1,scoring='accuracy')
    cv_selector = cv_selector.fit(X_train, Y_train)
    rfecv_mask = cv_selector.get_support() #list of booleans
    rfecv_features = []
    for bool, feature in zip(rfecv_mask, X_train.columns):
        if bool:
            rfecv_features.append(feature)