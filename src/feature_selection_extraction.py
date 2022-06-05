import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

def select_features(X_train, X_test, features):
    """
    Perform feature selection based on correlation coefficients.
    """

    df_train = pd.DataFrame(X_train, columns = features)
    df_test = pd.DataFrame(X_test, columns = features)
    
    corr_features = correlation(df_train, 0.2)
    # print(len(set(corr_features)))

    X_train_selected = (df_train.drop(corr_features,axis=1)).to_numpy()
    X_test_selected = (df_test.drop(corr_features,axis=1)).to_numpy()

    return X_train_selected, X_test_selected
    

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