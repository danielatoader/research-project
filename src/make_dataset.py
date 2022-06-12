import json
import os
import numpy as np
import pandas as pd
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

def get_X_y(search_budget, columns_to_group, score_metric, score_metric_filename, labels_dict=None, read_from=None):
    """
    Gets the dataset and the taget labels to further train the ML models.
    """

    medians_columns = [columns_to_group[0], columns_to_group[1], score_metric]
    class_column = columns_to_group[0]
    metrics_budget = 'class_metrics' + str(search_budget)
    from_col = 2

    if (read_from):
        signif_matched_metrics = pd.read_csv(read_from)
        from_col = 3
    else:
        filename_significant_matched_metrics = 'significant_matched_metrics_' + score_metric + '_' + str(search_budget) + '.csv'
        # filename_significant_matched_metrics = 'significant_matched_metrics_' + score_metric + '_' + str(search_budget) + '.txt'
  
        outdir = './significant_metrics_matched/'
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        filename_significant_matched_metrics = os.path.join(outdir, filename_significant_matched_metrics)  

        signif_matched_metrics = {}
        if (os.path.exists(filename_significant_matched_metrics)):
            signif_matched_metrics1 = pd.read_csv(filename_significant_matched_metrics)
            # with open(filename_significant_matched_metrics) as json_file:
                # signif_matched_metrics = json.load(json_file)
            signif_matched_metrics[metrics_budget] = signif_matched_metrics1
            from_col = 3
        else:
            signif_matched_metrics = data.get_significant_classes_matching_metrics(
                search_budget,
                columns_to_group,
                score_metric,
                score_metric_filename)

            # with open(filename_significant_matched_metrics, 'w') as file:
            #     file.write(json.dumps(signif_matched_metrics)) 
            signif_matched_metrics[metrics_budget].to_csv(filename_significant_matched_metrics)  

    medians = data.calculate_medians(search_budget, columns_to_group, score_metric, score_metric_filename)[medians_columns]

    X = []
    classes = []
    # from_col = 2
    features = signif_matched_metrics[metrics_budget].keys()[from_col:]

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
        for feature in signif_matched_metrics[metrics_budget].keys()[from_col:]:
            
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

    X = np.array(X)
    y = np.array(y)

    return X, y, features

def get_balanced_X_y(search_budget, columns_to_group,  score_metric, score_metric_filename, labels_dict=None):
    """
    Get X and y and also apply data balancing techniques.
    """
    
    # Get samples and their labels
    X, y, features = get_X_y(search_budget, columns_to_group, score_metric, score_metric_filename, labels_dict=None)

    # Apply dataset balancing techniques
    over_sampler = RandomOverSampler(random_state=42)
    X_res, y_res = over_sampler.fit_resample(X, y)

    # under_sampler = RandomUnderSampler(random_state=42)
    # X_res, y_res = under_sampler.fit_resample(X, y)

    return X_res, y_res, features