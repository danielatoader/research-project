import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from VD_A import VD_A

pd.options.mode.chained_assignment = None  # default='warn'

class ProcessEvoSuiteData:
    """
    Class that processes data from EvoSuite.
    """

    def __init__(self):
        self.name = "ProcessEvoSuiteData"

    def print_name(self):
        os.system(f"echo The name is: {self.name}")

    def calculate_medians(
            self,
            search_budget,
            columns_to_group,
            score_metric,
            score_metric_filename):
        """
        Extract results from the EvoSuite results csv files with given search bugdet.
        """

        # Read EvoSuite results
        if (not ('mutation' in score_metric_filename)):
            # Contains the coverage results per class from EvoSuite (e.g. coverage_res_filename = "res_data/results-60.csv")
            # 'res_data/results-' + str(search_budget) + '.csv'
            score_metric_filename = str(score_metric_filename) + str(search_budget) + '.csv'
            
        res_search_b = pd.read_csv(score_metric_filename)
        configuration_ids = [
            'weak_' + str(search_budget),
            'branch_' + str(search_budget),
            'default_' + str(search_budget)]
        all_columns = columns_to_group + [score_metric]

        # Configuration column name is the second in the columns_to_group list
        config_column = columns_to_group[1]

        # Sort by configuration
        res = res_search_b.loc[:, all_columns]
        result = res[res.apply(lambda row : row[config_column] in configuration_ids, axis=1)]

        # Take medians of the 10 runs of EvoSuite per class
        medians = result.groupby(columns_to_group)[score_metric].median()
        medians_filename = 'medians_' + str(score_metric) + '.csv'
        medians.to_csv(medians_filename)

        # Compute differences between BRANCH;WEAKMUTATION and BRANCH
        res_medians = pd.read_csv(medians_filename)
        
        return res_medians
    
    def get_ck_metrics(self):
        """
        Will be used as features for the model.
        """

        class_metrics = pd.read_csv("ck_data/class.csv")
        class_metrics = class_metrics.iloc[:, 1:]
        return class_metrics
    
    def get_matching_classes_metrics(
        self,
        output_csv,
        search_budget,
        columns_to_group,
        score_metric):
        """
        Matches what the ck tool measured on the SF110 with the classes in EvoSuite output files.
        """
        
        class_metrics = self.get_ck_metrics()

        # Features are the metrics themselves: cbo, loc, etc.
        features = np.array(class_metrics.columns.values)[2:]

        # Get data and columns to work with
        res = pd.read_csv(output_csv)
        configuration_ids = ['weak_' + str(search_budget), 'branch_' + str(search_budget), 'default_' + str(search_budget)]
        all_columns = columns_to_group + [score_metric]

        # Configuration column name is the second in the columns_to_group list
        config_column = columns_to_group[1]
        
        # Class column name is the first in the columns_to_group list
        class_column = columns_to_group[0]

        # Sort by configuration
        res = res.loc[:,all_columns]
        result = res[res.apply(lambda row : row[config_column] in configuration_ids, axis=1)]

        # Match classes from ck tool result with classes in res_data (EvoSuite's output)
        matching_classes = class_metrics[class_metrics['class'].isin(result[class_column])].drop_duplicates()
 
        return matching_classes

    def get_significant_classes_stats(
            self,
            output_csv,
            search_budget,
            compared_function,
            columns_to_group, 
            score_metric):
        """
        Get Wilcoxon and Vargha-Delaney statistics for all classes and return both significant and all classes along with their stats.
        """

        res = pd.read_csv(output_csv)
        configuration_ids = [
            'weak_' + str(search_budget),
            'branch_' + str(search_budget),
            'default_' + str(search_budget)]
        all_columns = columns_to_group + [score_metric]

        # Configuration column name is the second in the columns_to_group list
        config_column = columns_to_group[1]

        # Sort by configuration
        res = res.loc[:,all_columns]
        result = res[res.apply(lambda row : row[config_column] in configuration_ids, axis=1)]

        # "weak" groups
        weak_result = result.loc[result[config_column] == 'weak_' + str(search_budget)]
        weak_groups = weak_result.groupby(columns_to_group)[score_metric]

        # "compared" groups
        compared_func_result = result.loc[result[config_column] == str(compared_function) + '_' + str(search_budget)]
        compared_func_groups = compared_func_result.groupby(columns_to_group)[score_metric]

        # Create a dictionary with the 10 runs per class and the resulting branch coverage for the weak + branch fitness function
        weak_classes = dict()
        for name, group in weak_groups:
            weak_classes[name] = group.astype(float).to_numpy()

        # Create a dictionary with the 10 runs per class and the resulting branch coverage for the compared fitness function
        compared_func_classes = dict()
        for name, group in compared_func_groups:
            compared_func_classes[name] = group.astype(float).to_numpy()

        # If there are less than 10 runs for a class, pad the branch coverage with 0 
        def pad(val1, val2):
            if val1.shape[0] == val2.shape[0]:
                return val1, val2
            if val1.shape[0] < val2.shape[0]:
                return np.pad(val1, [(0, val2.shape[0]-val1.shape[0])]), val2
            else:
                return val1, np.pad(val2, [(0, val1.shape[0]-val2.shape[0])])

        # Calculate statistical significance per class
        # Apply the Wilcoxon test for (weak_classes, compared_classes) per batch of 10 runs
        # Apply A. Vargha and H. D. Delaney. per batch
        # Filter out non-significant classes and classes with less than 'large' effect size
        class_stats = dict()

        for ((key1, val1), (key2, val2)) in zip(weak_classes.items(), compared_func_classes.items()):

            val1, val2 = pad(val1, val2)

            # If both weak and compared score metrics are equal, then flag with (-2,-2)
            # Otherwise perform Wilcoxon and Vargha-Delaney
            stats_p = (-2,-2) if (np.sum(np.subtract(val1, val2)) == 0) else wilcoxon(val1, val2)
            vd = VD_A(val1.tolist(), val2.tolist())
            class_stats[key1[0]] = (stats_p, vd)
            # print(str(key1) + str(val1) + ", "+ str(key2) + str(val2) + " HAS P VALUE OF: " + str(p))

        significant_class_stats = dict()

        # Filter classes with 'p' < 0.05 and a large effect size 
        for (key, ((stats, p), vd)) in class_stats.items():
            if (p > -2 and p < 0.005 and vd[1] == 'large'):
                significant_class_stats[key] = ((stats, p), vd) 
        
        # Return statistically significant class stats & all classes stats
        return significant_class_stats, class_stats

    def get_significant_classes_matching_metrics(
            self,
            search_budget,
            columns_to_group,
            score_metric,
            score_metric_filename):
        """ 
        Get significant classes and their corresponding matching metrics.
        
        :param search_budget: Is None only for mutation, which is only ran with budget 60
        :param score_metric_filename: Is only changed for coverage, which is ran with budgets 60, 180 and 300

        :return signif_matched_metrics: a dictionary of statistically significant classes with their matched metrics corresponding to the CK tool.
        """

        significant_classes_stats = {}
        key_budget = 'stats' + str(search_budget)

        if (not ('mutation' in score_metric_filename)):
            # Contains the coverage results per class from EvoSuite (e.g. coverage_res_filename = "res_data/results-60.csv")
            # 'res_data/results-' + str(search_budget) + '.csv'
            score_metric_filename = str(score_metric_filename) + str(search_budget) + '.csv'

        # Contains statistically significant classes at [0] and all classes at [1]
        # res_dict['stats60'][0].items() contains pairs of type (class_name, p-value)
        significant_classes_stats[str(key_budget + '_branch')] = self.get_significant_classes_stats(
            score_metric_filename,
            search_budget,
            'branch',
            columns_to_group,
            score_metric)

        significant_classes_stats[str(key_budget + '_default')] = self.get_significant_classes_stats(
            score_metric_filename,
            search_budget,
            'default',
            columns_to_group,
            score_metric)

        matching_classes_metrics = {}
        signif_matched_metrics = {}
        key_class_metrics = 'class_metrics' + str(search_budget)

        # Contains the dictionary with matched metrics (for classes from EvoSuite output and CK tool)
        # Should probably be reindexed (or ignore index)
        matching_classes_metrics[key_class_metrics] = self.get_matching_classes_metrics(
            score_metric_filename,
            search_budget,
            columns_to_group,
            score_metric)

        # Contains names of statistically significant classes
        significant_classes = list(set(list(significant_classes_stats[str(key_budget + '_branch')][0].keys()) + list(significant_classes_stats[str(key_budget + '_branch')][0].keys())))

        # Matched metrics for the significant classes
        matched_metrics = matching_classes_metrics[key_class_metrics]
        signif_matched_metrics[key_class_metrics] = matched_metrics[matched_metrics.apply(lambda row : row['class'] in significant_classes, axis=1)]

        return signif_matched_metrics
