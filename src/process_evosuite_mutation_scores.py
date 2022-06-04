import os
from statistics import median
from unittest.mock import NonCallableMock
import numpy as np
import pandas as pd
import functools as ft
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
from VD_A import VD_A

pd.options.mode.chained_assignment = None  # default='warn'

class ProcessEvoSuiteMutationScores:
    """ Class that processes mutation score data from EvoSuite. """

    def __init__(self):
        self.name = "ProcessEvoSuiteMutationScores"

    def print_name(self):
        os.system(f"echo The name is: {self.name}")

    def calculate_medians(self, search_budget=60):
        """ Extract results from the EvoSuite results csv files with given search bugdet. """

        # Read EvoSuite results
        mutation_scores_filename = 'res_data/mutation_scores.csv'
        res_search_b = pd.read_csv(mutation_scores_filename)
        configuration_ids = ['weak_' + str(search_budget), 'branch_' + str(search_budget), 'default_' + str(search_budget)]
        
        # Sort by configuration
        res = res_search_b.loc[:,['class', 'configuration', 'project', 'mutation_score_percent']]
        result = res[res.apply(lambda row : row["configuration"] in configuration_ids, axis=1)]

        # Take medians of the 10 runs of EvoSuite per class
        medians = result.groupby(['class', 'configuration', 'project'])['mutation_score_percent'].median()
        medians.to_csv('medians_mutation_score.csv')

        res_medians = pd.read_csv('medians_mutation_score.csv')
        res_medians.to_csv('medians_mutation_score.csv')
        return res_medians

    def get_significant_classes_stats(self, output_csv, search_budget, compared_function):
        res = pd.read_csv(output_csv)
        configuration_ids = ['weak_' + str(search_budget), 'branch_' + str(search_budget), 'default_' + str(search_budget)]

        # Sort by configuration
        res = res.loc[:,['class', 'configuration', 'project', 'mutation_score_percent']]
        result = res[res.apply(lambda row : row['configuration'] in configuration_ids, axis=1)]

        # "weak" groups
        weak_result = result.loc[result['configuration'] == 'weak_' + str(search_budget)]
        weak_groups = weak_result.groupby(['class', 'configuration', 'project'])['mutation_score_percent']

        # "compared" groups
        compared_func_result = result.loc[result['configuration'] == str(compared_function) + '_' + str(search_budget)]
        compared_func_groups = compared_func_result.groupby(['class', 'configuration', 'project'])['mutation_score_percent']

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
            stats_p = (-2,-2) if (np.sum(np.subtract(val1, val2)) == 0) else wilcoxon(val1, val2)
            vd = VD_A(val1.tolist(), val2.tolist())
            class_stats[key1[0]] = (stats_p, vd)
            # print(str(key1) + str(val1) + ", "+ str(key2) + str(val2) + " HAS P VALUE OF: " + str(p))

        significant_class_stats = dict()
        for (key, ((stats, p), vd)) in class_stats.items():
            if (p > -2 and p < 0.05 and vd[1] == 'large'):
                significant_class_stats[key] = ((stats, p), vd) 
        
        # Return statistically significant class stats & all classes stats (stats = p-values)
        return significant_class_stats, class_stats

    def get_ck_metrics(self):
        """ Will be used as features for the model. """

        class_metrics = pd.read_csv("ck_data/class.csv")
        class_metrics = class_metrics.iloc[:, 1:]
        return class_metrics
    
    def get_matching_classes_metrics(self, output_csv, search_budget):
        """ Matches what the ck tool measured on the SF110 with the classes in EvoSuite output files. """
        
        class_metrics = self.get_ck_metrics()

        # Features are the metrics themselves: cbo, loc, etc.
        features = np.array(class_metrics.columns.values)[2:]

        res = pd.read_csv(output_csv)
        configuration_ids = ['weak_' + str(search_budget), 'branch_' + str(search_budget), 'default_' + str(search_budget)]

        # Sort by configuration
        res = res.loc[:,['class', 'configuration', 'project', 'mutation_score_percent']]
        result = res[res.apply(lambda row : row['configuration'] in configuration_ids, axis=1)]

        # Match classes from ck tool result with classes in res_data (EvoSuite's output)
        matching_classes = class_metrics[class_metrics['class'].isin(result['class'])].drop_duplicates()
        # print(len(matching_classes))
        return matching_classes   
    
# if __name__ == '__main__':
#     data = ProcessEvoSuiteData()
#     res_dict = {}
#     res_dict["results60"] = data.calculate_medians60()
#     res_dict["stats60"] = data.get_significant_classes_stats("res_data/results-60.csv", 60)
#     res_dict["stats180"] = data.get_significant_classes_stats("res_data/results-180.csv", 180)
#     res_dict["stats300"] = data.get_significant_classes_stats("res_data/results-300.csv", 300)

#     res_dict["stats60_default"] = data.get_significant_classes_stats("res_data/results-60.csv", 60, 'default')
#     res_dict["stats180_defaults"] = data.get_significant_classes_stats("res_data/results-180.csv", 180, 'default')
#     res_dict["stats300_defaults"] = data.get_significant_classes_stats("res_data/results-300.csv", 300, 'default')

    # print(res_dict)
            
        # TODO there are classes that do not come from SF110 in the data from supervisors => rerun tool to cover those
        # then retrain model