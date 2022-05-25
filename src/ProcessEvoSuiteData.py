import os
from statistics import median
import numpy as np
import pandas as pd
import functools as ft
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None  # default='warn'

class ProcessEvoSuiteData:
    """ Class that processes data from EvoSuite. """

    def __init__(self):
        self.name = "ProcessEvoSuiteData"

    def print_name(self):
        os.system(f"echo The name is: {self.name}")

    def calculate_medians60(self):
        """ Extract results from the EvoSuite results csv files with time bugdet 60. """

        # Read EvoSuite results
        res60 = pd.read_csv("res_data/results-60.csv")
        configuration_ids = ['weak_60', 'branch_60', 'default_60']
        
        # Sort by configuration
        res = res60.loc[:,['TARGET_CLASS', 'configuration_id', 'project.id', 'BranchCoverage']]
        result = res[res.apply(lambda row : row["configuration_id"] in configuration_ids, axis=1)]

        # Take medians of the 10 runs of EvoSuite per class
        medians = result.groupby(['TARGET_CLASS', 'configuration_id', 'project.id'])['BranchCoverage'].median()
        medians.to_csv('medians.csv')

        # Compute differences between BRANCH;WEAKMUTATION and BRANCH
        res_medians = pd.read_csv("medians.csv")
        res_medians.to_csv('res_medians.csv')
        return

    def run_statistical_tests(self, output_csv, search_budget, compared_function='branch'):
        res = pd.read_csv(output_csv)
        configuration_ids = ['weak_' + str(search_budget), 'branch_' + str(search_budget), 'default_' + str(search_budget)]

        # Sort by configuration
        res = res.loc[:,['TARGET_CLASS', 'configuration_id', 'project.id', 'BranchCoverage']]
        result = res[res.apply(lambda row : row["configuration_id"] in configuration_ids, axis=1)]

        # "weak" groups
        weak_result = result.loc[result['configuration_id'] == 'weak_' + str(search_budget)]
        weak_groups = weak_result.groupby(['TARGET_CLASS', 'configuration_id', 'project.id'])['BranchCoverage']

        # "branch" groups
        compared_func_result = result.loc[result['configuration_id'] == str(compared_function) + '_' + str(search_budget)]
        compared_func_groups = compared_func_result.groupby(['TARGET_CLASS', 'configuration_id', 'project.id'])['BranchCoverage']

        weak_classes = dict()
        for name, group in weak_groups:
            weak_classes[name] = np.array(group)

        compared_func_classes = dict()
        for name, group in compared_func_groups:
            compared_func_classes[name] = np.array(group)

        # If there are less than 10 runs for a class, pad the branch coverage with 0 
        def pad(val1, val2):
            if val1.shape[0] == val2.shape[0]:
                return val1, val2
            if val1.shape[0] < val2.shape[0]:
                return np.pad(val1, [(0, val2.shape[0]-val1.shape[0])]), val2
            else:
                return val1, np.pad(val2, [(0, val1.shape[0]-val2.shape[0])])

        # Calculate statistical significance per class
        class_stats = dict()
        for ((key1, val1), (key2, val2)) in zip(weak_classes.items(), compared_func_classes.items()):
            val1, val2 = pad(val1, val2)
            stats, p = (-2,-2) if (np.sum(np.subtract(val1, val2)) == 0) else wilcoxon(val1, val2)
            class_stats[key1[0]] = p
            # print(str(key1) + str(val1) + ", "+ str(key2) + str(val2) + " HAS P VALUE OF: " + str(p))

        significant_class_stats = dict()
        for (key, pval) in class_stats.items():
            if (pval > -2 and pval < 0.05):
                significant_class_stats[key] = pval 
            
        statistical_sign_nclasses_60 = len(significant_class_stats)
        cl_stats_len = (len(class_stats))
        
        return statistical_sign_nclasses_60, cl_stats_len

    def plot_stats(self):

        data = ProcessEvoSuiteData()
        res_dict = {}
        res_dict["results60"] = data.calculate_medians60()
        res_dict["stats60"] = data.run_statistical_tests("res_data/results-60.csv", 60)
        res_dict["stats180"] = data.run_statistical_tests("res_data/results-180.csv", 180)
        res_dict["stats300"] = data.run_statistical_tests("res_data/results-300.csv", 300)

        res_dict["stats60_default"] = data.run_statistical_tests("res_data/results-60.csv", 60, 'default')
        res_dict["stats180_defaults"] = data.run_statistical_tests("res_data/results-180.csv", 180, 'default')
        res_dict["stats300_defaults"] = data.run_statistical_tests("res_data/results-300.csv", 300, 'default')

        # x axis
        print(res_dict)

        height = [res_dict["stats60_default"][1] - res_dict["stats60_default"][0], res_dict["stats60_default"][0], res_dict["stats180_defaults"][1] - res_dict["stats180_defaults"][0], res_dict["stats180_defaults"][0], res_dict["stats300_defaults"][1] - res_dict["stats300_defaults"][0], res_dict["stats300_defaults"][0]])

        # corresponding y axis values
        bars = [60,60,180,180,300,300]
        
        # create a dataset
        x_pos = np.arange(len(bars))

        # Create bars
        plt.bar(x_pos, height, color=['black', 'black', 'purple', 'purple', 'blue', 'blue'])

        # naming the x axis
        plt.xlabel('Number of non-statistically & statistically significant classes')
        # naming the y axis
        plt.ylabel('Time budget')
        
        # giving a title to my graph
        plt.title('Number of statistically significant classes per time budget')

        # Create names on the x-axis
        plt.xticks(x_pos, bars)

        plt.show()

    def get_ck_metrics(self):
        """ Will be used as features for the model. """

        class_metrics = pd.read_csv("ck_data/class.csv")
        class_metrics = class_metrics.iloc[:, 1:]
        return class_metrics
    

if __name__ == '__main__':
    data = ProcessEvoSuiteData()
    res_dict = {}
    res_dict["results60"] = data.calculate_medians60()
    res_dict["stats60"] = data.run_statistical_tests("res_data/results-60.csv", 60)
    res_dict["stats180"] = data.run_statistical_tests("res_data/results-180.csv", 180)
    res_dict["stats300"] = data.run_statistical_tests("res_data/results-300.csv", 300)

    res_dict["stats60_default"] = data.run_statistical_tests("res_data/results-60.csv", 60, 'default')
    res_dict["stats180_defaults"] = data.run_statistical_tests("res_data/results-180.csv", 180, 'default')
    res_dict["stats300_defaults"] = data.run_statistical_tests("res_data/results-300.csv", 300, 'default')

    print(res_dict)

    data.plot_stats([res_dict["stats60_default"][1] - res_dict["stats60_default"][0], res_dict["stats60_default"][0], res_dict["stats180_defaults"][1] - res_dict["stats180_defaults"][0], res_dict["stats180_defaults"][0], res_dict["stats300_defaults"][1] - res_dict["stats300_defaults"][0], res_dict["stats300_defaults"][0]])
