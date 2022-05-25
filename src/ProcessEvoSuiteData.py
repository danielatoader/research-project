import os
from statistics import median
import numpy as np
import pandas as pd
import functools as ft
from scipy.stats import wilcoxon

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

    def run_statistical_tests(self, output_csv, search_budget):
        res60 = pd.read_csv(output_csv)
        configuration_ids = ['weak_' + str(search_budget), 'branch_' + str(search_budget), 'default_' + str(search_budget)]

        # Sort by configuration
        res = res60.loc[:,['TARGET_CLASS', 'configuration_id', 'project.id', 'BranchCoverage']]
        result = res[res.apply(lambda row : row["configuration_id"] in configuration_ids, axis=1)]

        # "weak" groups
        weak_result = result.loc[result['configuration_id'] == 'weak_' + str(search_budget)]
        weak_groups = weak_result.groupby(['TARGET_CLASS', 'configuration_id', 'project.id'])['BranchCoverage']

        # "branch" groups
        branch_result = result.loc[result['configuration_id'] == 'branch_' + str(search_budget)]
        branch_groups = branch_result.groupby(['TARGET_CLASS', 'configuration_id', 'project.id'])['BranchCoverage']

        weak_classes = dict()
        for name, group in weak_groups:
            weak_classes[name] = np.array(group)

        branch_classes = dict()
        for name, group in branch_groups:
            branch_classes[name] = np.array(group)

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
        for ((key1, val1), (key2, val2)) in zip(weak_classes.items(), branch_classes.items()):
            val1, val2 = pad(val1, val2)
            stats, p = (-2,-2) if (np.sum(np.subtract(val1, val2)) == 0) else wilcoxon(val1, val2)
            class_stats[key1[0]] = p
            # print(str(key1) + str(val1) + ", "+ str(key2) + str(val2) + " HAS P VALUE OF: " + str(p))

        significant_class_stats = dict()
        for (key, pval) in class_stats.items():
            if (pval > -2 and pval < 0.05):
                significant_class_stats[key] = pval 
            
        statistical_sign_nclasses_60 = len(significant_class_stats)
        cl_st60 = (len(class_stats))
        print(statistical_sign_nclasses_60)
        print(cl_st60)

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