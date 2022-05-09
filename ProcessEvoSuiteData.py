import os
import pandas as pd
import functools as ft

class ProcessEvoSuiteData:
    """ Class that processes data from EvoSuite. """

    def __init__(self):
        self.name = "ProcessEvoSuiteData"

    def print_name(self):
        os.system(f"echo The name is: {self.name}")

    def calculate_medians60(self):
        """ Extract results from the EvoSuite results csv files with time bugdet 60. """

        res60 = pd.read_csv("res_data/results-60.csv")

        default1 = res60[res60["configuration_id"] == "default_60"]
        branch = res60[res60["configuration_id"] == "branch_60"]
        branch_weak_mutation = res60[res60["configuration_id"] == "weak_60"]

        dfs = [default1, branch, branch_weak_mutation]
        res_dt = ft.reduce(lambda left, right: pd.merge(left, right, on="configuration_id"), dfs)

        default_median = default1["BranchCoverage"].median()
        branch_median = branch["BranchCoverage"].median()
        branch_weak_mutation = branch_weak_mutation["BranchCoverage"].median()

        metrics_with_filenames = self.get_ck_metrics()
        metrics = metrics_with_filenames.iloc[:, 1:]
        
        matching_metrics = metrics[metrics["class"].isin(branch["TARGET_CLASS"])]
        match_branch = branch[branch["TARGET_CLASS"].isin(metrics["class"])].median

        # print(branch)
        print(matching_metrics)
        
        return default_median, branch_median, branch_weak_mutation

    def get_ck_metrics(self):
        class_metrics = pd.read_csv("ck_data/class.csv")
        # classes = class_metrics.iloc[:, 1:]
        return class_metrics
        
    
if __name__ == '__main__':
    data = ProcessEvoSuiteData()
    res_dict = {}

    res_dict["results60"] = data.calculate_medians60()
    # print("\n _60: ", res_dict["results60"])
    
    # res_dict["results180"] = data.calculateMedians180()
    # print("\n _180: ", res_dict["results180"])
    # res_dict["results300"] = data.calculateMedians300()
    # print("\n _300: ", res_dict["results300"])
    


# Same strategy for 180 and 300 ------

    # def calculateMedians180(self):
    #     """ Extract results from the EvoSuite results csv files with time bugdet 180. """

    #     res180 = pd.read_csv("res_data/results-180.csv")

    #     default1 = res180[res180["configuration_id"] == "default_180"]
    #     branch = res180[res180["configuration_id"] == "branch_180"]
    #     branch_weak_mutation = res180[res180["configuration_id"] == "weak_180"]

    #     default_median = default1["BranchCoverage"].median()
    #     branch_median = branch["BranchCoverage"].median()
    #     branch_weak_mutation = branch_weak_mutation["BranchCoverage"].median()

    #     return default_median, branch_median, branch_weak_mutation

    # def calculateMedians300(self):
    #     """ Extract results from the EvoSuite results csv files with time bugdet 300. """

    #     res300 = pd.read_csv("res_data/results-300.csv")

    #     default1 = res300[res300["configuration_id"] == "default_300"]
    #     branch = res300[res300["configuration_id"] == "branch_300"]
    #     branch_weak_mutation = res300[res300["configuration_id"] == "weak_300"]

    #     default_median = default1["BranchCoverage"].median()
    #     branch_median = branch["BranchCoverage"].median()
    #     branch_weak_mutation = branch_weak_mutation["BranchCoverage"].median()

    #     return default_median, branch_median, branch_weak_mutation