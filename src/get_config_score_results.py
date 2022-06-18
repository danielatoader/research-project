import numpy as np
import pandas as pd


def get_config_results(
        output_csv,
        search_budget,
        compared_function,
        columns_to_group,
        score_metric):
        """
        Gets the score metric results (branch coverage/mutation score) from the specified csv file.
        """
        
        # Get data and columns to work with
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

       