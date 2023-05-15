import numpy as np
from extract_metrics import ExtractMetrics
from process_evosuite_data import ProcessEvoSuiteData

def get_significant_classes_matching_metrics_coverage(data):

    # Get significant classes metrics for branch coverage
    search_budgets=[60, 180, 300]
    columns_to_group=['TARGET_CLASS', 'configuration_id', 'project.id']
    score_metric='BranchCoverage'
    score_metric_filename='res_data/results-'

    print('BRANCH COVERAGE: ')
    res = {}
    for search_budget in search_budgets:

        metrics_budget = 'class_metrics' + str(search_budget)

        signif_matched_metrics = data.get_significant_classes_matching_metrics(
            search_budget,
            columns_to_group,
            score_metric,
            score_metric_filename)

        res[search_budget] = signif_matched_metrics
        signif_matched_metrics[metrics_budget].to_csv('significant_classes_with_stats_matched_coverage' + str(search_budget) + '.csv')
        print(len(signif_matched_metrics[metrics_budget]))
    
    # Features are the metrics themselves: cbo, loc, etc.
    # features = (signif_matched_metrics[metrics_budget].iloc[:,[0]]['class']).to_numpy()

    # print(signif_matched_metrics.keys())
    # print(features)

    return res

def get_significant_classes_matching_metrics_mutation_score(data):  
    """
    Get significant classes metrics for mutation score
    """

    search_budgets=[60]
    columns_to_group=['class', 'configuration', 'project']
    score_metric='mutation_score_percent'
    score_metric_filename='res_data/mutation_scores.csv'

    print('MUTATION SCORE: ')
    res = {}
    for search_budget in search_budgets:

        metrics_budget = 'class_metrics' + str(search_budget)

        signif_matched_metrics = data.get_significant_classes_matching_metrics(
            search_budget,
            columns_to_group,
            score_metric,
            score_metric_filename)
            
        res[search_budget] = len(signif_matched_metrics[metrics_budget])
        signif_matched_metrics[metrics_budget].to_csv('significant_classes_with_stats_matched_mutation_score' + str(search_budget) + '.csv')
        print(len(signif_matched_metrics[metrics_budget]))

    return res

if __name__ == '__main__':
    data = ProcessEvoSuiteData()
    get_significant_classes_matching_metrics_coverage(data)
    get_significant_classes_matching_metrics_mutation_score(data)

    # Calculate metrics using CK tools
    # class_metrics = ExtractMetrics()
    # class_metrics.printName()
    # class_metrics.calculateMetrics()
    # class_metrics.calculateMetrics(command="java -jar ckjm_ext.jar /home/daniela/rp-cse3000/benchmark/projects/1_tullibee/tullibee.jar")
