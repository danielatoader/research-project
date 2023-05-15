from process_evosuite_data import ProcessEvoSuiteData

def plot_significant_classes():
    data = ProcessEvoSuiteData()

    # matching = pd.DataFrame.from_dict(data.get_matching_classes_metrics("res_data/results-60.csv", 60))
    # matching.head()

    medians = data.calculate_medians60()
    # print(medians)

    res_dict = {}

    # Contains statistically significant classes at [0] and all classes at [1]
    # res_dict['stats60'][0].items() contains pairs of type (class_name, p-value)
    res_dict["stats60"] = data.run_statistical_tests("res_data/results-60.csv", 60)
    res_dict["stats180"] = data.run_statistical_tests("res_data/results-180.csv", 180)
    res_dict["stats300"] = data.run_statistical_tests("res_data/results-300.csv", 300)

    res_dict["stats60_default"] = data.run_statistical_tests("res_data/results-60.csv", 60, 'default')
    res_dict["stats180_default"] = data.run_statistical_tests("res_data/results-180.csv", 180, 'default')
    res_dict["stats300_default"] = data.run_statistical_tests("res_data/results-300.csv", 300, 'default')

    # print(res_dict)

    # Plot number of statistically significant classes for (weak - branch) and (weak - default) - 2nd bar
    # along with the number of non statistical classes - in 1st bar
    x_vals_branch = [
                len(res_dict["stats60"][1]) - len(res_dict["stats60"][0]), len(res_dict["stats60"][0])
            , len(res_dict["stats180"][1]) - len(res_dict["stats180"][0]), len(res_dict["stats180"][0])
            , len(res_dict["stats300"][1]) - len(res_dict["stats300"][0]), len(res_dict["stats300"][0])
            ]

    x_vals_default = [
                len(res_dict["stats60_default"][1]) - len(res_dict["stats60_default"][0]), len(res_dict["stats60_default"][0])
            , len(res_dict["stats180_default"][1]) - len(res_dict["stats180_default"][0]), len(res_dict["stats180_default"][0])
            , len(res_dict["stats300_default"][1]) - len(res_dict["stats300_default"][0]), len(res_dict["stats300_default"][0])
            ]

    y_vals = [60,60,180,180,300,300]

    data.plot_stats(x_vals_branch, y_vals, comparison='weak vs branch')
    data.plot_stats(x_vals_default, y_vals, comparison='weak vs default')