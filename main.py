from ExtractMetrics import ExtractMetrics

if __name__ == '__main__':

    class_metrics = ExtractMetrics()
    class_metrics.printName()
    # class_metrics.calculateMetrics()
    class_metrics.get_class_metric_values()

    # class_metrics.calculateMetrics(command="java -jar ckjm_ext.jar /home/daniela/rp-cse3000/benchmark/projects/1_tullibee/tullibee.jar")
    # class_metrics.calculateMetrics(project_jar_paths=["/home/daniela/rp-cse3000/benchmark/projects/1_tullibee/tullibee.jar"])
    # class_metrics.get_jars()
