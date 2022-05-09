import os
import string
import subprocess
import fnmatch
import sys

class ExtractMetrics:
    """ Class that extracts metrics from projects under /benchmark/projects """

    def __init__(self, project_path="/home/daniela/rp-cse3000/benchmark/projects"):
        self.name = "ExtractMetrics"
        self.root_dir = os.path.abspath(os.curdir)
        self.project_path = project_path

    def printName(self):
        os.system(f"echo The name is: {self.name}")

    def calculateMetrics(self, project_jar_paths=None, command=None):
        """ Given the project name create the run metrics command, and call the run function. """

        # If projects and command are not predefined
        if not command:
            if not project_jar_paths:
                # Get all jar projects under self.project_path
                project_jar_paths = self.get_jars()
        
            # Run ckjm_ext for all projects under self.project_path
            for project_jar in project_jar_paths:
                command = f"java -jar ckjm_ext.jar {project_jar}"

                # Get filename from command
                filename = command.rsplit('/',1)[1]

                command = command.split()
                self.runAndWriteToFile(command, filename=f"{filename}.txt")

        # Else run predefined command
        else:
            # Get filename from command
            filename = command.rsplit('/',1)[1]
            command = command.split()
            self.runAndWriteToFile(command, filename=f"{filename}.txt")

    def runAndWriteToFile(self, command=['ls', '-l'], filename="metrics.txt"):
        """ Get the command line output and paste it in a text file. """
        
        with open(filename, 'w') as out:
                return_code = subprocess.call(command, stdout=out)

        # try:
        #     original_umask = os.umask(0)
        #     filename = "metrics/" + filename
        #     os.makedirs(os.path.dirname(filename), mode=0o777, exist_ok=False)
        #     with open(filename, 'w') as out:
        #         return_code = subprocess.call(command, stdout=out)
        # finally:
        #     os.umask(original_umask)

    def get_jars(self):
        """ Extract jars from the projects at project_path. """

        jars = []
        
        for file in os.listdir(self.project_path):
            d = os.path.join(self.project_path, file)
            if os.path.isdir(d):
                for file in os.listdir(str(d)):
                    if fnmatch.fnmatch(file,'*.jar'):
                        jars.append(os.path.join(d, file))
        return jars

    def get_class_metric_values(self):
        """ Get metrics values after running the analysis tool. """

        metrics_folder = self.root_dir + "/metrics/"
        # for file in os.listdir(metrics_folder):
        #     d = os.path.join(metrics_folder, file)
        #     if not os.path.isdir(d):
        #         ...

        project_metrics = os.path.join(metrics_folder, 'tullibee.jar.txt')
        with open(project_metrics) as f:
            lines = f.readlines()
            for line in lines:
                metrics_class = line.partition("~")[0].replace("\n", "")
                vals = metrics_class.rsplit(' ', 1) 
                print(vals)
                

