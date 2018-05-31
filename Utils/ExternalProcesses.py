import os
import subprocess
from Utils.ExceptionHandler import ExceptionHandler


class ExternalProcesses(object):
    """
    Utility helper class for generating subprocessing for comunication with external java libraries
    """
    # Path of external libraries and files
    EXTERNAL_LIB = os.path.abspath('ExternalLib')

    # Size of memory buffer used by MOA software for reading datasets and writing them to files
    BUFFER_SIZE = '1000000'

    # Path of javaagent jar file (file: sizeofag file)
    MOA_JAVA_AGENT = os.path.abspath(os.path.join('ExternalLib\\lib', 'sizeofag-1.0.0.jar'))

    # Path of Weka jar file, for preprocessing and creation of ARFF files
    WEKA_PATH = os.path.abspath(os.path.join('ExternalLib', 'weka.jar'))

    # Path of Moa jar file, for executing stream tasks
    MOA_PATH = os.path.abspath(os.path.join('ExternalLib', 'moa.jar'))

    @staticmethod
    @ExceptionHandler.handle_exception(ExceptionHandler.external_lib_message)
    def run_process(args):
        """
        Run a subprocess from specified working directory on an external MOA/WEKA java files
        :param args: Arguments of the process call
        :return: None
        """
        a = ExternalProcesses.initiate_process(args)
        # try:
        subprocess.call(a, cwd=ExternalProcesses.EXTERNAL_LIB, shell=True)
        return

    @staticmethod
    def initiate_process(args):
        """
        Parse the arguments of the MOA/WEKA process and initiate the list of commands for calling the process:
        -l --> MOA learner
        -s --> MOA arff stream file
        -e --> MOA classification evaluator
        -o --> MOA classification predictions output file
        -B --> MOA buffer size
        > --> Result output file path
        :param args: Arguments of the process call
        :return: List of command arguments
        """
        for name, value in args.items():
            # Type of process call (e.g., Java call)
            if name == 'p_type':
                a = ['java', '-Xmx8G', '-cp']
            # jar file for execution (e.g., weka.jar)
            elif name == 't_type' and value == 'weka':
                a.append(ExternalProcesses.WEKA_PATH)
            elif name == 't_type' and value == 'moa':
                a.append(ExternalProcesses.MOA_PATH)
            # javaagent file for moa.jar
            elif name == 'javaagent':
                a.append('-' + name + ':' + ExternalProcesses.MOA_JAVA_AGENT)
            # Task to execute (e.g., evaluate prequential accuracy)
            elif name == 'task':
                a.extend(['moa.DoTask', value])
            # Class of task or path of file
            elif name in ['jclass', 'path']:
                a.append(str(value))

            # Stream file
            elif name == 's' and os.path.isfile(value):
                a.extend(['-' + name, '(ArffFileStream', '-f', str(value) + ')'])
            # output operator to file
            elif name == 'gt':
                a.extend(['>', str(value)])
            # output classification predictions to file
            elif name == 'pred':
                a.extend(['-o', str(value)])
            # Buffer size for WEKA dataset readers
            elif name == 'B':
                a.extend(['-' + name, ExternalProcesses.BUFFER_SIZE])
            else:
                a.extend(['-' + name, str(value)])
        return a
