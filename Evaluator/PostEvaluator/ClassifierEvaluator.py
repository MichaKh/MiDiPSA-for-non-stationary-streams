import os
import pandas as pd
from collections import OrderedDict
from Evaluator.PostEvaluator.APostEvaluator import APostEvaluator
from Utils.ExceptionHandler import ExceptionHandler
from Utils.ExternalProcesses import ExternalProcesses



class ClassifierEvaluator(APostEvaluator):
    """
    Class initiating a MOA classifier for evaluating the classification accuracy of data stream
    """
    MEASUREMENTS = ['AUC']# 'classifications correct (percent)',
                    # 'Recall (percent)',
                    # 'Precision (percent)',
                    # 'F1 Score (percent)',
                    # 'Kappa Statistic (percent)']

    def __init__(self, task, learner, evaluator):
        """
        Class constructor - initiate evaluation classifier object
        :param task: Task of evaluation (e.g., name: EvaluatePrequential, type:EvaluatePrequential)
        :param learner: Type of classification model learner algorithm (e.g., name: HoeffdingTree, type:trees.HoeffdingTree)
        :param evaluator: Evaluation metric (e.g., name: WindowEvaluator, type: WindowClassificationPerformanceEvaluator -w 100 -o)
        """
        super(ClassifierEvaluator, self).__init__()

        self.task_name, self.task_type = task
        self.learner_name, self.learner_type = learner
        self.evaluator_name, self.evaluator_type = evaluator

    @ExceptionHandler.handle_exception(ExceptionHandler.external_lib_message)
    def evaluate(self, dir, input, stream_size):
        """
        Evaluate a stream classifier from MOA on a given data stream
        :param dir: Directory of file path
        :param input: Path of input stream file (CSV format)
        :param stream_size: Size of stream (# of records), for calculating evaluation window size
        :return: Tuple containing predictions file path and evaluation measurement results
        """
        self.logger.info("Executing external subprocess to MOA: %s ON %s USING %s EVALUATED BY %s" % (self.task_name,
                                                                                                      input,
                                                                                                      self.learner_name,
                                                                                                      self.evaluator_name))
        # input_path = StreamWriter.convert_CSV_to_ARFF(dir, input)
        file_name = '{0}_{1}_{2}'.format(self.task_name, self.learner_name, "_".join(input.split('.')[0:-1]))
        input_path = os.path.abspath(os.path.join(dir, input))
        pred_path = os.path.abspath(os.path.join(dir, '{0}{1}'.format(file_name, '.pred')))
        out_path = os.path.abspath(os.path.join(dir, '{0}{1}'.format(file_name, '.csv')))

        measurements = self.run_evaluation(stream=input_path,
                                           stream_size=stream_size,
                                           task=self.task_type,
                                           learner=self.learner_type,
                                           evaluator=self.evaluator_type,
                                           pred_output=pred_path,
                                           res_output=out_path)

        return pred_path, measurements

    def run_evaluation(self, stream, stream_size, task, learner, evaluator, pred_output, res_output):
        """
        Execute evaluation of classification task on the data with specified learner and performance evaluator

        :param stream: ARFF file containing stream data
        :param stream_size: size fo stream (number of records)
        :param task: Task of evaluation (e.g., EvaluatePrequential)
        :param learner: Type of classification model learner algorithm (e.g., HoeffdingTree)
        :param evaluator: Evaluation metric (e.g., WindowClassificationPerformanceEvaluator)
        :param pred_output:  Path of predictions output file (with predicted class and real class)
        :param res_output: Path of result output file (with evaluation results)
        :return: Classification accuracy of classifier
        """
        ExternalProcesses.run_process(OrderedDict([
            ('p_type', 'java'),
            ('t_type', 'moa'),
            ('javaagent', ''),
            ('task', task),
            ('l', learner),
            ('s', stream),
            ('e', evaluator),
            ('f', int(0.05 * stream_size)),  # sample frequency = how many instances between samples of the performance
            ('pred', pred_output),
            ('gt', res_output)]))
        out_path = os.path.abspath(res_output)

        measures_dict = self.get_eval_measures(output=out_path,
                                               measures=ClassifierEvaluator.MEASUREMENTS)
        self.logger.info(", ".join(['{}={}'.format(*p) for p in measures_dict.items()]))

        self.logger.info("Subprocess output is written to: %s" % out_path)

        return measures_dict if measures_dict else None

    @staticmethod
    def get_eval_measures(output, measures):
        """
        Read MOA output file and extracts the classification measurements
        :param output: MOA result output file (CSV format)
        :param measures: Measurement performance results
        :return: Dictionary of performance results
        """
        eval_dict = OrderedDict()  # store {measure_name: measure score}
        if os.stat(output).st_size > 0:
            df = pd.read_csv(output)
            for measure in measures:
                if measure in df.columns:
                    eval_dict[measure] = df[measure].iloc[-1]
                else:
                    eval_dict[measure] = '?'
        return eval_dict
