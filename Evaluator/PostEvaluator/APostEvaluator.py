import logging
from abc import ABCMeta, abstractmethod


class APostEvaluator(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def evaluate(self, dir, input, stream_size):
        """
        Evaluate specific evaluator on a given data stream
        :param dir: Directory of file path
        :param input: Path of input stream file (CSV format)
        :param stream_size: Size of stream (# of records), for calculating evaluation window size
        :return: Tuple containing output and evaluation measurement results
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_eval_measures(output, measures):
        """
        Read output file and extracts the evaluation measurement results
        :param output: Result output file (CSV format)
        :param measures: Measurement performance results to extract
        :return: Dictionary of performance results
        """
        raise NotImplementedError
