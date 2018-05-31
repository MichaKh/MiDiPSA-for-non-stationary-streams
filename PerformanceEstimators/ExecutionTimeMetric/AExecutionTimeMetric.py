import logging
from abc import ABCMeta, abstractmethod
from PerformanceEstimators.AEstimator import AEstimator


class AExecutionTimeMetric(AEstimator):
    """
    Abstract class implementing Generic Execution Time estimators
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        super(AExecutionTimeMetric, self).__init__()
        self.logger = logging.getLogger(__name__)

    def __str__(self):
        return str(self.get_estimation())

    def get_estimation(self):
        """
        Get the final performance estimation of estimator
        :return: Performance metric
        """
        return self.get_exec_time()

    @abstractmethod
    def get_exec_time(self):
        """
        Get the current execution time performance estimation
        :return: Current execution time
        """
        raise NotImplementedError
