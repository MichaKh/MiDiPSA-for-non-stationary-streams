import logging
from abc import ABCMeta, abstractmethod
from PerformanceEstimators.AEstimator import AEstimator


class AInfoLossMetric(AEstimator):
    """
    Abstract class implementing Generic Information loss estimators
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        super(AInfoLossMetric, self).__init__()
        self.logger = logging.getLogger(__name__)

    def __str__(self):
        return str(self.get_estimation())

    def get_estimation(self):
        """
        Get the final performance estimation of estimator
        :return: Performance metric
        """
        return self.get_info_loss()

    @abstractmethod
    def get_info_loss(self):
        """
        Get the current information loss performance estimation
        :return: Current information loss
        """
        raise NotImplementedError
