import logging
from abc import ABCMeta, abstractmethod
from PerformanceEstimators.AEstimator import AEstimator


class ADisclosureRiskMetric(AEstimator):
    """
    Abstract class implementing disclosure risk estimators
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        super(ADisclosureRiskMetric, self).__init__()
        self.logger = logging.getLogger(__name__)

    def __str__(self):
        return str(self.get_estimation())

    def get_estimation(self):
        """
        Get the final performance estimation of estimator
        :return: Performance metric
        """
        return self.get_disclosure_risk()

    @abstractmethod
    def get_disclosure_risk(self):
        """
        Get the current disclosure risk performance estimation
        :return: Current disclosure risk
        """
        raise NotImplementedError
