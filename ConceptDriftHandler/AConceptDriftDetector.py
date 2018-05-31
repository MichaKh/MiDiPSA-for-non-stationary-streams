import logging
from abc import ABCMeta, abstractmethod


class AConceptDriftDetector(object):
    """
    Abstract class of concept drift detector
    """
    __metaclass__ = ABCMeta

    def __init__(self, conf):
        self.logger = logging.getLogger(__name__)
        self.__confidence = conf

        self.incremental_change = 0.
        self.windows_processed = 0
        self.metric_over_time = []

    @property
    def confidence(self):
        return self.__confidence

    @abstractmethod
    def detect(self, c):
        """
        Detect concept drift in stream, given two consecutive windows of samples
        :param c: Cluster to be inspected for concept drift. Contains both current and previous buffer windows.
        :return: True, if drift is detected, otherwise False
        """
        raise NotImplementedError

    def get_threshold(self, n, m):
        """
        Define the cut threshold for detecting concept drift
        Can be defined as static threshold, or statistical adaptive or some other global boundary
        :param n: Size of first buffer sample
        :param m: Size of second buffer sample
        :return: Numeric positive threshold
        """
        raise NotImplementedError

    def monitor_overtime_change(self):
        """
        Monitor the detection of concept drift over time.
        Monitor the change in the distance between two consecutive buffers in the cluster for which the buffers are full.
        :return:
        """
        raise NotImplementedError
