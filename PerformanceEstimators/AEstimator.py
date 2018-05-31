import logging
from abc import ABCMeta, abstractmethod


class AEstimator(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.processed_instances = 0
        self.incremental_metric = 0.
        self.current_metric = 0.
        self.metric_over_time = []

    @abstractmethod
    def update_estimation(self, time, record_pair, cluster=None):
        """
        Perform incremental update of the stream performance estimator
        :param time: Current time step in stream (last assigned tuple)
        :param record_pair: Pair of original record and its anonymization
        :param cluster: Cluster to which tuple belongs (Default: not needed).
        :return: None
        """
        raise NotImplementedError

    @abstractmethod
    def get_estimation(self):
        """
        Get the final performance estimation of estimator
        :return: Performance metric
        """
        raise NotImplementedError

    def monitor_overtime_change(self, window_size, show_incremental=True):
        """
        Monitor the incremental change in performance, over time.
        For example, reports the incremental change in information loss over time,
        sampling the performance using sliding windows of published tuples of given size.
        :param window_size: Size of sampling window.
        :param show_incremental: If True, reports the incremental change, otherwise report accumulated change (Default: True)
        :return: None
        """
        if show_incremental:
            point = (self.processed_instances, self.incremental_metric)
        else:
            point = (self.processed_instances, self.current_metric)
        if self.processed_instances % window_size == 0:
            self.metric_over_time.append(point)
