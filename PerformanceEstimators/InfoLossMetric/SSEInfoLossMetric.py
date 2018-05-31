import logging
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from PerformanceEstimators.InfoLossMetric.AInfoLossMetric import AInfoLossMetric
from Utils.ExceptionHandler import ExceptionHandler
from Utils.MetricsUtils import MetricsUtils


class SSEInfoLossMetric(AInfoLossMetric):
    """
    Class implementing a generic information loss metric for calculating the sum of square errors
    between original and anonymized records. This metric can be unbounded.
    SSE = (sigma(j=1..n -> ||x_j-c_j||**2), where c_j is the centroid of cluster to which x_j belongs.
    Referenced in: V. Torra, "Information Loss: Evaluation andMeasures", Data Privacy: Foundations, New Developments and the Big Data Challenge, 2017
    """

    def __init__(self):
        super(SSEInfoLossMetric, self).__init__()
        self.logger = logging.getLogger(__name__)

    def update_estimation(self, time, record_pair, cluster=None):
        """
        Update the SSE with arriving tuple. Calculate the squared error of the pair of original and anonymized records.
        Accumulate the calculated squared error.
        ||x_j-c_j||**2), where c_j is the centroid of cluster to which x_j belongs.
        :param record_pair: Pair of original record and its anonymization.
        :param cluster: Cluster from which the record is published (Default: not needed).
        :return: None
        """
        self.processed_instances += 1
        original = record_pair.original_record.quasi_identifier
        anonymized = record_pair.anonymized_record.quasi_identifier
        last_error = self.current_metric
        # square_error = MetricsUtils.distance(original, anonymized) ** 2
        square_error = MetricsUtils.distance_unbounded(original, anonymized)

        self.current_metric += square_error
        self.incremental_metric = self.current_metric - last_error
        # self.monitor_overtime_change(cluster.W_curr.max_size, show_incremental=False)
        self.monitor_overtime_change(window_size=100, show_incremental=False)

    @ExceptionHandler.handle_exception(ExceptionHandler.evaluation_message)
    def get_info_loss(self):
        """
        Get information loss on a given stream - Sum of square errors (SSE)
        :return: Information loss (Sum of squared errors)
        """
        self.logger.info("SSE InfoLoss Metric: %0.3f" % self.current_metric)
        return self.current_metric
