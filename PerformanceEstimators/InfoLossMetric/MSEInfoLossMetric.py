import logging
from PerformanceEstimators.InfoLossMetric.AInfoLossMetric import AInfoLossMetric
from Utils.ExceptionHandler import ExceptionHandler
from Utils.MetricsUtils import MetricsUtils


class MSEInfoLossMetric(AInfoLossMetric):
    """
    Class implementing a generic information loss metric that measures the mean squared error between original and anonymized records. This metric can be unbounded.
    MSE = SSE / (n) = (sigma(j=1..n -> ||x_j-c_j||**2) / (n),
    where c_j is the centroid of cluster to which x_j belongs, n is total number of records and m is number of attributes.
    Referenced in: V. Torra, "Information Loss: Evaluation andMeasures", Data Privacy: Foundations, New Developments and the Big Data Challenge, 2017
    """

    def __init__(self):
        super(MSEInfoLossMetric, self).__init__()
        self.logger = logging.getLogger(__name__)

        self.__qi_dimension = None

    def update_estimation(self, time, record_pair, cluster=None):
        """
        Update the SSE with arriving tuple. Calculate the squared error of the pair of original and anonymized records.
        Accumulate the calculated squared error.
        ||x_j-c_j||**2), where c_j is the centroid of cluster to which x_j belongs.
        :param record_pair: Pair of original record and its anonymization.
        :param cluster: Cluster from which the record is published (Default: not needed).
        :return: Return
        """
        original = record_pair.original_record.quasi_identifier
        anonymized = record_pair.anonymized_record.quasi_identifier

        self.processed_instances += 1
        if not self.__qi_dimension:
            self.__qi_dimension = len(original)
        last_error = self.current_metric
        square_error = MetricsUtils.distance(original, anonymized) ** 2

        self.current_metric += square_error
        self.incremental_metric = self.current_metric - last_error

        # Store the incremental error over time, for time series plot
        # self.monitor_overtime_change(cluster.W_curr.max_size, show_incremental=True)
        self.monitor_overtime_change(window_size=100, show_incremental=True)
        # timestamp = record_pair.anonymized_record.timestamp
        # point = (self.__N, self.__incremental_metric)
        # # point = (timestamp, self.__current_error)
        # if timestamp % cluster.W_curr.max_size == 0:
        #     self.__metric_over_time.append(point)

    @ExceptionHandler.handle_exception(ExceptionHandler.evaluation_message)
    def get_info_loss(self):
        """
        Get information loss on a given stream, normalized to the number of record pairs.
        MSE = SSE / (n) = (sigma(j=1..n -> ||x_j-c_j||**2) / (n)
        :return: Information loss (Mean sum of squared errors)
        """
        # info_loss = self.__current_error / (self.__N * self.__qi_dimension)
        info_loss = 0
        if self.processed_instances > 0:
            info_loss = self.current_metric / self.processed_instances
        self.logger.info("MSE InfoLoss Metric: %0.3f" % info_loss)
        # self.logger.info("MSE InfoLoss Metric: %0.4f" % self.__current_error)
        return info_loss
