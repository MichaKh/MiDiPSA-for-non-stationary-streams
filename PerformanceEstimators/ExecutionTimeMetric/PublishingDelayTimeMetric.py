import logging
from PerformanceEstimators.ExecutionTimeMetric.AExecutionTimeMetric import AExecutionTimeMetric


class PublishingDelayTimeMetric(AExecutionTimeMetric):
    """
    Class implementing a publishing delay estimator, measuring the average delay of published tuples
    """

    def __init__(self):
        """
        Class constructor - initialization
        """
        super(PublishingDelayTimeMetric, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.accumulated_tuple_delay_time = self.current_metric

    def update_estimation(self, time, record_pair, cluster=None):
        """
        Update the publishing delay of current published tuple.
        :param time: Current time step in stream (last assigned tuple).
        :param record_pair: Pair of original instance and its anonymization.
        :param cluster: Cluster from which the record is published (Default: not needed).
        :return:
        """
        self.processed_instances += 1
        last_delay = self.accumulated_tuple_delay_time
        tuple_timestamp = record_pair.anonymized_record.timestamp
        self.accumulated_tuple_delay_time += time - tuple_timestamp

        self.incremental_metric = self.accumulated_tuple_delay_time - last_delay
        self.current_metric = self.accumulated_tuple_delay_time / self.processed_instances
        # Store the incremental delay over time, for time series plot
        # self.monitor_overtime_change(cluster.W_curr.max_size, show_incremental=False)
        self.monitor_overtime_change(window_size=100, show_incremental=False)

    def get_exec_time(self):
        """
        Get average publishing delay of all the tuples in the stream.
        :return: Average publishing delay.
        """
        average_delay_time = round(float(self.accumulated_tuple_delay_time) / self.processed_instances, 3)
        self.logger.info("Average Stream Publishing Delay Metric: %0.3f" % average_delay_time)
        return average_delay_time
