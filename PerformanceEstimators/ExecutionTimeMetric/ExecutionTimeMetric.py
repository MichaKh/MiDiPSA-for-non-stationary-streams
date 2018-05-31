import logging
from timeit import default_timer as timer
from PerformanceEstimators.ExecutionTimeMetric.AExecutionTimeMetric import AExecutionTimeMetric


class ExecutionTimeMetric(AExecutionTimeMetric):
    """
    Class implementing simple elapsed execution time estimator
    """

    def __init__(self):
        """
        Class constructor - initiates the start time of the anonymizer
        """
        super(ExecutionTimeMetric, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.start_time = timer()
        self.end_time = None

    def __str__(self):
        ExecutionTimeMetric.format_time(self.get_exec_time())

    def update_estimation(self, time, record_pair, cluster=None):
        """
        Update the termination time of the execution of the algorithm.
        :param time: Current time step in stream (last assigned tuple).
        :param record_pair: Pair of original instance and its anonymization.
        :param cluster: Cluster from which the record is published (Default: not needed).
        :return: None
        """
        self.processed_instances += 1
        self.end_time = timer()

    def get_exec_time(self):
        """
        Measures the elapsed execution time.
        :return: Elapsed time
        """
        # exec_time = np.array(self.replications_times).mean()
        exec_time = self.end_time - self.start_time

        formatted_time = ExecutionTimeMetric.format_time(exec_time)
        self.logger.info("Execution time Metric: %s", formatted_time)
        return exec_time

    @staticmethod
    def format_time(time):
        """
        Format elapsed time in a more interpretable format.
        :param time: Elapsed time.
        :return: Formatted elapsed time.
        """
        m, s = divmod(time, 60)
        h, m = divmod(m, 60)
        scale = 'minutes' if (h == 0 and m > 0) else 'hours' if h > 0 else 'seconds'
        return "%02d:%02d:%02d %s" % (h, m, s, scale)
