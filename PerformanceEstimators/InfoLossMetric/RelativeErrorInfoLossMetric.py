import logging
from PerformanceEstimators.InfoLossMetric.AInfoLossMetric import AInfoLossMetric
from Utils.ExceptionHandler import ExceptionHandler
from Utils.MetricsUtils import MetricsUtils


class RelativeErrorInfoLossMetric(AInfoLossMetric):
    """
    Class implementing a generic information loss metric that can capture the distortion of anonymized record,
    measuring the relative error (RE) averaged over all attributes and records (Mean Relative Percentage Error).
    [D. Sanchez et. al, "Utility-preserving differentially private data releases via individual ranking microaggregation", 2016]
    """

    def __init__(self):
        super(RelativeErrorInfoLossMetric, self).__init__()
        self.logger = logging.getLogger(__name__)

    @ExceptionHandler.handle_exception(ExceptionHandler.evaluation_message)
    def get_info_loss(self):
        """
        Get average relative error over all attributes and records in stream.
        :return: Information loss (Average relative error)
        """
        info_loss = 0
        if self.processed_instances > 0:
            info_loss = 100 * float(self.current_metric) / self.processed_instances
        self.logger.info("Mean Relative Percentage Error InfoLoss Metric (Percent): %0.3f" % info_loss)
        return info_loss

    def update_estimation(self, time, record_pair, cluster=None):
        """
        Update relative error by average each error for each pair of records over number of attributes m.
        (sigma(j=1..m -> RE(a_j <-> a'_j)/(qi_dimensions=m)
        Relative error is in [0..1] range.
        Update total stream record number.
        :param record_pair: Pair of original record and its anonymization.
        :param cluster: Cluster from which the record is published (Default: not needed).
        :return: The accumulated relative error
        """
        self.processed_instances += 1
        last_error = self.current_metric

        original = record_pair.original_record.quasi_identifier
        anonymized = record_pair.anonymized_record.quasi_identifier

        self.current_metric += (MetricsUtils.relative_error(original, anonymized) / len(original))
        self.incremental_metric = self.current_metric - last_error
        # return self.__current_metric
