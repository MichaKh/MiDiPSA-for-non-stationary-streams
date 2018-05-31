import logging
from PerformanceEstimators.InfoLossMetric.AInfoLossMetric import AInfoLossMetric
from Utils.ExceptionHandler import ExceptionHandler


class ClassificationInfoLossMetric(AInfoLossMetric):
    """
    Class implementing a generic information loss metric that can capture the distortion of anonymized record,
    measuring the classification penalty of anonymized records.
    The penalty for each records is defined: 1 if records was randomized (suppressed) or its class label is not the majority class in its cluster.
    Sum of penalties normalized by total number of records in stream is calculated.
    Referenced in: V. Iyengar, Transforming Data to Satisfy Privacy Constraints, 2002.
    """

    def __init__(self):
        super(ClassificationInfoLossMetric, self).__init__()
        self.logger = logging.getLogger(__name__)

        self.__penalties_sum = self.current_metric

    @ExceptionHandler.handle_exception(ExceptionHandler.evaluation_message)
    def get_info_loss(self):
        """
        Sum of penalties of all records normalized by total number of records in stream.
        :return: Information loss (Majority classification metric)
        """
        info_loss = float(self.__penalties_sum) / float(self.processed_instances)
        self.logger.info("Classification InfoLoss Metric: %0.3f", info_loss)
        return info_loss

    def update_estimation(self, time, record_pair, cluster=None):
        """
        Calculate penalty for each arriving records in the stream:
         {1 if record r is suppressed
         {1 if class(r) != majority_class(cluster of r)
         {0 else
        :param record_pair: Record publication status [0 published anonymized, 1/2 randomized (suppressed)]).
        :param cluster: Cluster from which the record is published (Default: not needed).
        :return: The accumulated records penalty
        """
        self.processed_instances += 1
        label = record_pair.anonymized_record.sensitive_attr

        # frequency distribution of values in sensitive attr
        sensitive_dict = cluster.categorical_freq[cluster.categorical_freq.keys()[-1]]
        common_value = max(sensitive_dict, key=sensitive_dict.get)
        if record_pair.status == 1 or record_pair.status == 2 or common_value != label:
            self.__penalties_sum += 1

