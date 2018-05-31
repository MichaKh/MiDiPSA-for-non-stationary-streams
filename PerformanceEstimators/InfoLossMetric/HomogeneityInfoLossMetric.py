import logging
from PerformanceEstimators.InfoLossMetric.AInfoLossMetric import AInfoLossMetric
from Utils.ExceptionHandler import ExceptionHandler
from Utils.MetricsUtils import MetricsUtils


class HomogeneityInfoLossMetric(AInfoLossMetric):
    """
    Class implementing a generic information loss metric that can capture the distortion of anonymized record,
    measuring quotient between the SSE and the SST of cluster, always in the range [0, 1].
    Zero-mean and unit-variance normalization of reocrds are assumed during the distance calculation.
    Referenced in:
    1) D. Rebollo-Monedero et.al, "K-Anonymous microaggregation with preservation of statistical dependence", 2016.
    2) V. Torra, "Information Loss: Evaluation andMeasures", Data Privacy: Foundations, New Developments and the Big Data Challenge, 2017
    3) J. Domingo-Ferrer, "Practical Data-Oriented Microaggregation for Statistical Disclosure Control", 2002.
    """

    def __init__(self, original_tuples):
        super(HomogeneityInfoLossMetric, self).__init__()
        self.logger = logging.getLogger(__name__)

        self.__SSE = 0
        self.__SST = 0
        self.__qi_dimension = len(original_tuples[0].quasi_identifier)

        # Mean of whole dataset (stream), for calculating SST
        self.__dataset_centroid = MetricsUtils.calculate_centroid(original_tuples)

    @ExceptionHandler.handle_exception(ExceptionHandler.evaluation_message)
    def get_info_loss(self):
        """
        Get information loss on given stream ([0..1] range)
        SSE is the overall distance of elements x to corresponding cluster centers,
        SST is the distance of elements to the mean of X
        Calculated as SSE/SST:
        SSE = (sigma(j=1..n -> ||x_j-c_j||**2), where c_j is the centroid of cluster to which x_j belongs.
        SST = (sigma(j=1..n -> ||x_j-X^||**2), where X^ is the mean of whole dataset (stream)
        :return: Information loss (Homogeneity degree of clusters)
        """
        info_loss = float(self.__SSE) / float(self.__SST)
        self.logger.info("Homogeneity InfoLoss Metric: %0.3f", info_loss)
        return info_loss

    def update_estimation(self, time, record_pair, cluster=None):
        """
        Updates the distortion of the quasi-identifiers, normalized by the number of samples and dimension of QI
        (sigma(j=1..n -> ||xj-x'j||**2)/(qi_dimensions * N)
        Update total stream record number.
        :param record_pair: Pair of original record and its anonymization.
        :param cluster: Cluster from which the record is published (Default: not needed).
        :return: The accumulated SSE
        """
        self.processed_instances += 1
        original = record_pair.original_record.quasi_identifier
        cluster_centroid = record_pair.anonymized_record.quasi_identifier
        # if not self.__qi_dimension:  # m parameter
        #     self.__qi_dimension = len(original)
        self.__SSE += MetricsUtils.distance(original, cluster_centroid) ** 2
        self.__SST += MetricsUtils.distance(original, self.__dataset_centroid) ** 2
        return self.__SSE
