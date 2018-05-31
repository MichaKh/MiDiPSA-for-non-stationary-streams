import copy
import logging
from abc import ABCMeta, abstractmethod
from Instances.RecordPair import RecordPair
from Utils.DistributionUtils import DistributionUtils
from Utils.MetaUtils import MetaUtils
from Utils.MetricsUtils import MetricsUtils


class APublisher:
    """
    Abstract class of publisher
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def publish(self, original_rec, c, publish_qi):
        """
        Publish anonymized tuple
        :param original_rec: The original version of published tuple
        :param c: Cluster to which the original tuple belongs
        :param publish_qi: Quasi-identifier with which the tuple is published
        :return: Tuple of status of publishing (successful publishing or suppression) and the pair of original and anonymized tuples
        """
        # status 0 = anonymized and published
        # status = 2 = record is randomized due to diversity < l, and published
        # status = 1 = randomized due to c.size < k-anonymity constraint, and published
        status = 0
        if c.size >= c.k_anonymity[0]:  # cluster size is at least k-anonymity (k_min)
            # if c.check_distinct_diversity():
            # if c.check_entropy_diversity():
            if c.check_recursive_diversity():
                anonymized = self.replace_with_centroid(original_rec, publish_qi)

            else:
                status = 2
                anonymized = self.suppress(original_rec, c.W_curr.buffer)

        else:  # small cluster (cluster size less than k_min)
            anonymized = self.suppress(original_rec, c.W_curr.buffer)
            status = 1
        rp = RecordPair(original_rec, anonymized, status)
        return status, rp

    @abstractmethod
    def suppress(self, t, w=None):
        """
        Suppress (randomize) tuple of stream
        :param t: Tuple to be suppressed
        :param w: Current buffer batch of tuples in the cluster to which t belongs. Default None
        :return: Randomized version of the tuple
        """
        return self.randomize(t, w)

    @staticmethod
    def randomize(t, w=None):
        """
        Randomize values of each attribute in tuple
        For numerical attributes - return uniform random value in [min_val, max_val] range
        For categorical attributes - return random value from set of unique attribute values
        :param t: Tuple to be randomized
        :param w: Current buffer batch of tuples in the cluster to which t belongs. Default None
        :return: Randomized version of the tuple
        """
        qi = list(t.quasi_identifier)
        for i in range(0, len(qi)):
            if not isinstance(qi[i], str):  # Numerical attributes
                min_val = MetaUtils.get_attr_metadata(i, 'Min_Val')
                max_val = MetaUtils.get_attr_metadata(i, 'Max_Val')
                if not w:
                    qi[i] = DistributionUtils.get_uniform_rand(min=min_val, max=max_val, dtype=type(qi[i]))
                else:
                    records = map(lambda x: x.quasi_identifier, w)
                    attr_vals = list(zip(*records))[i]
                    x = DistributionUtils.get_estimated_rand(sample_batch=attr_vals, dtype=type(qi[i]))
                    qi[i] = MetricsUtils.truncate_value(x, l=min_val, u=max_val, dtype=type(qi[i]))
            else:  # Categorical attributes
                unique_values = MetaUtils.get_attr_metadata(i, 'Distinct_Val')
                qi[i] = DistributionUtils.get_uniform_rand(sample_batch=unique_values, dtype=type(qi[i]))
        anonymized = copy.copy(t)
        anonymized.quasi_identifier = qi
        return anonymized

    @staticmethod
    def replace_with_centroid(t, publish_qi):
        """
        Replace each tuple's values with the values of the centroid of cluster to which the tuple belongs.
        Validate the original type of attribute (int, float, string...), to maintain utility of data.
        :param t: Tuple to be published and replaced
        :param publish_qi: Centroid of cluster to which the original tuple belongs (if required, after noise addition)
        :return: Anonymized version of tuple
        """
        anonymized = copy.copy(t)
        anonymized.quasi_identifier = publish_qi

        for i, attr in enumerate(anonymized.quasi_identifier):
            anonymized.quasi_identifier[i] = MetaUtils.validate_dtype(attr, i)
        return anonymized
