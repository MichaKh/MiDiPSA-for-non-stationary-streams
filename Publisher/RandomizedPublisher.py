from Instances.RecordPair import RecordPair
from Publisher.APublisher import APublisher


class RandomizedPublisher(APublisher):
    """
    Publisher class for replacing each tuple's values with random values
    """

    def publish(self, original_rec, c, publish_qi):
        """
        Publish anonymized tuple
        :param original_rec: The original version of published tuple
        :param c: None
        :param publish_qi: Quasi-identifier with which the tuple is published
        :return: Randomized version of the tuple
        """
        anonymized = super(RandomizedPublisher, self).suppress(original_rec)
        rp = RecordPair(original_rec, anonymized)
        return 0, rp  # status = 0 = randomized

    def suppress(self, t, w=None):
        """
        Suppress (randomize) values of tuple
        :param t: Tuple to be published
        :param w: Current buffer batch of tuples in the cluster to which t belongs. Default None
        :return: Randomized version of the tuple
        """
        return super(RandomizedPublisher, self).suppress(t)
