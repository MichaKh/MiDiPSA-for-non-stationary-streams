from Publisher.APublisher import APublisher


class CentroidPublisher(APublisher):
    """
    Publisher class for replacing each tuple's values with the values of the centroid of cluster to which it is assigned
    In case tuple is suppressed, random value from Uniform distribution are drawn (according to each attribute's domain)
    """

    def publish(self, original_rec, c, publish_qi):
        """
        Publish anonymized tuple
        :param original_rec: The original version of published tuple
        :param c: Cluster to which the original tuple belongs
        :param publish_qi: Quasi-identifier with which the tuple is published
        :return: Tuple of status of publishing (successful publishing or suppression) and the pair of original and anonymized tuples
        """
        return super(CentroidPublisher, self).publish(original_rec, c)

    def suppress(self, t, w=None):
        """
        Suppress (randomize) values of tuple
        :param t: Tuple to be published
        :param w: Current buffer batch of tuples in the cluster to which t belongs. Default None
        :return: Randomized version of the tuple
        """
        return super(CentroidPublisher, self).suppress(t)



