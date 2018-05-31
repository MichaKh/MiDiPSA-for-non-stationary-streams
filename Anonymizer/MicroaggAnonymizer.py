import logging
import time
from copy import deepcopy, copy
from Anonymizer.AAnonymizer import AAnonymizer
from Buffer.Buffer import Buffer
from Cluster import Cluster
from ConceptDriftHandler.ConceptDriftDetector import ConceptDriftDetector
from Instances.Record import Record
from Instances.RecordPair import RecordPair
from PerformanceEstimators.InfoLossMetric.SSEInfoLossMetric import SSEInfoLossMetric
from Utils.ExceptionHandler import ExceptionHandler
from Utils.MetricsUtils import MetricsUtils


class MicroaggAnonymizer(AAnonymizer):
    """
    Microaggregation-Based Differential private Stream Anonymization algorithm
    """

    # k=k-anonymity, dist_thr, eps = epsilon for Differential Privacy
    def __init__(self, stream,
                 k,
                 l,
                 c,
                 eps,
                 b,
                 delta,
                 dist_thr,
                 datatypes,
                 publisher,
                 noiser,
                 estimators,
                 change_detector=None):
        """
        :param stream: Stream tuples
        :param k: K-anonymity parameter
        :param l: L-diversity parameter
        :param c: Recursive (C,L)-diversity parameter
        :param eps: Differential privacy parameter epsilon
        :param b: Size of batch of tuples used for detecting concept drift
        :param delta: Maximum publishing delay of tuples in a non-k-anonymity cluster
        :param dist_thr: Distance threshold for assigning tuples to a cluster
        :param cd_thr: Concept drift threshold to declaring Concept drift
        :param cd_conf: Statistical confidence level for detecting concept drift (Default: 0.1)
        :param datatypes: Attributes information file
        :param publisher: Instance of publisher of stream tuples
        :param noiser: Instance of noiser for adding noise to values in tuple
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing micro-aggregation anonymizer...")
        super(MicroaggAnonymizer, self).__init__(stream, k, l, c, eps, datatypes, publisher)

        self.__buff = b
        self.__delta = delta
        self.__dist_thr = dist_thr
        self.__buffer = Buffer(b)
        self.__cluster_set = []

        # Monitor the distribution change is cluster to detect concept drift
        self.__change_detector = change_detector

        # Monitor the increase in error in cluster (inforamtion loss) after assigning a tuple to it
        self.__error_increase_estimator = SSEInfoLossMetric()

        self.estimators = estimators
        self.total_randomized_diversity = 0
        self.total_randomized_size = 0
        self.total_opened_clusters = 0
        self.final_remaining_tuples = 0
        self.noiser = noiser

        self.logger.info(
            'Algorithm parameters: k={0}, l={1}, c={2}, eps={3}, b={4}, delta={5}, dist_thr={6}, cd_thr={7}'.format(
                (k[0], k[-1]), l, c, eps, b, delta, dist_thr, change_detector.cd_factor if change_detector else 'Disabled'))

    @property
    def buff_size(self):
        return self.__buff

    @property
    def delta(self):
        return self.__delta

    @property
    def dist_thr(self):
        return self.__dist_thr

    @property
    def buffer(self):
        return self.__buffer

    @buffer.setter
    def buffer(self, value):
        self.__buffer.insert(value)

    @property
    def drift_detector(self):
        return self.__change_detector

    @property
    def cluster_set(self):
        return self.__cluster_set

    @cluster_set.setter
    def cluster_set(self, c):
        self.__cluster_set.append(c)

    # @ExceptionHandler.handle_exception(ExceptionHandler.halt_message)
    def anonymize(self):
        """
        Anonymize data stream using Microaggregation.
        Handles the stream one tuple at-a-time and passes it through the anonymizer.
        :return: Record pair of original record and its anonymized version
        """
        self.logger.info("Running anonymizer...")
        for t in self.stream:
            if t:
                c = self.search_best_cluster(t)
                if c:
                    if c.W_curr and c.W_curr.size > 0:
                        c.exist_time = t.timestamp - c.W_curr.peek().timestamp
                    if c.exist_time >= self.delta:
                        self.remove_cluster(c)
                        # Open new cluster and add t to the new cluster
                        c = self.create_new_cluster()
                        self.cluster_set = c

                    c.update_cluster_tuples(t)

                    if c.size == self.k[0]:  # Cluster satisfies k-anonymity and tuples in it can be published
                        publish_qi = self.noiser.add_noise(c.centroid)
                        for tp in c.W_curr.buffer:  # Publish all accumulated tuples of cluster
                            if not tp.is_published:
                                self.anonymization_pairs = self.publish(t.timestamp, tp, c, publish_qi)

                    # if the cluster contains at least k records
                    elif self.k[0] < c.size <= self.k[-1] + 1:
                        publish_qi = self.noiser.add_noise(c.centroid)
                        self.anonymization_pairs = self.publish(t.timestamp, t, c, publish_qi)

                    # if cluster is too large (c.size > k_max), remove cluster
                    if c.size > self.k[-1]:
                        self.cluster_set.remove(c)

                    # Buffer of cluster reached the size threshold and concept drift can be detected
                    if c.W_curr.is_full:
                        # True = Buffer is full (reached size limit)
                        is_drift = self.detect_concept_drift(c)
                        if is_drift:
                            c.reset_centroid()

                else:
                    # Open new cluster and add t to the new cluster
                    c = self.create_new_cluster()
                    self.cluster_set = c
                    c.update_cluster_tuples(t)

        # no more tuples arrive, publish the remaining tuples and remove clusters
        for c in copy(self.cluster_set):
            self.final_remaining_tuples += self.remove_cluster(c)

        self.logger.info("Anonymization completed!")
        return self.anonymization_pairs

    @ExceptionHandler.handle_exception(ExceptionHandler.halt_message)
    def search_best_cluster(self, t):
        """
        Find the nearest and least info-loss increasing cluster w.r.t a new arriving tuple.
        Calculate the distance from the arrived tuple to each cluster in the system, bounded to a given threshold.
        Increase in information loss is computed as the incremental squared error (SSE) of a candidate cluster.
        :param t: New arriving tuple
        :return: The nearest and least info-loss increasing cluster
        """
        cluster_dist_dict = {}
        if not self.cluster_set:
            return None
        else:
            for c in self.cluster_set:
                centr = c.centroid.quasi_identifier
                dist = MetricsUtils.distance(centr, t.quasi_identifier)

                if dist <= self.dist_thr:
                    cluster_dist_dict[c] = dist
            if len(cluster_dist_dict) > 0:
                cluster_dist_dict = sorted(cluster_dist_dict.items(), key=lambda x: x[1])
                return self.check_min_info_loss_increase(cluster_dist_dict, t)
                # return cluster_dist_dict[0][0]
            return None

    def check_min_info_loss_increase(self, cluster_dist_dict, t):
        """
        Check each cluster in the system as a candidate for assigning the arriving tuple to it.
        Compute the incremental squared error (information loss) that incurred by adding tuple to the cluster.
        Find the least info loss increasing cluster and verify that its distance to the tuple does not exceed preset threshold.
        :param cluster_dist_dict: Dictionary of cluster and the distance from its centroid to the arriving tuple
        :param t: Arriving tuple to be assigned.
        :return: Cluster if satisfies the distance threshold, otherwise None.
        """
        # Dictionary of cluster and the increase in error its incurs after assigning arriving tuple to it.
        cluster_error_dict = {}
        for c in cluster_dist_dict:
            record_pair = RecordPair(t, Record(c[0].centroid.quasi_identifier, centr=True))

            self.__error_increase_estimator.update_estimation(time=None, record_pair=record_pair)

            # calculate the incremental info loss (SSE) from assigning tuple t into the cluster
            incremental_error = self.__error_increase_estimator.incremental_metric
            cluster_error_dict[c] = incremental_error

        if len(cluster_dist_dict) > 0:
            item_list = cluster_error_dict.items()
            cluster_error_dict = sorted(item_list, key=lambda x: x[1])
            return cluster_error_dict[0][0][0]
        return None

    def create_new_cluster(self):
        """
        Create new cluster and add it to the cluster set for reuse.
        Creation time is documented for monitoring the activity rate of the cluster.
        :return: The new formed cluster.
        """
        new_cluster = Cluster(time.time(),
                              self.k,
                              self.l_diversity,
                              self.c_diversity,
                              self.buff_size,
                              self.drift_detector)
        self.total_opened_clusters += 1
        return new_cluster

    def check_clusters_activity(self, cp_thr, curr_count):
        """
        Check whether the cluster exists more than predefined time without new tuples assigned into it.
        Calculates the arrival rate of tuples to the cluster.
        Removes cluster with arrival rate smaller than some threshold, or existence more than delta.
        :param cp_thr: Threshold of cluster activity.
        :param curr_count: Timestamp (ID) of last arrived tuple.
        :return: None
        """
        for c in copy(self.cluster_set):
            c.exist_time = curr_count - c.W_curr.peek().timestamp

            # if small cluster exists more than a threshold delay, remove cluster
            if c.size < self.k[0] and c.exist_time >= self.delta:
                self.remove_cluster(c)
        return

    def remove_cluster(self, c):
        """
        Remove cluster from the cluster set, and publish all remaining tuples in it.
        :param c: Cluster to be removed.
        :return: Number of published tuples in the removed cluster.
        """
        current_published = 0
        if c:
            for t in c.W_curr.buffer: # if there exist unpublished tuples in the cluster
                if not t.is_published:
                    self.anonymization_pairs = self.publish(self.size, t, c, c.centroid.quasi_identifier)
                    current_published += 1
        self.cluster_set.remove(c)
        return current_published

    def detect_concept_drift(self, c):
        """
        Detect concept drift in a given cluster, using the initialized detector of the anonymizer.
        :param c: Cluster to be checked for concept drift.
        :return: True, is cluster represents concept drift, otherwise False.
        """
        is_drift = False
        if c.W_curr and c.W_prev and c.drift_detector:
            # is_drift = self.drift_detector.detect(c)
            is_drift = c.drift_detector.detect(c)

        c.W_prev = deepcopy(c.W_curr)  # Copy current buffer batch to previous
        c.W_curr.reset()  # Clear current batch after checking for concept drift
        return is_drift

    def publish(self, time, tp, c, publish_qi):
        """
        Publish given tuple with the given quasi-identifier (centroid of the cluster to which it belongs).
        :param tp: Tuple to be published.
        :param c: Cluster to which the given tuple belongs.
        :param publish_qi: Quasi-identifier to publish
        :return: Record pair of the original record and its anonymization
        """
        self.monitor_progress(tp.timestamp, status='Publishing tuple: {0}'.format(len(self.anonymization_pairs)))
        status_rec_pair = self.publisher.publish(tp, c, publish_qi)

        status = status_rec_pair[0]
        tp.is_published = True
        self.total_randomized_size += 1 if status == 1 else 0
        self.total_randomized_diversity += 1 if status == 2 else 0

        self.update_metrics(time, c, tp, status_rec_pair)
        return status_rec_pair[1]  # the original record and its anonymization

    def update_metrics(self, time, c, tp, record_pair):
        """
        Incrementally update the performance metrics of the anonymizer.
        These may include information loss, disclosure risk and other estimators.
        :param time: Current time step in stream (last assigned tuple)
        :param c: Cluster of last published tuple.
        :param tp: Last published tuple
        :param record_pair: Pair of {publication status , Record pair :{Original record, Anonymized record}}
        :return:
        """
        for estimator in self.estimators:
            if estimator:
                estimator.update_estimation(time, record_pair[1], c)
