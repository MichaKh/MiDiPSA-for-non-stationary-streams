import math
import numpy as np
from scipy import stats
from ConceptDriftHandler.AConceptDriftDetector import AConceptDriftDetector
from Utils.MetaUtils import MetaUtils
from Utils.MetricsUtils import MetricsUtils
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class ConceptDriftDetector(AConceptDriftDetector):
    """
    Class implementing concept drift detector in evolving stream.
    The concept drift is detected using the distance between two consecutive buffer windows in a given cluster.
    The change is computed as the distance between the centroids of previous and current windows of preset size.

    A two-samples Kolmogorov-Smirnov statistical test is performed to estimate where two consecutive samples in some
     cluster are from different distributions.
    Similar idea of measuring distance between centroids (or windows) is presented in:
    D. Kifer et al., "Detecting Change in Data Streams", 2004
    E.J. Spinosa et al., "OLINDDA: A cluster-based approach for detecting novelty and concept drift in data streams", 2007
    E.R. Faria et al., "Novelty Detection Algorithm for Data Streams Multi-Class Problems", 2013
    D. Reis et. al., "Fast Unsupervised Online Drift Detection Using Incremental Kolmogorov-Smirnov Test", 2016
    """

    def __init__(self, buff_size, factor=1, conf=0.05):
        """
        Class constructor - initiate Concept drift detector object
        :param conf: Statistical confidence for declaring true concept drift (Default : 0.05)
        :param factor: Factor for scaling the statistic value of the hypothesis statistical test. (Default: 1)
        :param buff_size: Size of window buffer in each cluster
        """
        super(ConceptDriftDetector, self).__init__(conf)
        self.__buff_size = buff_size
        self.__factor = factor
        self.__threshold = self.get_threshold(buff_size, buff_size)

    @property
    def cd_factor(self):
        return self.__factor

    def get_threshold(self, n, m):
        """
        Compute the statistical threshold for considering a concept drift.
        The smaller the threshold is, the more likely it is to detect small changes in the distribution,
         but the larger is our risk of false alarm.

        Perform two-tailed Kolmogorov-Smirnov statistical test:
            H0: Observations in A and B originate from the same probability distribution
            H1: Otherwisw
        D >? c(alpha) * sqrt[(n+m) / (n*m)],
            where c(alpha) is retrieved from KS-table, n amd m are the size of each sample respectively,
            D is the Kolmogorov-Smirnov statistic, i.e., the obtained p-value.
        From: D. Reis et. al., "Fast Unsupervised Online Drift Detection Using Incremental Kolmogorov-Smirnov Test", 2016
        :param n: Size of first buffer sample
        :param m: Size of second buffer sample
        :return: Concept drift threshold
        """
        KS_table = {0.10: 1.22, 0.05: 1.36, 0.025: 1.48, 0.01: 1.63, 0.005: 1.73, 0.001: 1.95}
        KS_critical_val = KS_table[self.confidence]
        p_value = KS_critical_val * math.sqrt(float((n + m)) / (n * m))
        return p_value

    def distance_based_test(self, c):
        """
        Compare the distance of the two centroids of the buffers in the given cluster with the p-value of the K-S test.
        Used for mix-type tuples.
        :param c: Cluster to be inspected for concept drift. Contains both current and previous buffer windows.
        :return:
        """
        z1 = c.W_prev.update_buffer_centroid().quasi_identifier
        z2 = c.W_curr.update_buffer_centroid().quasi_identifier
        self.incremental_change = self.__factor * MetricsUtils.distance(z1, z2)
        return self.incremental_change

    def distribution_based_test(self, c):
        """
        Compare the cumulative distribution difference between the two buffers (as samples).
        Used for all-numerical tuples.
        :param c: Cluster to be inspected for concept drift. Contains both current and previous buffer windows.
        :return:
        """
        z1 = [record.quasi_identifier for record in c.W_prev.buffer]
        l_z1 = np.array([np.array(xi) for xi in z1])

        z2 = [record.quasi_identifier for record in c.W_curr.buffer]
        l_z2 = np.array([np.array(xi) for xi in z2])

        standardized_z1 = StandardScaler().fit_transform(l_z1)
        standardized_z2 = StandardScaler().fit_transform(l_z2)

        assert (len(z1) == len(z2))

        pca1 = PCA(n_components=1)
        pca2 = PCA(n_components=1)

        s1 = pca1.fit_transform(standardized_z1)
        s2 = pca2.fit_transform(standardized_z2)
        statistic, p_value = stats.ks_2samp(s1.flatten(), s2.flatten())
        return statistic, p_value

    # def detect(self, c):
    #     """
    #     Detect concept drift in stream, given two consecutive windows of samples.
    #     Checks whether both windows differs (distance) by more than the threshold.
    #     If (dist > threshold) declare concept drift.
    #     :param c: Cluster to be inspected for concept drift. Contains both current and previous buffer windows.
    #     :return: True, if drift is detected, otherwise False
    #     """
    #     self.windows_processed += 1
    #
    #     z1 = c.W_prev.update_buffer_centroid().quasi_identifier
    #     z2 = c.W_curr.update_buffer_centroid().quasi_identifier
    #
    #     assert (len(z1) == len(z2))
    #
    #     self.incremental_change = self.__factor * MetricsUtils.distance(z1, z2)
    #     self.monitor_overtime_change()
    #
    #     if self.incremental_change > self.__threshold:
    #         self.logger.warning("Concept change is detected. Change between consecutive buffers is {0:0.3f},"
    #                             " p-value is {1:0.3f},"
    #                             " confidence level is {2:0.1f}%".format(self.incremental_change,
    #                                                                     self.__threshold,
    #                                                                     (1 - self.confidence) * 100))
    #         return True
    #     return False

    def detect(self, c):
        """
        Detect concept drift in stream, given two consecutive windows of samples.
        Checks whether both windows differ by more than the threshold, using Kolmogorov-Smirnov statistical test
        If (KS_statistic > critical_value) declare concept drift.
        :param c: Cluster to be inspected for concept drift. Contains both current and previous buffer windows.
        :return: True, if drift is detected, otherwise False
        """
        if MetaUtils.is_all_numeric:
            critical, p_value = self.distribution_based_test(c)
        else:
            critical = self.distance_based_test(c)

        if critical > self.__threshold:
            self.logger.warning("Concept change is detected. Change between consecutive buffers is {0:0.3f},"
                                "p-value is {1:0.3f},"
                                "confidence level is {2:0.1f}%".format(critical,
                                                                       self.__threshold,
                                                                       (1 - self.confidence) * 100))
            return True
        return False

    def monitor_overtime_change(self):
        """
        Monitor the detection of concept drift over time.
        Monitor the change in the distance between two consecutive buffers in the cluster for which the buffers are full.
        :return: None
        """
        point = (self.windows_processed * self.__buff_size, self.incremental_change)
        self.metric_over_time.append(point)
