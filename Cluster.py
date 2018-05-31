import time
import numpy as np
from pandas import Series
from scipy import stats
from Buffer.Buffer import Buffer
from Instances.Record import Record
from Utils.MetaUtils import MetaUtils
from Utils.MetricsUtils import MetricsUtils


class Cluster(object):
    """
    Class representing cluster
    """

    def __init__(self, ct, k, l, c, buff_size, drift_detector):
        """
        Class constructor - initiate cluster object
        :param ct: Creation time of cluster, for tracking its activity
        :param k: K-anonymity parameter
        :param l: L-Diversity parameter
        :param c: Recursive (C,L)-Diversity parameter
        :param buff_size: Maximum size of buffer of cluster
        :param drift_detector: Instance of the concept drift detector class
        """
        self.__creation_time = ct
        self.__k = k
        self.__l = l
        self.__c = c
        self.__centroid = Record([], centr=True)
        self.__exist_time = 0  # How much time the cluster exists, for setting limit on existence of non-k cluster
        self.__drift_detector = drift_detector
        self.__last_arrival_time = 0
        self.__arrival_rate = 0.
        self.__sum_time_intervals = 0
        self.__GSE = 0  # square error in group(cluster) for calculating SSE (InfoLoss)
        self.__GSE_T = 0  # total square error in group(cluster) for calculating SST (InfoLoss)

        self.W_prev, self.W_curr = None, Buffer(buff_size)
        self.size = 0  # size of cluster
        # Counter of freq of each categorical value (dictionary of dictionaries for each categorical attr)
        self.categorical_freq = {}

    @property
    def creation_time(self):
        """
        Creation time of cluster, for tracking its activity
        """
        return self.__creation_time

    @property
    def k_anonymity(self):
        """
        K-anonymity parameter
        """
        return self.__k

    @property
    def l_diversity(self):
        """
        L-Diversity parameter
        """
        return self.__l

    @property
    def recursive_cl_diversity(self):
        """
        Tuple of (C,L)-diversity parameter
        """
        return self.__c, self.__l

    @property
    def arrival_rate(self):
        return self.__arrival_rate

    @arrival_rate.setter
    def arrival_rate(self, value):
        self.__arrival_rate = value

    @property
    def centroid(self):
        """
        Centroid of cluster (instance of Record)
        """
        return self.__centroid

    @centroid.setter
    def centroid(self, value):
        self.__centroid = value

    @property
    def exist_time(self):
        """
        Existence time of cluster (in records units)
        """
        return self.__exist_time

    @exist_time.setter
    def exist_time(self, v):
        """
        Existence time of cluster (in records units)
        """
        self.__exist_time = v

    @property
    def drift_detector(self):
        """
        Concept drift detector of the cluster
        """
        return self.__drift_detector

    def update_cluster_tuples(self, t):
        """
        Assign new tuple to cluster, update the cluster centroid and its activity
        :param t: Tuple of record to be assigned
        """
        self.size += 1
        self.update_categorical_freq(t)
        # insert to buffer both the original tuple,
        # for storing records until cluster's size reaches K
        self.W_curr.insert(t)

        centr = self.update_cluster_centroid(t)
        self.arrival_rate = self.compute_activity()
        return self.centroid

    def update_cluster_centroid(self, t):
        """
        Calculates the new centroid of the entire cluster, after assigning the last tuple into it.
        :return: Updated centroid record of the entire cluster.
        """
        if self.size <= 1:
            self.centroid = t
        else:
            centr = MetricsUtils.update_centroid(self.centroid.quasi_identifier,
                                                 t.quasi_identifier,
                                                 self.size,
                                                 self.categorical_freq)
            self.centroid = Record(centr, centr=True)
        return self.centroid

    def update_categorical_freq(self, t):
        """
        Update the frequency of categorical values in a cluster.
        Used for estimating the distribution of the sensitive value in a cluster, and for updating the cluster centroid.
        :param t: The new arrived tuple.
        :return: Dictionary of frequency of categorical values.
        """
        attr_list = t.quasi_identifier + [t.sensitive_attr]  # append sensitive value
        for index, attr_val in enumerate(attr_list):
            if index not in self.categorical_freq:
                self.categorical_freq[index] = {}
            if isinstance(attr_val, str):
                if attr_val not in self.categorical_freq[index]:
                    self.categorical_freq[index][attr_val] = 1
                else:
                    self.categorical_freq[index][attr_val] += 1
        return self.categorical_freq

    def reset_centroid(self):
        """
        Reset centroid to the centroid of last stored buffer, denoted by W_prev.
        Performed in case concept drift is detected in cluster.
        :return: None
        """
        if self.W_prev:
            old_centr = self.W_prev.update_buffer_centroid()
            self.centroid = old_centr
        return

    def compute_activity(self):
        """
        Calculate the activity of a cluster, based on the rate of arrival of tuples to it.
        [cluster size - 1] / [sigma (i=1 -> number of tuples in cluster){current time - last arrival time} + 1]
        :return: Arrival activity of cluster
        """
        # Arrival interval : current time - last tuple's arrival time

        current_time = time.time()
        if self.__last_arrival_time == 0:
            self.__last_arrival_time = current_time
        self.__sum_time_intervals += current_time - float(self.__last_arrival_time)
        return (self.size - 1) / (self.__sum_time_intervals + 1)

    # Check whether Diversity(cluster + t) >= Diversity(cluster)
    def check_distinct_diversity(self):
        """
        Checks whether each equivalence class has at least l well-represented sensitive values.
        :return: True if cluster satisfies l-diversity, otherwise False.
        """
        # frequency distribution of values in sensitive attr
        sensitive_dict = self.categorical_freq[self.categorical_freq.keys()[-1]]

        count_distinct_values = len(sensitive_dict.keys())
        return count_distinct_values >= self.l_diversity

    def check_recursive_diversity(self):
        """
        Checks whether most common value appears too often while less common values do not appear too infrequently.
        Sorts the number of times each sensitive value appears in the cluster, naming them r_1...r_m.
        Satisfy r1 < c(r_l+ r_l+1 +...+ r_m), for some constant c.
        :return: True if cluster satisfies (c,l)-diversity, otherwise False.
        """
        rl_to_rm = 0
        c, l = self.recursive_cl_diversity

        # frequency distribution of values in sensitive attr
        sensitive_dict = self.categorical_freq[self.categorical_freq.keys()[-1]]

        sorted_values = sorted(sensitive_dict, key=sensitive_dict.get, reverse=True)
        r1 = sensitive_dict[sorted_values[0]]
        for i in range(l-1, len(sensitive_dict.keys())):
            rl_to_rm += sensitive_dict[sorted_values[i]]
        return r1 < c * rl_to_rm

    def check_entropy_diversity(self):
        """
        Checks whether in each equivalence class the different sensitive values are distributed evenly enough.
        Satisfy -sum(p_s*log(p_s)) >= log(l)
        :return: True if cluster satisfies entropy-diversity, otherwise False.
        """
        # frequency distribution of values in sensitive attr
        sensitive_dict = self.categorical_freq[self.categorical_freq.keys()[-1]]

        total = sum(sensitive_dict.values())
        prob_dict = {key: float(value) / total for (key, value) in sensitive_dict.items()}
        p_data = Series(prob_dict.values())  # calculates the probabilities
        entropy = stats.entropy(p_data, base=2)  # input probabilities to get the entropy
        return entropy > np.log2(self.l_diversity)

    # GSE(Gi) = sigma(Dist(xij, xi)) / xi= centroid of cluster i, xij = element j of cluster i
    def update_GSE(self):
        xi = self.centroid
        xij = self.W_curr.peek().quasi_identifier
        dataset_centroid = MetaUtils.dataset_centroid
        appended_SSE_dist = MetricsUtils.distance(xij, xi)
        self.__GSE += appended_SSE_dist
        appended_SST_dist = MetricsUtils.distance(xij, dataset_centroid)
        self.__GSE_T += appended_SST_dist
