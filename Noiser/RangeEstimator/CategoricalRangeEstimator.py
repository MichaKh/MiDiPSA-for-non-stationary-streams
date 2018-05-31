import numpy as np
from random import randint
from Noiser.RangeEstimator.ANoiseEstimator import ANoiseEstimator


class CategoricalNoiseEstimator(ANoiseEstimator):
    """
    Class of the range noise estimator for categorical attributes,
    based on values of attribute of previously processed instances.
    Returns the estimated noise.
    """

    def __init__(self, weight=1, noise_thr=0.1):
        """
        Class constructor - Categorical Noise Estimator object
        :param weight: Weight of each attribute in the stream (Default is 1)
        :param noise_thr: Threshold for performing the replacement of the noisy categorical value
        """
        self.__noise_thr = noise_thr
        self.__attr_weight = weight
        self.__observed_values = set()
        self.init = False

    @property
    def noise_thr(self):
        """
        Threshold for performing the replacement of the noisy categorical value (Default is 0.1)
        """
        return self.__noise_thr

    @property
    def attr_weight(self):
        """
        Weight of each attribute in the stream (Default is 1)
        """
        return self.__attr_weight

    @property
    def observed_values(self):
        """
        Set containing unique values of attribute, observed in previously anonymized records
        """
        return self.__observed_values

    @observed_values.setter
    def observed_values(self, v):
        self.__observed_values.add(v)

    def estimate(self, value):
        """
        Estimate the amount of noise added to the given value.
        For categorical attributes, one previously observed value is selected randomly as the noisy value.
        :param value: Value of attribute to be replaced with its noisy version
        :return: Index of the noisy value with which the replacement will be preformed
        """
        if not self.init:
            self.init = True
        self.observed_values = value
        e = np.random.normal(0, 1, 1)
        num_of_observed = len(self.observed_values)
        # If there at least 2 unique values in the previously observed set,
        # and e < thr (relatively to the weight of the attribute)
        # the larger the weight of attribute, the larger the threshold (i.e., noisy replacement is more likely)
        if num_of_observed > 1 and e < self.noise_thr * self.attr_weight:
            rand_int = randint(0, len(self.observed_values) - 1)
            while list(self.observed_values)[rand_int] == value:
                rand_int = randint(0, len(self.observed_values) - 1)
            return rand_int
        else:
            return list(self.observed_values).index(value)

    def get_noise(self, p):
        """
        Get noise to add to original value in the original record.
        For categorical attributes, the value in the estimated index position is returned
        :param p: Index of estimated value for noise addition
        :return: Categorical value in the estimated index position
        """
        return list(self.observed_values)[p]



