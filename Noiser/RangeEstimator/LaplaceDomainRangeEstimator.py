from Noiser.RangeEstimator.ANoiseEstimator import ANoiseEstimator
import numpy as np


class LaplaceDomainNoiseEstimator(ANoiseEstimator):
    """
    Class of the domain range noise estimator for numerical attributes,
    based on each attribute's domain. Returns the estimated noise.
    Differential privacy sensitivity is referenced from:
    D. Sanchez, "Utility-preserving differentially private data releases via individual ranking micro-aggregation", 2016
    J. Soria-Comas, "Enhancing data utility in differential privacy via microaggregation-based k-anonymity", 2017
    """

    def __init__(self, k, m, epsilon, loc, scale):
        """
        Class constructor - initiate domain range estimator object
        :param epsilon: Differential privacy parameter epsilon
        :param loc: Location of Laplace distribution (\mu)
        :param scale: Scale (diversity) of Laplace distribution (\b)
        """
        self.__k = k
        self.__m = m
        self.__eps = epsilon
        self.__loc = loc
        self.__scale = scale
        self.__min = None
        self.__max = None
        self.init = False

        self.__history_observed = []
        self.__window = int(np.sqrt(k))
        # self.__window = 5

    @property
    def k(self):
        """
        K-anonymity parameter
        """
        return self.__k

    @property
    def m(self):
        """
        Number of attributes in the stream (cardinality of data)
        """
        return self.__m

    @property
    def epsilon(self):
        """
        Differential privacy parameter epsilon
        """
        return self.__eps

    @property
    def loc(self):
        """
        Location of Laplace distribution (\mu)
        """
        return self.__loc

    @property
    def scale(self):
        """
        Scale (diversity) of Laplace distribution (\b)
        """
        return self.__scale

    @property
    def min(self):
        """
        Minimum value of attribute X_j
        """
        return self.__min

    @property
    def max(self):
        """
        Maximum value of attribute X_j
        """
        return self.__max

    @min.setter
    def min(self, v):
        self.__min = v

    @max.setter
    def max(self, v):
        self.__max = v

    def estimate(self, value):
        """
        Estimate the scale (\b) parameter of the Laplace distribution for adding appropriate amount of noise
        Considers domain of the attribute, cardinality of data m, differential privacy \epsilon, and cluster size k
        bj = 0.5 * m * (max(D(Xj)) - min(D(Xj)))/ (k * eps)
        :param value: Value to which noise is added
        :return: Scale of noise to add
        """
        if not self.init:
            self.init = True
            self.min = value
            self.max = value
        else:
            if value > self.max:
                self.max = value
            if value < self.min:
                self.min = value

        # moving avergare of global sensitivity (can be considered local sensitivity)
        if len(self.__history_observed) > self.__window:
            predicted_sensitivity = np.mean(self.__history_observed)
            self.__history_observed[:-1] = self.__history_observed[1:]
            self.__history_observed[-1] = self.max - self.min
        else:
            predicted_sensitivity = self.max - self.min
            self.__history_observed.append(predicted_sensitivity)

        return 0.5 * self.m * predicted_sensitivity / (self.k * self.epsilon)
        # return 1.5 * self.m * predicted_sensitivity / (self.k * self.epsilon)
        # return 1.5 * self.m * (self.max - self.min) / (self.k * self.epsilon)
        # return 1.5 * (self.max - self.min) / (self.k * self.epsilon)

    def get_noise(self, s):
        """
        Get noise to add to original value in the original record.
        For numerical attributes, random Laplace distributed number is returned
        :param s: Scale parameter of Laplace distribution, for drawing random number from it
        :return: Estimated Laplace noise
        """
        return self.get_next_random_laplace(s)

    def get_next_random_laplace(self, scale):
        """
        Get random number from Laplace distribution
        X = \mu - b *\sgn(U) *ln(1 - 2|U|)
        :param scale: Scale (diversity) of Laplace distribution
        :return: Non-negative random Laplace number
        """
        sign = 1
        # Given a random variable U from the iniform distribution in the interval [-0.5,0.5]
        unif = np.random.uniform() - 0.5

        # 1 - 2*|U|
        diff = max(np.nextafter(0, 1), (1.0 - 2.0*abs(unif)))

        # sgn function
        if unif < 0:
            sign = -1

        # \mu - b *\sgn(U) * ln(1 - 2 | U |)
        return abs(self.loc - (scale * sign * np.log(diff)))
