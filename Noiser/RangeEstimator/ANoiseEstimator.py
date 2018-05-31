from abc import ABCMeta, abstractmethod


class ANoiseEstimator(object):
    """
    Abstract class for estimation the range (scale) of noise to be added,
    and return the estimated noise.
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def estimate(self, value):
        """
        Estimate the amount of noise added to the given value.
        :param value: Value of attribute in original reocrd
        :return: Scale of noise ( for numerical attributes) or index of nominal value (for categorical attributes)
        """
        raise NotImplementedError

    @abstractmethod
    def get_noise(self, p):
        """
        Get noise to add to original value in the original record.
        :param p: Scale parameter of Laplace distribution (for numerical attr)
        or index of nominal value (for categorical attributes)
        :return: Noise to be added to value in original record
        """
        raise NotImplementedError
