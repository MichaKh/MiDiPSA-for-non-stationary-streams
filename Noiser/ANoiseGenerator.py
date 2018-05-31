from abc import ABCMeta, abstractmethod


class ANoiseGenerator(object):
    """
    Abstract class for initializing noiser for satisfying differential privacy
    """
    __metaclass__ = ABCMeta

    def __init__(self, k, eps, noise_thr):
        """
        Class constructor - initiate noiser object
        :param k: K-anonymity parameter
        :param eps: Differential privacy parameter epsilon
        :param noise_thr: Threshold for performing the replacement of the noisy categorical value
        """
        self.__k = k
        self.__eps = eps
        self.__noise_thr = noise_thr

    @property
    def k(self):
        """
        K-anonymity parameter
        :return:
        """
        return self.__k

    @property
    def epsilon(self):
        """
        Differential privacy parameter epsilon
        :return:
        """
        return self.__eps

    @property
    def noise_thr(self):
        """
        Threshold for performing the replacement of the noisy categorical value
        """
        return self.__noise_thr

    @abstractmethod
    def add_noise(self, record):
        """
        Add noise to attributes in record w.r.t its domain and noise scale
        :param record: Record to which noise is added
        :return: Noisy version of record, for anonymization
        """
        raise NotImplementedError
