from abc import ABCMeta, abstractmethod
import sys


class AAnonymizer(object):
    """
    Abstract class of anonymizer object
    """
    __metaclass__ = ABCMeta

    def __init__(self, stream, k, l, c, eps, datatypes, publisher):
        """
        Class constructor - initiate anonymizer object
        :param stream: Stream tuples
        :param k: K-anonymity parameter
        :param l: L-diversity parameter
        :param c: Recursive (L,C)-diversity parameter
        :param eps: Differential privacy parameter
        :param datatypes: Attributes information file
        :param publisher: Instance of publisher of stream tuples
        """
        self.__stream = stream
        self.__k = k
        self.__l_diversity = l
        self.__c_diversity = c
        self.__epsilon = eps
        self.__data_types = datatypes
        self.__publisher = publisher
        self.__anonymization_pairs = []

    @property
    def stream(self):
        """
        Stream tuples
        """
        return self.__stream

    @property
    def size(self):
        """
        Size if stream (number of tuples)
        """
        return len(self.__stream)

    @property
    def k(self):
        """
        K-anonymity parameter
        """
        return self.__k

    @property
    def l_diversity(self):
        """
        L-diversity parameter
        """
        return self.__l_diversity

    @property
    def c_diversity(self):
        """
        Recursive (L,C)-diversity parameter
        """
        return self.__c_diversity

    @property
    def recursive_lc_diversity(self):
        """
        Recursive (L,C)-diversity parameter
        """
        return self.__c_diversity, self.__l_diversity

    @property
    def epsilon(self):
        """
        Differential privacy parameter
        """
        return self.__epsilon

    @property
    def data_types(self):
        """
        Attributes information file
        """
        return self.__data_types

    @property
    def publisher(self):
        """
        Instance of publisher of stream tuples
        """
        return self.__publisher

    @property
    def anonymization_pairs(self):
        """
        Record pair of original records and its anonymization
        """
        return self.__anonymization_pairs

    @anonymization_pairs.setter
    def anonymization_pairs(self, pair):
        if pair.original_record and pair.anonymized_record:
            self.__anonymization_pairs.append(pair)

    @abstractmethod
    def anonymize(self): raise NotImplementedError

    def monitor_progress(self, current_count, status=''):
        bar_len = 60
        total = self.size
        filled_len = int(round(bar_len * current_count / float(total)))

        percents = round(100.0 * current_count / float(total), 1)
        bar = '=' * filled_len + '-' * (bar_len - filled_len)

        sys.stdout.write('[%s] %s%s  ...%s\r' % (bar, percents, '%', status))
        sys.stdout.flush()
