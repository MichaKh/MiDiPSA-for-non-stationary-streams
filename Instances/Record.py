

class Record(object):
    """
    Class representing a record in dataset
    """

    def __init__(self, tup, centr=False):
        """
        Class constructor - initiate record object
        :param tup: Raw tuple
        :param centr: Flag indicating whether the record is a calculated centroid or raw stream tuple
        """
        self.__raw_tuple = tup
        self.__timestamp = None
        self.__quasi_identifier = None
        self.__sensitive_attr = None
        self.__is_published = False
        self.__delayed_time = 0
        self.parse_record(tup, centr)

    @property
    def raw_tuple(self):
        """
        Raw version of tuple as it appears in dataset
        """
        return self.__raw_tuple

    @property
    def timestamp(self):
        """
        Unique PID (or timestamp) of tuple in dataset, defines the order of records
        """
        return self.__timestamp

    @property
    def quasi_identifier(self):
        """
        Quasi identifier (i.e., minimal set of attributes for identification of the individual)
        """
        return self.__quasi_identifier

    @property
    def sensitive_attr(self):
        """
        The target attribute of the dataset
        """
        return self.__sensitive_attr

    @property
    def is_published(self):
        """
        Boolean flag indicating whether the tuple has been published
        """
        return self.__is_published

    @property
    def delayed_time(self):
        """
        The amount of time a tuple is delayed in a cluster (until it is published)
        """
        return self.__delayed_time

    @timestamp.setter
    def timestamp(self, v):
        self.__timestamp = v

    @quasi_identifier.setter
    def quasi_identifier(self, v):
        self.__quasi_identifier = v

    @sensitive_attr.setter
    def sensitive_attr(self, v):
        self.__sensitive_attr = v

    @is_published.setter
    def is_published(self, v):
        self.__is_published = v

    @delayed_time.setter
    def delayed_time(self, v):
        self.__delayed_time = v

    def __str__(self):
        # formatted_list = [float(Decimal("%.6f" % e)) if not isinstance(e, str) else e for e in self.quasi_identifier]
        formatted_list = ["{0:.4f}".format(e) if isinstance(e, float) else e for e in self.quasi_identifier]
        return "%s%s%s%s%s" % (self.timestamp,
                               ',',
                               str(formatted_list).replace('\'', '').replace(' ', '')[1:-1],
                               ',',
                               self.sensitive_attr)

    def parse_record(self, t, centr):
        """
        Parse tuple to its components (timestamp, QI and sensitive attribute)
        In case the record represents centroid, only QI is set (timestamp and sensitive attr are None)
        :param t: Tuple to be parsed
        :param centr: Flag indicating whether the record is a calculated centroid or raw stream tuple
        :return: None
        """
        t = list(t)
        if not centr:
            self.__timestamp = t[0]
            self.__quasi_identifier = t[1:-1]
            self.__sensitive_attr = t[-1]
        else:
            self.__quasi_identifier = t[:]
