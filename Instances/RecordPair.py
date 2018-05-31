class RecordPair(object):
    """
    Class representing a record pair of original record and its anonymized version record
    """

    def __init__(self, original_record, anonymized_record, status=0):
        """
        Class constructor - initiate record pair object
        :param original_record: Original version of record
        :param anonymized_record: Anonymized version of record
        :param status: Publication status [0: published anonymized (Default), 1/2: randomized (suppressed)]
        """
        self.__original_record = original_record
        self.__anonymized_record = anonymized_record
        self.__status = status

    @property
    def original_record(self):
        return self.__original_record

    @property
    def anonymized_record(self):
        return self.__anonymized_record

    @property
    def status(self):
        """
        Publication status:
         0: published anonymized (OK),
         1/2: randomized (suppressed)
        :return:
        """
        return self.__status
