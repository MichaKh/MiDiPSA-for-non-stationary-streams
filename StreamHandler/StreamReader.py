import logging
import os
import random
import re
from collections import OrderedDict, Counter
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from numpy import isnan
from copy import deepcopy, copy
from Instances.Record import Record
from Utils.ExceptionHandler import ExceptionHandler
# dataset_centroid = list()  # centroid of entire dataset, used for calculating SSG
from Utils.MetaUtils import MetaUtils


class StreamReader(object):

    def __init__(self, dir, source_file, datatypes_file):
        """
        Class constructor - initiate stream reader
        :param dir: Directory of file path
        :param source_file: Path to stream source file
        :param datatypes_file: Path to attributes information file
        """
        self.logger = logging.getLogger(__name__)
        self.__source_path = os.path.join(dir, source_file)
        self.__datatypes_path = os.path.join(dir, datatypes_file)
        self.__tuples = []

    @property
    def tuples(self):
        """
        Tuples representing records in data stream
        """
        return self.__tuples

    @tuples.setter
    def tuples(self, t):
        """
        Append new tuple to stream tuples
        :param t: New arrived tuple
        """
        # self.__tuples.append(t)
        self.__tuples = t

    @ExceptionHandler.handle_exception(ExceptionHandler.halt_message, TypeError, IOError)
    def read_file(self):
        """
        Read file in txt format and parse it
        :return: Formed tuples of records in data stream
        """
        with open(self.__source_path, 'r') as f:
            lines = f.readlines()
            lines = [x.strip() for x in lines]
        self.parse_lines(lines)
        return self.tuples

    @ExceptionHandler.handle_exception(ExceptionHandler.halt_message, TypeError, IOError, KeyError)
    def read_csv_file(self, shuffle=False, duplicate_frac=None):
        """
        Read file in CSV format containing the dataset used for simulating stream
        Determine relevant metadata of stream, including type of attribute and their parameters.
        :param shuffle: True if the dataset needs to be shuffled to avoid sorted or organized data, otherwise False (default)
        :param duplicate_frac: None if no tuples with duplicate PID are needed in stream, otherwise percentage of such tuples to be added
        :return: Attribute information dictionary
        """
        self.logger.info("Preparing stream dataset for anonymization...")
        df = pd.read_csv(self.__source_path, skipinitialspace=True)

        # If dataset is sorted, shuffle its records
        if shuffle:
            df = self.shuffle_tuples(df)
        # For simulation of data stream with duplicate pids:
        # Randomly select some fraction of records and insert them back into the original dataset.
        if duplicate_frac:
            self.reinsert_duplicates(duplicate_frac)

        # Read dataset structure file and get attributes types
        StreamReader.STREAM_METADATA = self.change_dtypes(df, self.__datatypes_path)

        # Determine whether all attributes are numeric or nominal, for efficient future processing
        dtypes = [StreamReader.STREAM_METADATA[key]['Type'] for key in StreamReader.STREAM_METADATA.keys()[0:-1]]

        MetaUtils.stream_metadata = StreamReader.STREAM_METADATA.values()
        MetaUtils.stream_attr_names = StreamReader.STREAM_METADATA.keys()
        MetaUtils.check_is_all_nominal = all(v == 'categorical' for v in dtypes)
        MetaUtils.is_all_numeric = all(v == 'numeric' for v in dtypes)

        self.create_tuples(df)

        self.logger.info("Total %s records in stream!" % (len(self.tuples)))
        self.logger.info("Quasi-identifier consists of %s numeric attributes and %s categorical attributes."
                         " Sensitive attribute is '%s' (%s)" %
                         (Counter(dtypes)['numeric'],
                          Counter(dtypes)['categorical'],
                          StreamReader.STREAM_METADATA.keys()[-1],
                          StreamReader.STREAM_METADATA[StreamReader.STREAM_METADATA.keys()[-1]]['Type']))
        return StreamReader.STREAM_METADATA

    def parse_lines(self, lines):
        """
        Parse rows in txt file and split it to its values
        :param lines: Rows of txt file containing the data
        :return: None
        """
        for line in lines:
            line_arr = re.split("\t", line)
            t = self.create_tuple(line_arr)
            self.tuples = t
        return

    @staticmethod
    @ExceptionHandler.handle_exception(ExceptionHandler.halt_message, TypeError, IOError)
    def create_tuple(split_line):
        """
        Create tuple representing record in the data stream, by extracting the PID, QI identifiers and sensitive attribute
        :param split_line: List of split values of record in a stream
        :return: Formed tuple representing the stream records
        """
        stream_tuple = ()
        timestamp = int(split_line[0])
        quasi_identifier = split_line[1:(len(split_line)-1)]
        quasi_identifier = [float(x) for x in quasi_identifier if x is not str]
        sensitive_attr = split_line[len(split_line)-1]
        stream_tuple = (timestamp, quasi_identifier, sensitive_attr)
        return stream_tuple

    def create_tuples(self, df):
        """
        Create an instance record out of tuple consisting of timestamp, QI and sensitive attribute
        :param df: Dataframe object of data
        :return: List of formed records
        """
        rows = df.to_records(index=False)
        self.tuples = [Record(tuple(x)) for x in rows]
        return self.tuples

    @staticmethod
    @ExceptionHandler.handle_exception(ExceptionHandler.halt_message, TypeError, IOError)
    def change_dtypes(df, datatypes):
        """
        Create dictionary of stream attributes and their corresponding types.
        Contains information about each attribute, including type, weight, max rank (for ordinal variables),
        maximum and minimum values (for numerical variables) and unique values (for nominal variables).
        Numeric attributes provide the MAD (Mean absolute distance) and STD (Standard deviation) statistic metrics.
        :param df: Dataframe object of data
        :param datatypes: Path to datatypes file, containing information about the attributes
        :return: Ordered dictionary of attributes and their characteristics
        """
        f_list = OrderedDict()
        # Iterate through each row and assign variable type.
        # Note: astype is used to assign types
        col_types = pd.read_csv(datatypes, skipinitialspace=True)
        for i, row in col_types.iterrows():  # i: dataframe index; row: each row in series format
            if row['Type'] == "categorical":
                df[row['Feature']] = df[row['Feature']].astype(np.str)
            elif row['Type'] == "numeric":  # Numeric values
                if row['Val_Type'] == "discrete":  # Discrete values
                    df[row['Feature']] = df[row['Feature']].astype(np.int)
                if row['Val_Type'] == "continuous":  # Continuous values
                    df[row['Feature']] = df[row['Feature']].astype(np.float)

            if i in range(1, len(df.columns)):
                value_type = type(df[row['Feature']])
                f_list[row['Feature']] = {'Type': row['Type'],
                                          'Val_Type': row['Val_Type'],
                                          'Weight': row['Weight'],
                                          'Max_Rank': (lambda: row['Max_Rank']
                                                       if not isnan(row['Max_Rank']) else None)(),
                                          'Min_Val': (lambda: row['Min_Val']
                                                       if not np.math.isnan(row['Min_Val']) else None)(),
                                          'Max_Val': (lambda: row['Max_Val']
                                                       if not np.math.isnan(row['Max_Val']) else None)(),
                                          'Distinct_Val': (lambda: row['Distinct_Val'][1:-1].split(',')
                                                       if isinstance(row['Distinct_Val'], basestring) else None)(),
                                          'Mean': (lambda: df[row['Feature']].mean()
                                                       if not row['Type'] == 'categorical' else None)(),
                                          'std': (lambda: df[row['Feature']].std()
                                                       if not row['Type'] == 'categorical' else None)(),
                                          'mad': (lambda: df[row['Feature']].mad()
                                                       if not row['Type'] == 'categorical' else None)()
                                          }
        return f_list

    def reinsert_duplicates(self, fraction):
        """
        Appends specified fraction of records back into dataset as a simulation of tuples with duplicate PID in stream
        :param fraction: Percent of records to be re-inserted (e.g., 10% of dataset size)
        :return: Extended list of tuples forming the data stream
        """
        n = int(float(len(self.tuples)) * fraction)
        tuples_fraction = random.sample(deepcopy(self.tuples), n)
        last_pid = self.tuples[-1].timestamp
        for t in tuples_fraction:
            last_pid += 1
            t.timestamp = last_pid
        self.tuples.extend(tuples_fraction)
        return self.tuples

    @staticmethod
    def shuffle_tuples(df):
        """
        Shuffle records in dataset, in case data is sorted or grouped according to some attribute.
        :param df: Dataframe of dataset
        :return: Shuffled dataframe of dataset
        """
        timestamp_col = (df.iloc[:, [0]]).copy()
        df = shuffle(df.iloc[:, 0:])
        df['Timestamp'] = timestamp_col.values
        df.reset_index(inplace=True, drop=True)
        return df
