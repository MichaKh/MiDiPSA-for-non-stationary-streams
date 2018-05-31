import csv
import logging
import os
from collections import OrderedDict
from Utils.ExceptionHandler import ExceptionHandler
from Utils.MetaUtils import MetaUtils
from Utils.ExternalProcesses import ExternalProcesses


class StreamWriter(object):
    """
    Class implementing writer functions for different types of data to different file types
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @staticmethod
    @ExceptionHandler.handle_exception(ExceptionHandler.writer_message, TypeError, IOError)
    def write_tuple_to_file(dir, file_name, t):
        """
        Writes single tuple of stream to file of specified path
        :param dir: Directory of file path
        :param file_name: Name of target file
        :param t: Tuple representing record in the data stream
        :return: True if write is successful, otherwise False
        """
        path = os.path.join(dir, file_name)
        with open(path, 'a') as f:
            # f.write(''.join(str(s).join(', ') for s in t) + '\n')
            f.write(str(t) + '\n')
            print("Record is appended to file")
        return True

    @staticmethod
    @ExceptionHandler.handle_exception(ExceptionHandler.writer_message, TypeError, IOError)
    def write_dict_to_CSV(dir, file_name, csv_columns, dict_data):
        """
        Writes data in a Dictionary format to CSV file of specified path
        :param dir: Directory of target CSV file path
        :param file_name: Name of CSV file
        :param csv_columns: Definition of CSV file header (columns name)
        :param dict_data: Dictionary object of data
        :return: True if write is successful, otherwise False
        """
        has_header = False
        path = os.path.join(dir, file_name)
        if os.path.isfile(path):
            has_header = True

        with open(path, 'ab') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            if not has_header:
                writer.writeheader()
            for data in dict_data:
                writer.writerow(data)
        return True

    @staticmethod
    @ExceptionHandler.handle_exception(ExceptionHandler.writer_message, TypeError, IOError)
    def write_df_to_CSV(dir, file_name, df):
        """
        Writes data in a dataframe format to CSV file of specified path
        :param dir: Directory of target CSV file path
        :param file_name: Name of CSV file
        :param df: Dataframe object of data
        :return: True if write is successful, otherwise False
        """
        if not os.path.exists(dir):
            os.makedirs(dir)
        df.to_csv(os.path.join(dir, file_name), index=False, quoting=csv.QUOTE_NONE)
        return True

    @staticmethod
    @ExceptionHandler.handle_exception(ExceptionHandler.writer_message, TypeError, IOError)
    def convert_CSV_to_ARFF(dir, file_name):
        """
        Converts file of CSV format to a WEKA / MOA readable ARFF format
        :param dir: Directory of source and target files paths
        :param file_name: Name of file to be converted (dataset name)
        :return: True if conversion is successful, otherwise False
        """
        # Create temproray path for ARFF file, Weka does not interpret attribute types correctly
        arff_file = file_name.split('.')[0] + '.arff'

        csv_path = os.path.abspath(os.path.join(dir, file_name))
        arff_path = os.path.abspath(os.path.join(dir, arff_file))
        weka_csv_class = 'weka.core.converters.CSVLoader'

        #  Correct only categorical attributes, which are interpreted as numerical by Weka
        nominal_indx = MetaUtils.get_all_nominal_indx()
        nominal_indx = ','.join(str(i) for i in nominal_indx)

        ExternalProcesses.run_process(OrderedDict([
                                    ('p_type', 'java'),
                                    ('t_type', 'weka'),
                                    ('jclass', weka_csv_class),
                                    ('path', csv_path),
                                    ('N', nominal_indx),
                                    ('gt', arff_path),
                                    ('B', '')]))

        # Save corrected arrtibute types to new ARFF file and delete temp file
        # file_path = StreamWriter.correct_ARFF_attribute_types(tmp_arff_path)
        # os.remove(tmp_arff_path)
        # return file_path
        return arff_path

    @staticmethod
    @ExceptionHandler.handle_exception(ExceptionHandler.writer_message, TypeError, IOError)
    def correct_ARFF_attribute_types(arff_path):
        """
        Correct definition of categorical attributes in ARFF file, which are interpreted as numerical by Weka
        :param arff_path: Path of ARFF file
        :return: True if correction is successful, otherwise False
        """
        corrected_path = arff_path.replace('_temp', '')
        weka_filter_class = 'weka.filters.unsupervised.attribute.NumericToNominal'

        # Correct only categorical attributes, which are interpreted as numerical by Weka
        # nominal_indx = [i + 1 for i, key in enumerate(sr.STREAM_METADATA)
        #                 if sr.STREAM_METADATA[key]['Type'] == "categorical"]
        nominal_indx = [i + 1 for i in range(0, len(MetaUtils.stream_attr_names))
                        if MetaUtils.get_attr_metadata(i, 'Type') == "categorical"]
        nominal_indx = ','.join(str(i) for i in nominal_indx)

        ExternalProcesses.run_process(OrderedDict([
                                    ('p_type', 'java'),
                                    ('t_type', 'weka'),
                                    ('jclass', weka_filter_class),
                                    ('i', arff_path),
                                    ('o', corrected_path),
                                    ('R', nominal_indx)]))

        return corrected_path
