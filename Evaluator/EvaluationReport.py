import glob
import logging
from collections import OrderedDict
import os
import pandas as pd
from StreamHandler.StreamWriter import StreamWriter
from Utils.MetaUtils import MetaUtils


class EvaluationReport(object):
    """
    Performance evaluator and builder of full report file
    """

    # Default values of directories and files
    # Directory to which the total values of the evaluation are written
    EVAL_DIR = "Output\\Total_Evaluation\\"
    # Directory to which incremental values (over time) of the evaluation are written
    INCREMENTAL_EVAL_DIR = "Output\\Incremental_Evaluation\\"
    REPORT = "Reports\\report3.csv"
    ORIGINAL = 'original'
    ANONYMIZED = 'anonymized'
    ORIGINAL_CSV = 'original.csv'
    ANONYMIZED_CSV = 'anonymized.csv'
    ORIGINAL_ARFF = 'original.arff'
    ANONYMIZED_ARFF = 'anonymized.arff'
    ANONYMIZED_PAIRS = 'anonymization_pairs.csv'

    def __init__(self, dataset_name, anonymization_pairs, anonymizer, estimators):
        """
        Class constructor - initiate evaluator object
        :param dataset_name: Name of dataset for documenting its results
        :param anonymization_pairs: Instance of RecordPair object containing the original record and its anonymization
        :param anonymizer: Instance of the anonymizer that was used for anonymization
        :param estimators: Dictionary of performance measures and their values
        """
        self.logger = logging.getLogger(__name__)
        self.__dataset_name = dataset_name
        self.EVAL_DIR += dataset_name
        self.__anonymization_pairs = anonymization_pairs
        self.__anonymizer = anonymizer
        self.__stream_size = len(anonymization_pairs)
        self.estimators = estimators
        self.params = {}
        self.unpack_params(anonymizer, estimators)

    @property
    def dataset_name(self):
        """
        Name of dataset for documenting its results
        """
        return self.__dataset_name

    @property
    def anonymization_pairs(self):
        """
        Instance of RecordPair object containing the original record and its anonymization
        """
        return self.__anonymization_pairs

    @property
    def stream_size(self):
        """
        Size of stream (number of records in dataset)
        """
        return self.__stream_size

    @property
    def anonymizer(self):
        """
        Instance of the anonymizer that was used for anonymization
        """
        return self.__anonymizer

    def unpack_params(self, anonymizer, estimators):
        """
        Prepare the CSV header of performance report, and the execution corresponding results
        :param anonymizer: Instance of the anonymizer that was used in the experiments
        :param estimators: Dictionary of performance measures and their values
        :return: None
        """
        self.params = OrderedDict([("Dataset", self.dataset_name),
                                   ("Data Size", len(self.anonymization_pairs)),
                                   ("K_min", str(anonymizer.k[0])),
                                   ("K_max", str(anonymizer.k[-1])),
                                   ("L-Diversity", str(anonymizer.l_diversity)),
                                   ("LC-Diversity", str(anonymizer.c_diversity)),
                                   ("Epsilon", str(anonymizer.epsilon)),
                                   ("Dist Thr", str(anonymizer.dist_thr)),
                                   ("Concept Drift Thr", str(anonymizer.drift_detector.cd_factor) if anonymizer.drift_detector else 'Disabled'),
                                   ("Delta", str(anonymizer.delta)),
                                   ("Buffer Size", str(anonymizer.buffer.max_size)),
                                   ("Execution Time", str(estimators["Execution Time"].get_estimation())),
                                   ("Average Publishing Delay", str(estimators["Average Publishing Delay"].get_estimation())),
                                   ("MSE Info Loss", str(estimators["MSE Info Loss"].get_estimation())),
                                   # ("SSE/SST Info Loss", str(estimators["Homogeneity Info Loss"])),
                                   ("SSE Info Loss", str(estimators["SSE unbounded Info Loss"].get_estimation())),
                                   ("RE Info Loss (Percent)", str(estimators["Relative Percentage Error Info Loss"].get_estimation())),
                                   ("Classification Metric", str(estimators["Classification Metric"].get_estimation())),
                                   ("Disclosure Risk", str(estimators["Disclosure Risk"].get_estimation())),
                                   ("# Suppressed_LC_Diversity", str(anonymizer.total_randomized_diversity)),
                                   ("# Suppressed_K_Size", str(anonymizer.total_randomized_size)),
                                   ("Total Opened Clusters", str(anonymizer.total_opened_clusters))])

        # Statistics of suppressed tuples and number of opened clusters
        self.logger.info("Num of randomized (suppressed) records due to insufficient LC-diversity  = %s" %
                         self.params['# Suppressed_LC_Diversity'])
        self.logger.info("Num of randomized (suppressed) records due to insufficient K-Cluster size = %s" %
                         self.params['# Suppressed_K_Size'])
        self.logger.info("Total opened clusters  = %s" % self.params['Total Opened Clusters'])

        self.EVAL_DIR = '{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}'.format(
            self.EVAL_DIR,
            self.params["K_min"],
            self.params["K_max"],
            self.params["L-Diversity"],
            self.params["LC-Diversity"],
            self.params["Epsilon"],
            self.params["Delta"],
            self.params["Buffer Size"],
            self.params["Dist Thr"],
            self.params["Concept Drift Thr"])

    def print_pairs_to_CSV(self):
        """
        Save record pairs (original records and its anonymized version) to CSV file
        :return: True if writing is successful, otherwise False
        """
        csv_header = ["Original", "Anonymized"]
        data = [{"Original": str(pair.original_record), "Anonymized": str(pair.anonymized_record)} for pair in
                self.anonymization_pairs]
        self.logger.info("Writing record pairs [original, anonymized] to CSV file in path: %s\%s" %
                         (self.EVAL_DIR, self.ANONYMIZED_PAIRS))
        return StreamWriter.write_dict_to_CSV(self.EVAL_DIR, self.ANONYMIZED_PAIRS, csv_header, data)

    def print_records(self):
        """
        Save original and anonymized records separately to designated files, in CSV and ARFF file format
        :return: True if writing is successful, otherwise False
        """
        status = self.print_records_to_CSV()
        if status:
            self.logger.info("Writing original records to ARFF file in path: %s\%s" %
                             (self.EVAL_DIR, self.ORIGINAL_ARFF))
            self.logger.info("Writing original records to ARFF file in path: %s\%s" %
                             (self.EVAL_DIR, self.ORIGINAL_ARFF))

            StreamWriter.convert_CSV_to_ARFF(self.EVAL_DIR, EvaluationReport.ORIGINAL_CSV)
            StreamWriter.convert_CSV_to_ARFF(self.EVAL_DIR, EvaluationReport.ANONYMIZED_CSV)
            return True
        return False

    def print_records_to_CSV(self):
        """
        Save original and anonymized records separately to two CSV files.
        In the anonymized records file, the records are appended in the order of their publication.
        :return: True if writing is successful, otherwise False
        """
        original = [str(pair.original_record).split(',')[1:] for pair in self.anonymization_pairs]
        anonymized = [str(pair.anonymized_record).split(',')[1:] for pair in self.anonymization_pairs]

        header = MetaUtils.stream_attr_names
        data_original = pd.DataFrame(original, columns=header)
        data_anonymized = pd.DataFrame(anonymized, columns=header)

        self.logger.info("Writing original records to CSV file in path: %s\%s" %
                         (self.EVAL_DIR, self.ORIGINAL_CSV))
        self.logger.info("Writing anonymized records to CSV file in path: %s\%s" %
                         (self.EVAL_DIR, self.ANONYMIZED_CSV))
        return StreamWriter.write_df_to_CSV(self.EVAL_DIR, self.ORIGINAL_CSV, data_original) and \
               StreamWriter.write_df_to_CSV(self.EVAL_DIR, self.ANONYMIZED_CSV, data_anonymized)

    def print_post_evaluation(self, task_name, learner_name, measures_origin, measures_anonym, origin_anonym_kappa):
        """
        Save post evaluation performance results on original and anonymized records to report instance.
        Save the inter-performance kappa measure between original and anonymized post evaluation (e.g., classification)
        :param task_name: Task of evaluation (e.g., EvaluatePrequential)
        :param learner_name: Type of classification model learner algorithm (e.g., HoeffdingTree)
        :param measures_origin: Post evaluation performance results on original records
        :param measures_anonym: Post evaluation performance results on anonymized records
        :param origin_anonym_kappa: inter-performance kappa measure between original and anonymized post evaluation
        :return: None
        """
        if measures_origin and measures_anonym:
            for measure in measures_origin:
                param1 = '{0}_{1}_{2}_{3}'.format(measure, 'Original', task_name, learner_name)
                param2 = '{0}_{1}_{2}_{3}'.format(measure, 'Anonymized', task_name, learner_name)
                self.params[param1] = measures_origin[measure]
                self.params[param2] = measures_anonym[measure]

        # Cohen's Kappa between original classification and anonymized classification
        param = '{0}_{1}_{2}'.format('Kappa_Original_Anonymized', task_name, learner_name)
        self.params[param] = origin_anonym_kappa
        return

    def print_report_to_CSV(self):
        """
        Save full report to a designated folder containing execution performance results in CSV format
        :return: True if writing is successful, otherwise False
        """
        csv_header = [key for key in self.params.iterkeys()]
        data = [self.params]

        self.logger.info("Writing evaluation report to CSV file in path: %s%s" %
                         ('.\\', self.REPORT))
        return StreamWriter.write_dict_to_CSV('.\\', self.REPORT, csv_header, data)

    def print_incremental_eval_to_CSV(self):
        """
        Print the incremental changes of each estimator (as a function of time) to CSV file.
        A separate file is created for each estimator, containing two columns: 'time' and 'Value of estimator'.
        :return: None
        """
        k = self.params['K_min']
        eps = self.params['Epsilon']
        l = self.params['L-Diversity']

        # performance over time of estimators: SSE/MSE info loss, disclosure risk, relative error, publishing delay
        for estimator_name, estimator in self.estimators.iteritems():
            if estimator.metric_over_time:
                df = pd.DataFrame.from_records(estimator.metric_over_time, columns=['time', estimator_name])

                file_name = '{0}@{1}_{2}_{3}_{4}.csv'.format(self.dataset_name, k, eps, l, estimator_name)
                StreamWriter.write_df_to_CSV(self.INCREMENTAL_EVAL_DIR, file_name, df)

        #  performance over time of concept drift detector
        if self.anonymizer.drift_detector:
            df = pd.DataFrame.from_records(self.anonymizer.drift_detector.metric_over_time, columns=['time', 'change'])
            file_name = '{0}@{1}_{2}_{3}_{4}.csv'.format(self.dataset_name, k, eps, l, 'Concept Drift')
            StreamWriter.write_df_to_CSV(self.INCREMENTAL_EVAL_DIR, file_name, df)

    @staticmethod
    def print_incremental_comparison_to_CSV(eval_dir, dataset, group_by='k', k=None, eps=None, l=None, estimator=None):
        """
        Compare the incremental change between varying values of k (k-anonymity), l (l-diversity) and eps (epsilon).
        Group all files relating to one parameter from all incremental evaluation files generated.
        Print the combined comparison dataframe to CSV file.
        :param eval_dir: Directory of incremental evaluation files.
        :param dataset: Dataset to be analyzed.
        :param group_by: The privacy parameter to be analyzed (str: k, l, eps, default: k).
        :param k: K-anonymity value (default: None).
        :param eps: Differential privacy value (default: None).
        :param l: L-diversity value (default: None).
        :param estimator: Estimator to analyze.
        :return: Dataframe with each column as performance under different parameter value.
        """
        i = 0
        saving_file_name = '{0}_{1}_{2}_{3}_{4}_groupby_{5}.csv'.format(dataset, k, eps, l, estimator, group_by)
        saving_file_dir = '{0}Merged'.format(eval_dir)
        combined_df = pd.DataFrame()
        if os.path.exists(eval_dir):
            filter_match = '{0}{1}_{2}_{3}_{4}_{5}.csv'.format(eval_dir, dataset, k, eps, l, estimator)
            for f in glob.glob(filter_match.replace('_', '*')):
                i += 1
                k, eps, l, estimator = f.split('\\')[-1].split('@')[-1].split('_')
                df = pd.read_csv(f)
                if not df.columns[0] in combined_df:
                    combined_df['time'] = df.iloc[:, 0].values
                if group_by == 'k':
                    col_name = '{0} = {1}'.format(group_by, k)
                    combined_df[col_name] = df.iloc[:, -1].values
                elif group_by == 'eps':
                    col_name = '{0} = {1}'.format(group_by, eps)
                    combined_df[col_name] = df.iloc[:, -1].values
                elif group_by == 'l':
                    col_name = '{0} = {1}'.format(group_by, l)
                    combined_df[col_name] = df.iloc[:, -1].values
        StreamWriter.write_df_to_CSV(saving_file_dir, saving_file_name, combined_df)
        return combined_df
