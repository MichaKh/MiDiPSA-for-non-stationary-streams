import logging
import os
import shutil
import sys
from Anonymizer.MicroaggAnonymizer import MicroaggAnonymizer
from ConceptDriftHandler.ConceptDriftDetector import ConceptDriftDetector
from Evaluator.EvaluationReport import EvaluationReport
from Evaluator.PostEvaluator.ClassifierEvaluator import ClassifierEvaluator
from Noiser.DiffPrivateNoiseGenerator import DiffPrivateNoiseGenerator
from PerformanceEstimators.DisclosureRiskMetric.BufferedDisclosureRiskMetric import BufferedDisclosureRiskMetric
from PerformanceEstimators.ExecutionTimeMetric.ExecutionTimeMetric import ExecutionTimeMetric
from PerformanceEstimators.ExecutionTimeMetric.PublishingDelayTimeMetric import PublishingDelayTimeMetric
from PerformanceEstimators.InfoLossMetric.ClassificationInfoLossMetric import ClassificationInfoLossMetric
from PerformanceEstimators.InfoLossMetric.HomogeneityInfoLossMetric import HomogeneityInfoLossMetric
from PerformanceEstimators.InfoLossMetric.MSEInfoLossMetric import MSEInfoLossMetric
from PerformanceEstimators.InfoLossMetric.SSEInfoLossMetric import SSEInfoLossMetric
from PerformanceEstimators.InfoLossMetric.RelativeErrorInfoLossMetric import RelativeErrorInfoLossMetric
from Publisher.RandomizedPublisher import RandomizedPublisher
from Publisher.SmartCentroidPublisher import SmartCentroidPublisher
from StreamHandler.StreamReader import StreamReader
from Utils.MetricsUtils import MetricsUtils


DIR = 'Datasets_small'

# Referenced in: A. Bifet and R. Gavalda, "Adaptive Learning from Evolving Data Streams", 2009.
# Simulate concept drift by ordering the datasets by one of its attributes:
# For Adult dataset (with all attributes): order by education attribute.
# For Poker-hand dataset: order by first attribute (r1)
# ** Real data with synthetic drift **
datasets = ['Adult_1_numeric_only_class_50K',  # dist ~ 0.2 (best=0.6)
            'Adult_2_numerical_categorical_class_50K_drift',  # dist ~ 0.4 (best=0.5)
            'sea',  # dist ~ 0.4 (best=0.3)
            'airlines']  # dist ~ 0.5
datatypes = ['datatypes_adult_1_class_50K',
             'datatypes_adult_2_class_50K',
             'datatypes_sea',
             'datatypes_airlines']
estimators = ['Average Publishing Delay',
              'Disclosure Risk',
              'MSE Info Loss',
              'SSE unbounded Info Loss']

classification_task = {'EvaluatePrequential': 'EvaluatePrequential'}
classification_evaluation = {'ImbalancedWindowEvaluator': '(WindowAUCImbalancedPerformanceEvaluator -w 100)'}
classification_learner = {'MajorityClass': 'functions.MajorityClass',
                          'HoeffdingAdaptiveTree': 'trees.HoeffdingAdaptiveTree',
                          'LeveragingBag': '(meta.LeveragingBag -l trees.HoeffdingAdaptiveTree)',
                          'NaiveBayes': 'bayes.NaiveBayes',
                          'AdaptiveRandomForest': 'meta.AdaptiveRandomForest',
                          'SGD': '(functions.SGD -l 0.05 -r 0.005)'}


def main():
    """
    1) The Adult dataset aims to predict whether a person makes over 50k a year, based on census data.
        Adult consists of 30,162 instances, 11 attributes (6 continuous and 5 nominal) after removing records with missing values.
        The dataset is divided into two separate datasets:
        * Only numeric attributes: age, fnlwgt, education.num, capital.gain, capital.loss, hours.per.week --> class {<=50K, >50K}
        * All numeric and 5 more nominal attributes: education, marital.status, workclass, native.country, occupation --> class {<=50K, >50K}
        From : Lichman, M. (2013). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
            Irvine, CA: University of California, School of Information and Computer Science

    2) SEA (Streaming Ensemble Algorithm) synthetic Dataset with 60,000 examples, 3 attributes and 3 class labels.
        Attributes are numeric between 0 and 10. There are four concepts, 15,000 examples each. Contains concept drift.
        * Attributes: Attr1, Attr2, Attr3 --> class {0,1}
        From: W. Street, Y. Kim, "A streaming ensemble algorithm (SEA) for large- scale classification", 2001

    3) Airlines dataset contains flight arrival and departure details for all the commercial flights within the USA, from October 1987 to April 2008.
        The task is to predict whether a given flight will be delayed, given the information of the scheduled departure.
        This reduced dataset contains 539,383 instances, with 7 attributes. Contains concept drift.
        Inspired in the regression dataset from Elena Ikonomovska
        * Attributes: Airline, Flight, AirportFrom, AirportTo, DayOfWeek, Time, Length --> Delay {0, 1}
        From: E. Ikonomovska, "Learning model trees from evolving data streams", 2010
        [http://kt.ijs.si/elena_ikonomovska/data.html]
    """

    for idx, dataset in enumerate(datasets):
        stream_path = '%s.csv' % dataset
        datatypes_path = '%s.csv' % datatypes[idx]
        k_anonymity = [20,
                       50,
                       100,
                       200,
                       400,
                       800]
        l_diversity = [2, 2, 2,
                       2, 2, 2, 2, 2]
        c = 7
        eps = [0.01,
               0.05,
               0.1,
               1]

        start_dist_thr = [0.6,
                          0.5,
                          0.5,
                          0.06,
                          0.3,
                          0.65,
                          0.5,
                          0.05]

        # Factor for multiplying the KS statistic in the two-samples KS test (detection of concept change)
        # The factors are set according to buffer with size equals to k-anonymity parameter.
        cd_factor = [1.2,
                     1.1,
                     0.8,
                     1.2,
                     2.8,
                     2,
                     0.37,
                     10.1
                     ]
        noise_thr = 0.1

        dist = start_dist_thr[idx]
        for e in eps:
            for k in k_anonymity:
                for l in range(2, l_diversity[idx]+1):
                    run(log_file="Run_Log1.log",
                        dir=DIR,
                        stream_path=stream_path,
                        datatypes_path=datatypes_path,
                        k=range(k, 4 * k),
                        l=l,
                        c=c,
                        eps=e,
                        b=3 * k,
                        delta=10 * k,
                        dist_thr=dist,
                        cd_thr=cd_factor[idx],
                        cd_conf=0.1,
                        noise_thr=noise_thr)

        for estimator in estimators:
            EvaluationReport.print_incremental_comparison_to_CSV(eval_dir="Output\\Incremental_Evaluation\\",
                                                                 dataset=datasets[5],
                                                                 group_by='k',
                                                                 k='_',
                                                                 eps=0.01,
                                                                 l=2,
                                                                 estimator=estimator)

            EvaluationReport.print_incremental_comparison_to_CSV(eval_dir="Output\\Incremental_Evaluation\\",
                                                                 dataset=datasets[5],
                                                                 group_by='eps',
                                                                 k=100,
                                                                 eps='_',
                                                                 l=2,
                                                                 estimator=estimator)


def run(log_file, dir, stream_path, datatypes_path, k, l, c, eps, b, delta, dist_thr, cd_thr, cd_conf, noise_thr):
    logging.basicConfig(filename=log_file,
                        filemode='w',
                        stream=sys.stdout,
                        level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.getLogger("").addHandler(logging.StreamHandler(sys.stdout))

    logging.info("---------- Program started ----------")
    logging.info("Dataset: %s" % stream_path.split('.')[0])

    fs = StreamReader(dir, stream_path, datatypes_path)
    features = fs.read_csv_file(shuffle=False, duplicate_frac=None)

    logging.info("Preparation of stream dataset completed!")
    logging.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

    exec_time_estimator = ExecutionTimeMetric()
    average_publishing_delay_estimator = PublishingDelayTimeMetric()
    mse_info_loss_estimator = MSEInfoLossMetric()
    sse_unbounded_info_loss_estimator = SSEInfoLossMetric()
    relative_err_info_loss = RelativeErrorInfoLossMetric()
    classification_info_loss_estimator = ClassificationInfoLossMetric()
    disclosure_risk_estimator = BufferedDisclosureRiskMetric(buffer_size=100)

    logging.info("Initializing anonymizer...")
    try:
        assert (delta >= k[0])
        assert (k[0] <= b <= k[-1] + 1)
    except AssertionError:
        print("Size of buffer should be between {0} and {1}".format(k[0], k[-1] + 1))
        exit(1)

    anonymizer = MicroaggAnonymizer(stream=fs.tuples,
                                    k=k,
                                    l=l,
                                    c=c,
                                    eps=eps,
                                    b=b,
                                    delta=delta,
                                    dist_thr=dist_thr,
                                    datatypes=features,
                                    publisher=SmartCentroidPublisher(),
                                    noiser=DiffPrivateNoiseGenerator(epsilon=eps, k=k[0], noise_thr=noise_thr),
                                    change_detector=ConceptDriftDetector(conf=cd_conf, buff_size=b, factor=cd_thr),
                                    estimators=[exec_time_estimator,
                                                average_publishing_delay_estimator,
                                                mse_info_loss_estimator,
                                                sse_unbounded_info_loss_estimator,
                                                relative_err_info_loss,
                                                classification_info_loss_estimator,
                                                disclosure_risk_estimator])

    anonymization_pairs = anonymizer.anonymize()

    if not anonymization_pairs:
        logging.info("Failed to anonymize. Program aborts!")
        return

    logging.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    logging.info("Calculating performance report to CSV file")

    estimators = {"Execution Time": exec_time_estimator,
                  "Average Publishing Delay": average_publishing_delay_estimator,
                  "MSE Info Loss": mse_info_loss_estimator,
                  "SSE unbounded Info Loss": sse_unbounded_info_loss_estimator,
                  "Relative Percentage Error Info Loss": relative_err_info_loss,
                  "Classification Metric": classification_info_loss_estimator,
                  "Disclosure Risk": disclosure_risk_estimator
                  }

    logging.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    logging.info("Initiating evaluation report...")
    eval_report = EvaluationReport(dataset_name=stream_path.split('.')[0],
                                   anonymization_pairs=anonymization_pairs,
                                   anonymizer=anonymizer,
                                   estimators=estimators)

    eval_report.print_records()

    eval_report.print_incremental_eval_to_CSV()

    logging.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    logging.info("Performing post-analysis evaluation using stream classifier")

    for learner in classification_learner:
        task = classification_task.keys()[0]
        evaluator = classification_evaluation.keys()[0]
        classifier = ClassifierEvaluator(task=(task, classification_task[task]),
                                         learner=(learner, classification_learner[learner]),
                                         evaluator=(evaluator, classification_evaluation[evaluator]))

        #  Evaluate original performance
        pred_origin, measures_origin = classifier.evaluate(dir=eval_report.EVAL_DIR,
                                                           input=eval_report.ORIGINAL_ARFF,
                                                           stream_size=eval_report.stream_size)

        #  Evaluate anonymized performance
        pred_anonym, measures_anonym = classifier.evaluate(dir=eval_report.EVAL_DIR,
                                                           input=eval_report.ANONYMIZED_ARFF,
                                                           stream_size=eval_report.stream_size)

        #  Evaluate difference between original and anonymized performance
        origin_anonym_kappa = MetricsUtils.calculate_kappa(pred_origin, pred_anonym)
        eval_report.print_post_evaluation(task_name=task,
                                          learner_name=learner,
                                          measures_origin=measures_origin,
                                          measures_anonym=measures_anonym,
                                          origin_anonym_kappa=origin_anonym_kappa)

    # Save evaluation report to directory of current run, and save the log file
    logging.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    result = eval_report.print_report_to_CSV()
    logging.info("Done!") if result else logging.info("Failed to save report!")

    # Shutdown all loggers and flush their handlers.
    # Save the log from current run to the designated directory.
    logging.shutdown()
    while logging.getLogger("").handlers:
        logging.getLogger("").handlers.pop()
    shutil.move(log_file, os.path.abspath(os.path.join(eval_report.EVAL_DIR, log_file)))


if __name__ == "__main__":
    main()
