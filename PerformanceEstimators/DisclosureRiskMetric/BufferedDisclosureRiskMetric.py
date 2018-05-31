import logging
from PerformanceEstimators.DisclosureRiskMetric.ADisclosureRiskMetric import ADisclosureRiskMetric
from Utils.MetricsUtils import MetricsUtils


class BufferedDisclosureRiskMetric(ADisclosureRiskMetric):
    """
    Class implementing buffered individual record linker to estimate the risk of records re-identification.
    Reference from: D. M. Rodriguez et al.,"Towards the adaptation of SDC methods to stream mining", 2017
    """

    def __init__(self, buffer_size):
        """
        Class constructor - initiate the disclosure risk estimator
        :param buffer_size: Size of re-identification buffer (holds original instances)
        """
        super(BufferedDisclosureRiskMetric, self).__init__()
        self.logger = logging.getLogger(__name__)

        self.__linkage_prob_sum = self.current_metric
        self.__incremental_linkage_prob = self.incremental_metric
        self.__buffer_size = buffer_size
        self.__original_instances_buffer = []

    @property
    def original_instances_buffer(self):
        """
        Current re-identification buffer (holds original instances)
        :return:
        """
        return self.__original_instances_buffer

    @original_instances_buffer.setter
    def original_instances_buffer(self, record):
        """
        Adds the given instance in the re-identification buffer and discards older instances if necessary.
        :param record: Records to be added to current buffer
        :return:
        """
        if record is not list():
            if len(self.original_instances_buffer) >= self.__buffer_size:
                del self.original_instances_buffer[-1]  # remove the last one
            self.__original_instances_buffer.append(record)
            self.processed_instances += 1
        else:
            self.__original_instances_buffer = []

    def __str__(self):
        return str(self.get_disclosure_risk())

    def get_disclosure_risk(self):
        """
        Get disclosure risk of re-identification.
        DR =  sigma(x in X) {P(x')} / N, where N is the total number of processed instances.
        Disclosure risk is estimated in the [0..1] range.
        :return: Disclosure risk
        """
        disclosure_risk = self.__linkage_prob_sum / float(self.processed_instances)

        self.logger.info("Disclosure risk Metric: %0.3f" % disclosure_risk)
        return disclosure_risk

    def restart(self):
        """
        Initiate the disclosure risk estimator
        :return: None
        """
        self.__linkage_prob_sum = 0.0
        self.processed_instances = 0
        self.original_instances_buffer = []

    def get_nearest_records(self, anonymized_record):
        """
        Calculate distance between the anonymized instance being re-identified to instances in original buffer.
        Each instance xi in buffer is stored in a set G if the distance d to x' is the minimum distance.
        Once an instance at distance d < A is found, all instances from G are removed and A is updated.
        Finally, the algorithm checks if the target instance is in G.
        :param anonymized_record: Records to be re-identified.
        :return: Indices of identified records if such are found, otherwise None.
        """
        # initialization
        # iterator = iter(self.original_instances_buffer or [])
        # original_record = next(iterator, None)
        original_record = self.original_instances_buffer[0]
        if original_record:
            minimum = MetricsUtils.distance(anonymized_record.quasi_identifier, original_record.quasi_identifier)
            indices = list()
            indices.append(original_record.timestamp)

            # traversal
            for r in self.original_instances_buffer[1:]:
                distance = MetricsUtils.distance(anonymized_record.quasi_identifier, r.quasi_identifier)
                if distance < minimum:
                    minimum = distance
                    indices = list()
                    indices.append(r.timestamp)
                elif distance == minimum:
                    indices.append(r.timestamp)
            return indices
        return

    def estimate_record_linkage_prob(self, anonymized_record):
        """
        Estimate the probability of re-identification, using the buffer and collected group G of original instances.
        P(x') = 0 if x not in G, 1/ |G| if x is in G.
        :param anonymized_record: Records to be re-identified.
        :return: None
        """
        nearest_records = self.get_nearest_records(anonymized_record)
        target_record = self.original_instances_buffer[-1]
        if target_record.timestamp in nearest_records:
            self.__linkage_prob_sum += float(1.0 / float(len(nearest_records)))
            self.current_metric = self.__linkage_prob_sum / float(self.processed_instances)

    def update_estimation(self, time, record_pair, cluster=None):
        """
        Add the original instance to the buffer and estimate the record linkage of the anonymized instance.
        :param time: Current time step in stream (last assigned tuple)
        :param record_pair: Pair of original instance and its anonymization.
        :param cluster: Cluster from which the record is published (Default: not needed).
        :return: None
        """
        last_disclosure_risk = self.__linkage_prob_sum
        # adds the instance, keeping the buffer with a maximum fixed size
        self.original_instances_buffer = record_pair.original_record
        self.estimate_record_linkage_prob(record_pair.anonymized_record)

        # Update the incremental disclosure risk change
        self.__incremental_linkage_prob = self.current_metric - last_disclosure_risk
        self.incremental_metric = self.__incremental_linkage_prob / self.processed_instances

        # Store the incremental DR over time, for time series plot
        # self.monitor_overtime_change(cluster.W_curr.max_size, show_incremental=False)
        self.monitor_overtime_change(window_size=100, show_incremental=False)
