from ANoiseGenerator import ANoiseGenerator
from Noiser.RangeEstimator.CategoricalRangeEstimator import CategoricalNoiseEstimator
from Noiser.RangeEstimator.LaplaceDomainRangeEstimator import LaplaceDomainNoiseEstimator
from Utils.ExceptionHandler import ExceptionHandler
from Utils.MetaUtils import MetaUtils
from Utils.MetricsUtils import MetricsUtils


class DiffPrivateNoiseGenerator(ANoiseGenerator):
    """
    Class for initiating Laplace noiser
    """

    def __init__(self, k, epsilon, noise_thr, loc=0.0, scale=1.0):
        """
        Class constructor - initiate evaluator object
        :param k: K-anonymity parameter
        :param epsilon: Differential privacy parameter epsilon
        :param noise_thr: Threshold for performing the replacement of the noisy categorical value
        :param loc: Location of Laplace distribution (\mu)
        :param scale: Scale (diversity) of Laplace distribution (\b)
        """
        super(DiffPrivateNoiseGenerator, self).__init__(k=k, eps=epsilon, noise_thr=noise_thr)

        self.__loc = loc
        self.__scale = scale
        self.__seed = 12345678
        self.__attribute_scale_estimators = {}

    @property
    def loc(self):
        """
        Location of Laplace distribution
        """
        return self.__loc

    @property
    def scale(self):
        """
        Scale (diversity) of Laplace distribution
        """
        return self.__scale

    @property
    def seed(self):
        """
        Initial seed
        :return:
        """
        return self.__seed

    @property
    def attribute_scale_estimators(self):
        """
        Dictionary of the scale of noise that can be added to each attribute, determined by its domain
        :return:
        """
        return self.__attribute_scale_estimators

    @ExceptionHandler.handle_exception(ExceptionHandler.halt_message)
    def add_noise(self, r):
        """
        Add noise to attributes in record w.r.t its domain and noise scale
        In case the noisy value exceed the domain range, truncate a value inside its domain boundaries
        :param r: Record to which noise is added
        :return: Noisy version of quasi-identifier of record
        """
        qi = list(r.quasi_identifier)
        m = len(qi)
        for i in range(0, m):
            w = MetaUtils.stream_metadata[i]['Weight']
            l = MetaUtils.stream_metadata[i]['Min_Val']
            u = MetaUtils.stream_metadata[i]['Max_Val']

            scale_estimator = self.attribute_scale_estimators.get(i)
            if not scale_estimator:
                if not isinstance(qi[i], str):  # Numerical attribute
                    scale_estimator = LaplaceDomainNoiseEstimator(self.k, m, self.epsilon, self.loc, self.scale)
                else:  # Categorical attribute
                    scale_estimator = CategoricalNoiseEstimator(w, self.noise_thr)
            self.attribute_scale_estimators[i] = scale_estimator

            p = scale_estimator.estimate(qi[i])
            noise = scale_estimator.get_noise(p)
            if not isinstance(qi[i], str):  # Numerical attribute
                x = qi[i] + noise

                # Truncate a value inside its domain boundaries
                # (if value after adding noise lies outside its domain range)
                qi[i] = MetricsUtils.truncate_value(x, l, u, dtype=type(qi[i]))
            else:  # Categorical attribute
                qi[i] = noise

        return qi
