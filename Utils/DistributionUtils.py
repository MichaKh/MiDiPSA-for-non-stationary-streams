import random
import numpy as np
from sklearn.neighbors import KernelDensity


class DistributionUtils(object):
    """
    Utility class for generating distributions and fitting data to a distribution for further analysis
    """

    @staticmethod
    def get_estimated_rand(dtype, sample_batch):
        """
        Generate random value based on empirical distribution of given sequence of values
        Distribution is estimated using a kernel distribution estimator (KDE)
        :param dtype: Type of variables in the sample batch used for fitting estimation (e.g., int, float)
        :param sample_batch: Sequence of value for fitting a estimated kernel distribution
        :return: Random value sample of the estimated distribution
        """
        try:
            X = np.array(sample_batch).reshape(1, -1)  # Reshape to 1D numpy array
            kde = KernelDensity()  # Scikit-learn KernelDensity estimator
            kde.fit(X)
            est = kde.sample().item(0)
        # In case, no distribution can be estimated (too small sample batch)
        # Estimate appropriate random value selecting randomly one previousy observed value
        except:
            return random.choice(sample_batch)
        return abs(est) if 'float' in str(dtype) else abs(int(est))

    @staticmethod
    def get_uniform_rand(dtype, **params):
        """
        Generate uniform random value.
        For numeric attributes, uses Uniform(a,b) distribution.
        For categorical attributes, randomly picks one nominal values out of given batch.
        :param dtype: Type of variables in the sample batch used for drawing random value (e.g., int, float)
        :param params: Parameters for generation of random values. For uniform distribution uses min and max parameters.
        :return: Random generated numeric or nominal value
        """
        if dtype is str:  # Categorical values
            if 'sample_batch' in params:
                values = params['sample_batch']
                return random.choice(values)
        elif 'min' in params and 'max' in params:  # Numeric values
            min_range = params['min']
            max_range = params['max']
            rand = random.uniform(min_range, max_range)
            return float(rand) if 'float' in str(dtype) else int(rand)