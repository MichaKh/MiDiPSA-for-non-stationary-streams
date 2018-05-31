import numpy as np
import pandas as pd
from scipy.spatial import distance
from collections import Counter
from scipy.spatial.distance import cosine
from sklearn.metrics import cohen_kappa_score
from Utils.ExceptionHandler import ExceptionHandler
from Utils.MetaUtils import MetaUtils


class MetricsUtils(object):
    """
    Utility class for calculating similarity between tuples and clusters
    """

    @staticmethod
    @ExceptionHandler.handle_exception(ExceptionHandler.halt_message)
    def euclidean_distance(x, y):
        """
        Calculate euclidean distance between two normalized vectors containing numeric values only
        d(i,j) = sqrt(((xi1-xj1)^2 + ... + (xip-xjp)^2)
        :param x: First vector
        :param y: Second vector
        :return: Normalized euclidean distance
        """
        normalized_x = MetricsUtils.normalize(x, method='minmax')
        normalized_y = MetricsUtils.normalize(y, method='minmax')
        return distance.euclidean(normalized_x, normalized_y)

    @staticmethod
    # @ExceptionHandler.handle_exception(ExceptionHandler.halt_message)
    def transformed_euclidean_distance(x, y):
        """
        Calculate unit-scaled euclidean distance between two normalized vectors containing numeric values only
        The distance is transformed into a [0..1] metric:
        d(i,j) = sqrt([(xi1-xj1) / (max_val1 - min_val1))^2 + ... + ((xip-xjp) / (max_valp - min_valp))^2] / m)),
        where m is the number of variables in the given vectors.
        :param x: First vector
        :param y: Second vector
        :return: Unit-scaled normalized euclidean distance
        """
        dist = 0
        for i in range(0, len(x)):
            w = MetaUtils.stream_metadata[i]['Weight']
            max_val = MetaUtils.stream_metadata[i]['Max_Val']
            min_val = MetaUtils.stream_metadata[i]['Min_Val']

            dist += MetricsUtils.distance_interval_feature(x[i], y[i], min_val, max_val, w) ** 2

        return np.math.sqrt(dist / len(x))

    @staticmethod
    # @ExceptionHandler.handle_exception(ExceptionHandler.halt_message)
    def distance_all_nominal(x, y):
        """
        Calculate distance between two vectors containing nominal values only
        d(i,j) = (p-m) /p , for m is number of matches, and p is total number of variables
        :param x: First vector
        :param y: Second vector
        :return: Nominal distance
        """
        p = len(x)
        m = sum(map(lambda (a, b): 0 if a == b else 1, zip(x, y)))
        return float(p - m) / p

    @staticmethod
    def distance_nominal_feature(x, y, w_indicator):
        """
        Calculate distance between two nominal values:
        d_xy = 0 if x=y, otherwise d_xy = 1
        :param x: First value
        :param y: Second value
        :param w_indicator: Weight of attribute
        :return: 0 if x=y, otherwise 1
        """
        return 0 if x == y else w_indicator * 1

    @staticmethod
    # @ExceptionHandler.handle_exception(ExceptionHandler.halt_message)
    def distance_interval_feature(x, y, min_val, max_val, w_indicator):
        """
        Calculate distance between two values of interval-based variable:
        abs(x-y) / (max_val - min_val)
        :param max_val: Max value of attribute being evaluated
        :param min_val: Min value of attribute being evaluated
        :param x: First value
        :param y: Second value
        :param w_indicator: Weight of attribute
        :return: Normalized distance
        """
        if max(max_val, min_val) == 0:
            return 0
        else:
            return float(w_indicator) * abs(x-y) / (max_val - min_val)

    @staticmethod
    @ExceptionHandler.handle_exception(ExceptionHandler.halt_message)
    def distance_ordinal_feature(x_rank, y_rank, max_rank, w_indicator):
        """
        Calculate distance between two ordinal variables
        z_if = (r_if - 1)/ (M_f - 1) , for rank r_if in {1,...,M_f)
        :param x_rank: Rank of first value
        :param y_rank: Rank of second value
        :param max_rank: Maximum rank of variable
        :param w_indicator: Weight of attribute
        :return: Interval-based distance distance between the ranked representations
        """
        zfx = float(int(x_rank) - 1) / (int(max_rank)-1)
        zfy = float(int(y_rank) - 1) / (int(max_rank) - 1)
        return w_indicator * MetricsUtils.distance_interval_feature(zfx, zfy, 1, max_rank, w_indicator)

    @staticmethod
    # @ExceptionHandler.handle_exception(ExceptionHandler.halt_message)
    def distance(x, y, normalize=True):
        """
        Calculate distance between two tuples (vectors)
        :param normalize: Whether normalization is required for calculating distance (Default: True)
        :param x: First tuple
        :param y: Second tuple
        :return: Distance between possibly mixed types, considering the attributes' weights
        """
        f_list_len = len(MetaUtils.stream_metadata) - 1  # Ignoring the last class attribute

        dist, sum_weights = 0, 0

        if MetaUtils.is_all_nominal:
            return MetricsUtils.distance_all_nominal(x, y)

        elif MetaUtils.is_all_numeric:
            return MetricsUtils.transformed_euclidean_distance(x, y)
        else:
            for i in range(0, f_list_len):
                dtype, w, max_rank, min_val, max_val, distinct_val = MetaUtils.get_all_attr_metadata(i)

                sum_weights += w
                if dtype == "ordinal":
                    dist += MetricsUtils.distance_ordinal_feature(x[i], y[i], max_rank, w)
                elif dtype == "continuous" or dtype == "discrete":
                    dist += MetricsUtils.distance_interval_feature(x[i], y[i], min_val, max_val, w)
                elif dtype == "nominal":
                    dist += MetricsUtils.distance_nominal_feature(x[i], y[i], w)
            return dist / sum_weights

    @staticmethod
    def distance_unbounded(x, y):
        dist = 0
        for i, v in enumerate(x):
            if not isinstance(x[i], (int, float)):
                if x[i] != y[i]:
                    dist += 1
            else:
                dist += np.array((x[i]-y[i]), dtype='float64') ** 2
        return np.sqrt(dist)

    @staticmethod
    @ExceptionHandler.handle_exception(ExceptionHandler.halt_message)
    def relative_error(x, y):
        """
        Calculate error between original value and the masked value, for each attribute in given vector.
        1) For numeric attribute i: | a_j - a'_j | / max(DOM(A_j).
        2) For categorical attribute i: dist(a_j, a'_j) / m,  for nominal distance, normalized  by number of attributes m.
        Error is summed over all attributes.
        :param x: First vector
        :param y: Second vector
        :return: Sum of relative error for given pair of vectors
        """
        diff = 0
        length = len(x)
        pair_wise_list = list(zip(*[x, y]))
        for idx, pair in enumerate(pair_wise_list):
            if isinstance(pair[0], (int, float)):  # numeric attribute
                min_val = MetaUtils.get_attr_metadata(idx, 'Min_Val')
                max_val = MetaUtils.get_attr_metadata(idx, 'Max_Val')
                if pair[0] != 0 or pair[1] != 0:
                    diff += float(abs(pair[0] - pair[1])) / (max_val - min_val)

            elif isinstance(pair[1], str):  # Categorical attribute
                dtype = MetaUtils.get_attr_metadata(idx, 'Val_Type')
                w = MetaUtils.get_attr_metadata(idx, 'Weight')
                max_rank = MetaUtils.get_attr_metadata(idx, 'Max_Rank')
                if dtype == "ordinal":
                    diff += MetricsUtils.distance_ordinal_feature(pair[0], pair[1], max_rank, w) / float(length)
                elif dtype == "nominal":
                    diff += MetricsUtils.distance_nominal_feature(pair[0], pair[1], w) / float(length)
        return diff

    @staticmethod
    @ExceptionHandler.handle_exception(ExceptionHandler.halt_message)
    def cosine_distance(x, y):
        """
        Calculate cosine distance between two numeric vectors, using the Scipy package
        Cosine distance is defined as 1.0 minus the cosine similarity
        1- [<X, Y> / (||X||*||Y||)]
        :param x: First tuple
        :param y: Second tuple
        :return: Cosine distance
        """
        x = np.array(x).reshape(-1, 1)
        y = np.array(y).reshape(-1, 1)
        s = cosine(x, y)
        return s

    @staticmethod
    @ExceptionHandler.handle_exception(ExceptionHandler.halt_message)
    def calculate_centroid(records):
        """
        Calculate center of list of Record instances (centroid).
        Mean is calculated for numerical attributes, and mode is calculated for categorical attributes.
        :param records: List of records to be summed.
        :return: Centroid vector of records.
        """
        centr = []
        vectors = map(lambda x: x.quasi_identifier, records)
        zipped = list(zip(*vectors))
        for l in zipped:
            if isinstance(l[0], str):
                centr.append(Counter(l).most_common(1)[0][0])
            elif isinstance(l[0], (int, float)):
                centr.append(np.array(l).mean())
        return centr

    @staticmethod
    @ExceptionHandler.handle_exception(ExceptionHandler.halt_message)
    def update_centroid(old_centr, new_record, size, cat_counter):
        """
        Update the cluster centroid incrementally given a new tuple assigned to the cluster
        For numeric values : (prev_val * N + new_val) / N+1
        For nominal values: Compute the mode of all nominal values combined with new value
        :param old_centr: Current centroid to update
        :param new_record: New arriving tuple to cluster
        :param size: Current size of cluster
        :param cat_counter: Frequency count of nominal values in cluster
        :return: Updated cluster centroid
        """
        centr = []
        for i, val in enumerate(old_centr):
            if isinstance(val, (int, float)):
                centr.append((val * size + new_record[i]) / (size + 1))
            elif isinstance(val, str):
                mode = max(cat_counter[i], key=cat_counter[i].get)
                centr.append(mode)
        return centr

    @staticmethod
    @ExceptionHandler.handle_exception(ExceptionHandler.halt_message)
    def build_numeric_vec(z):
        """
        Convert non-numeric vector to categorical vector, replacing nominal value with a binary vector
        Each nominal attribute is replaced with (0,....,0,1,0...,0) vector of length corresponding to number of unique values
        Numeric representation is used for calculating distance/similarity with numeric similarity functions
        :param z: Non-numeric vector to be converted
        :return:
        """
        v = []
        m = len(z) if len(z) else None
        if m:
            for i in range(0, m):
                if not isinstance(z[i], str):  # Numerical attribute
                    v.append(z[i])
                else:  # Categorical attribute
                    distinct_vals = MetaUtils.get_attr_metadata(i, 'Distinct_Val')
                    cat = [1 if z[i] == d else 0 for d in distinct_vals]
                    v.extend(cat)
        return v

    @staticmethod
    @ExceptionHandler.handle_exception(ExceptionHandler.halt_message)
    def truncate_value(v, l, u, dtype):
        """
        Truncate a value inside its domain boundaries (if value after adding noise falls outside its domain range)
        :param v: Value to truncate
        :param l: Lower bound of domain
        :param u: Upper bound of domain
        :param dtype: Type of attribute to truncate
        :return: Truncated value
        """
        if l <= v <= u:
            # return v
            trunc_val = v
        elif v < l:
            # return l
            trunc_val = l
        else:
            # return u
            trunc_val = u
        return float(trunc_val) if 'float' in str(dtype) else int(trunc_val)

    @staticmethod
    @ExceptionHandler.handle_exception(ExceptionHandler.halt_message)
    def normalize(v, method='minmax', new_min=0, new_max=1):
        """
        Normalize vector of numeric values (not in [0,1] range) to standardized measurement (z-score),
        for efficient distance calculations
        Use mean absolute deviation or std unit-variance for variable for each numeric variable f
        z_if = (x_if - mean_f)/mad_f
        :param v: Vector to be standardized
        :param method: Method used for normalization:
         'minmax': linear transformation on the original data to the range [new minA,new maxA]
                  [(val - min_val) / (max_val - min_val)] * (new_max - new min) + new_min
         'zscore_std': z-score normalization for zero-mean, and unit-variance values) denominator is standard deviation
                  (val - mean) / std
         'zscore_mad': z-score normalization for zero-mean, and unit-variance values) denominator is mean absolute deviation
                  (val - mean) / mad
        :param new_max: New maximum value of range to normalize to (Default: 1).
        :param new_min: New minimum value of range to normalize to (Default: 0).
        :return: Standardized vector
        """
        standardize = []
        for i, val in enumerate(v):
            if not isinstance(val, str):
                if method == 'minmax':  # linear transformation on the original data to the range [new minA,new maxA]
                    min_val = MetaUtils.get_attr_metadata(i, 'Min_Val')
                    max_val = MetaUtils.get_attr_metadata(i, 'Max_Val')
                    standardize.append((float(val - min_val) / (max_val - min_val)) * (new_max - new_min) + new_min)
                elif method == 'zscore_std':
                    mean = MetaUtils.get_attr_metadata(i, 'Mean')
                    s = MetaUtils.get_attr_metadata(i, 'std')
                    standardize.append((val - mean) / s)
                elif method == 'zscore_mad':
                    mean = MetaUtils.get_attr_metadata(i, 'Mean')
                    s = MetaUtils.get_attr_metadata(i, 'mad')
                    standardize.append((val - mean) / s)
            else:
                standardize.append(val)
        return standardize

    @staticmethod
    @ExceptionHandler.handle_exception(ExceptionHandler.evaluation_message)
    def calculate_kappa(pred1, pred2):
        """
        Calculate kappa agreement measure between two classification annotators (in percents)
        (p0 - pc) / (1.0 - pc), where p0 is the observed agreement, and pc is the expected probability of chance agreement.
        :param pred1: First classification annotator
        :param pred2: Second classification annotator
        :return: Kappa (in [-100, 100] percentage interval)
        """
        if pred1 and pred2:
            df1 = pd.read_csv(pred1, sep=',', header=None)
            df2 = pd.read_csv(pred2, sep=',', header=None)
            pred_list1 = df1.ix[:, 0].tolist()
            pred_list2 = df2.ix[:, 0].tolist()
            kappa = cohen_kappa_score(pred_list1, pred_list2)
            return kappa * 100
        else:
            return 0
