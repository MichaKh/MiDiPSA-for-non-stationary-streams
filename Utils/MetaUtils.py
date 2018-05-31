
class MetaUtils(object):

    # information on each attribute in the dataset and its type
    stream_metadata = None
    # stream attributes names
    stream_attr_names = None
    # Are all attributes in stream numeric?
    is_all_nominal = False
    # Are all attributes in stream categorical?
    is_all_numeric = False
    # centroid of entire dataset, used for calculating SSG
    dataset_centroid = []

    @staticmethod
    def get_attr_metadata(idx, meta):
        """
        Get specified metadata of given attribute at index in STREAM_METADATA Dictionary,
        (e.g., Min/Max values, statistics and distinct values of attribute)
        :param idx: Index of attribute in STREAM_METADATA Dictionary
        :param meta: The meta property of attribute
        :return: Value of given property of attribute
        """
        if meta in MetaUtils.stream_metadata[idx]:
            return MetaUtils.stream_metadata[idx][meta]
        return None

    @staticmethod
    def get_all_attr_metadata(idx):
        """
        Get all relevant metadata of given attribute at index in STREAM_METADATA Dictionary,
        according to its type (numeric / nominal / ordinal).
        (e.g., Min/Max values, statistics and distinct values of attribute).
        :param idx: Index of attribute in STREAM_METADATA Dictionary.
        :return: All values of attribute metadata.
        """
        dtype = MetaUtils.stream_metadata[idx]['Val_Type']
        w = MetaUtils.stream_metadata[idx]['Weight']
        max_rank = MetaUtils.stream_metadata[idx]['Max_Rank']
        max_val = MetaUtils.stream_metadata[idx]['Max_Val']
        min_val = MetaUtils.stream_metadata[idx]['Min_Val']
        distinct_val = MetaUtils.stream_metadata[idx]['Distinct_Val']

        return dtype, w, max_rank, min_val, max_val, distinct_val

    @staticmethod
    def get_all_nominal_indx():
        nominal_indx = []
        for i in range(0, len(MetaUtils.stream_metadata)):
            if MetaUtils.stream_metadata[i]['Type'] == "categorical":
                nominal_indx.append(i + 1)
        return nominal_indx

    @staticmethod
    def validate_dtype(val, attr_idx):
        """
        Check the dtype of value of a given attribute at index, and compare it with the stream metadata
        Converts continuous variables to float type, and discrete variables to int
        :param val: Value to be validated
        :param attr_idx: Index of attribute in stream metadata dictionary
        :return: Converted value
        """
        f_type = MetaUtils.stream_metadata[attr_idx]['Val_Type']
        # Round float numbers to int, for preserving the utility of original data types
        if f_type and f_type == 'discrete':
            val = int(val)
        elif f_type == 'continuous':
            val = float(val)
        return val

    @staticmethod
    def check_is_all_nominal(*a):
        """
        Check if vector/s consist of nominal values only
        :param a: List of vector/s to check
        :return: True if all values are nominal, otherwise False
        """
        vectors = list(a)
        return all(all(isinstance(x, str) for x in v) for v in vectors)

    @staticmethod
    def check_is_all_interval(*a):
        """
        Check if vector/s consist of numeric interval-based values only
        :param a: List of vector/s to check
        :return: True if all values are numeric interval-based, otherwise False
        """
        vectors = list(a)
        return all(all(isinstance(x, (int, float)) for x in v) for v in vectors)
