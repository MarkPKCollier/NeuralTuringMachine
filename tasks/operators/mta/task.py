class MTATask:
    name = 'mta'

    @staticmethod
    def offset(max_len_placeholder, numbers_count: int, weight_precision: int, alpha_precision: int):
        """
        Gives offset from which label, or answer, is given in the input

        In particular, if we have number of experts=2, then the structure will be like this:
        <first 2-tuple encoded as TPR, e.g. [1 0 0 0]>
        <end number marker, e.g. 1>
        <second 2-tuple encoded as TPR, e.g. [0 1 0 0]>
        <end number marker, e.g. 1>
        <result 2-tuple encoded as TPR, e.g. [1 1 1 0]>
        <end marker, e.g. 0>

        :param numbers_count: number of assessments that should be aggregated
        :param weight_precision: number of digits after floating point to keep for weights
        :param alpha_precision: number of digits after floating point to keep for alpha
        :return: required shift
        """
        assert alpha_precision == 1, 'Alpha precisions other than 1 are not currently supported'
        assert weight_precision == 1, 'Weight precisions other than 1 are not currently supported'

        return numbers_count * (max_len_placeholder + 1)
