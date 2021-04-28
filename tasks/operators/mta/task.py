class MTATask:
    name = 'mta'

    @staticmethod
    def offset(max_len_placeholder, numbers_count):
        """
        Gives offset from which label, or answer, is given in the input

        In particular, if we have number of experts=3, then the structure will be like this:
        <first number, e.g. [1 0 0 0]>
        <end number marker, e.g. 1>
        <second number, e.g [0 1 0 0]>
        <end number marker, e.g. 1>
        <third number, e.g [0 0 1 0]>
        <end number marker, e.g. 1>
        <result number, e.g. [1 1 1 0]>
        <end marker, e.g. 0>

        :param max_len_placeholder: number of bits required for a single number representation
        :param numbers_count: number of assessments that should be aggregated
        :return: required shift
        """
        raise ValueError('Unknown placeholder')
        return numbers_count * (max_len_placeholder + 1)
