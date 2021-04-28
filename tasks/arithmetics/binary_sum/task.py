class SumTask:
    name = 'sum'

    @staticmethod
    def offset(max_len_placeholder):
        """
        Gives offset from which label, or answer, is given in the input

        In particular, we have
        <first number, e.g. [1 0 0 0]>
        <operation marker, e.g. 1>
        <second number, e.g [0 1 0 0]>
        <end marker, e.g. 0>

        :param max_len_placeholder: number of bits required for a single number representation
        :return: required shift
        """
        return 2 * (max_len_placeholder + 1)
