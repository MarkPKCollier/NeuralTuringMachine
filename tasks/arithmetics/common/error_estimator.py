import numpy as np


class BinaryVectorErrorEstimator:
    @staticmethod
    def error_per_seq(labels, outputs, num_seq):
        """
        Better to duplicate the method so that keeping current interface
        :param labels:
        :param outputs:
        :param num_seq:
        :return:
        """
        outputs[outputs >= 0.5] = 1.0
        outputs[outputs < 0.5] = 0.0
        bit_errors = np.sum(np.abs(labels - outputs))
        return bit_errors / num_seq
