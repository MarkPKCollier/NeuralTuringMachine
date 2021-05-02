import sys

import numpy as np

from tasks.arithmetics.common.binary_arithmetics import BinaryUtils
from tasks.common.error_estimator import BinaryVectorErrorEstimator


def new_empty_placeholder(batch_size, bits_per_number, bits_per_vector):
    return np.zeros(
        (
            batch_size,
            2 * (bits_per_number + 1),  # we sum two numbers, each followed by special marker
            bits_per_vector
        ), dtype=np.float32)


class SumTaskData(BinaryVectorErrorEstimator):
    def generate_batches(self, num_batches, batch_size, bits_per_vector=3, curriculum_point=20, max_seq_len=4,
                         curriculum='uniform', pad_to_max_seq_len=False):
        bits_per_number = max_seq_len
        batches = []
        for i in range(num_batches):
            if curriculum != 'none':
                sys.exit(f'Current "{curriculum}" curriculum is not supported by SumTaskData task')

            # actually just a requirement of the network architecture
            # TODO: need to understand why exactly it requires such a blob
            bits_per_vector_for_inputs = bits_per_vector + 1

            bits_per_vector_for_outputs = bits_per_vector

            inputs, outputs = self._generate_batches(batch_size,
                                                     bits_per_number,
                                                     bits_per_vector_for_inputs,
                                                     bits_per_vector_for_outputs)

            # TODO: should it be a full row of ones as it is in other tasks? Or as in
            # TODO: binary arithmetic paper - just a flag?
            eos = np.ones([batch_size, 1, bits_per_vector_for_inputs])
            output_inputs = np.zeros((batch_size, bits_per_number + 1, bits_per_vector_for_inputs))

            full_inputs = np.concatenate((inputs[:, :-1, :], eos, output_inputs), axis=1)

            batches.append(
                (
                    bits_per_number,
                    full_inputs,
                    outputs
                )
            )
        return batches

    def _generate_batches(self, batch_size, bits_per_number, bits_per_vector_for_inputs,
                          bits_per_vector_for_outputs):
        num1 = np.random.binomial(1, 0.5, (batch_size, bits_per_number))
        num2 = np.random.binomial(1, 0.5, (batch_size, bits_per_number))

        # + 1 is required to satisfy needs of the NTM architecture implementation
        # in this library
        example_input = new_empty_placeholder(batch_size, bits_per_number, bits_per_vector_for_inputs)

        sum_array = np.zeros((batch_size, bits_per_number + 1), dtype=np.float32)
        for i in range(batch_size):
            sum_array[i] = BinaryUtils.sum_binary_numpy(num1, num2, i)

        BinaryUtils.log_generated_sum_sample(num1[0], num2[0], sum_array[0])

        example_input[:, :bits_per_number, 0] = num1
        example_input[:, bits_per_number, 1] = 1  # binary operation encoding (0 1 0) in the original paper

        example_input[:, (bits_per_number + 1):(2 * bits_per_number + 1), 0] = num2
        example_input[:, (2 * bits_per_number + 1), 2] = 1  # end of the second number

        example_output = np.zeros((batch_size, bits_per_number + 1, bits_per_vector_for_outputs))

        example_output[:, :, 0] = sum_array

        return example_input, example_output
