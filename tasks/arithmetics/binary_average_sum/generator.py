import sys
import numpy as np

from tasks.arithmetics.common.binary_arithmetics import BinaryUtils
from tasks.arithmetics.common.error_estimator import BinaryVectorErrorEstimator


def new_empty_placeholder(numbers_quantity, batch_size, bits_per_number, bits_per_vector):
    return np.zeros(
        (
            batch_size,
            numbers_quantity * (bits_per_number + 1),  # we sum N numbers, each followed by special marker
            bits_per_vector
        ), dtype=np.float32)


class AverageSumTaskData(BinaryVectorErrorEstimator):
    def generate_batches(self, num_batches, batch_size, bits_per_vector=3, curriculum_point=20, max_seq_len=4,
                         curriculum='uniform', pad_to_max_seq_len=False, numbers_quantity=3):
        bits_per_number = max_seq_len
        batches = []
        for i in range(num_batches):
            if curriculum != 'none':
                sys.exit(f'Current "{curriculum}" curriculum is not supported by AverageSumTaskData task')

            # actually just a requirement of the network architecture
            # TODO: need to understand why exactly it requires such a blob
            bits_per_vector_for_inputs = bits_per_vector + 1

            bits_per_vector_for_outputs = bits_per_vector

            inputs, outputs = self._generate_batches(numbers_quantity,
                                                     batch_size,
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

    def _generate_batches(self, numbers_quantity, batch_size, bits_per_number, bits_per_vector_for_inputs,
                          bits_per_vector_for_outputs):
        numbers = [np.random.binomial(1, 0.5, (batch_size, bits_per_number)) for _ in range(numbers_quantity)]

        # + 1 is required to satisfy needs of the NTM architecture implementation
        # in this library
        example_input = new_empty_placeholder(numbers_quantity, batch_size, bits_per_number, bits_per_vector_for_inputs)

        sum_array = np.zeros((batch_size, bits_per_number + 1), dtype=np.float32)
        for i in range(batch_size):
            sum_array[i] = BinaryUtils.average_sum_binary_numpy(numbers, i)

        BinaryUtils.log_generated_average_sum_sample([number[0] for number in numbers], sum_array[0])

        for num_index, number in enumerate(numbers):
            number_starts_at = num_index * (bits_per_number + 1)
            number_finishes_at = number_starts_at + bits_per_number

            example_input[:, number_starts_at:number_finishes_at, 0] = number
            example_input[:, number_finishes_at, 1] = 1  # binary operation encoding (0 1 0) in the original paper

        example_output = np.zeros((batch_size, bits_per_number + 1, bits_per_vector_for_outputs))

        example_output[:, :, 0] = sum_array

        return example_input, example_output
