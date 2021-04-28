from math import floor
from typing import List

import numpy as np


class BinaryUtils:
    @staticmethod
    def _from_decimal_to_little_endian_binary(decimal_number: int):
        return bin(decimal_number)[2:][::-1]

    @staticmethod
    def _from_decimal_to_little_endian_binary_numpy(decimal_number: int, bits_per_number):
        bin_str = BinaryUtils._from_decimal_to_little_endian_binary(decimal_number)
        bin_array = np.array([int(i) for i in list(bin_str)], dtype=np.float32)
        res_array = np.zeros((bits_per_number,))
        res_array[:len(bin_array)] = bin_array
        return res_array

    @staticmethod
    def _from_little_endian_binary_to_decimal(binary_number: str):
        return int(binary_number[::-1], 2)

    @staticmethod
    def _from_binary_numpy_to_decimal(binary_number: np.ndarray) -> int:
        bin_str = BinaryUtils._from_binary_numpy_to_binary_str(binary_number)
        return BinaryUtils._from_little_endian_binary_to_decimal(bin_str)

    @staticmethod
    def _from_binary_numpy_to_binary_str(binary_number: np.ndarray) -> str:
        return ''.join([str(round(i)) for i in binary_number.tolist()])

    @staticmethod
    def sum_binary_numpy(a: np.ndarray, b: np.ndarray, index: int) -> np.ndarray:
        indexed_a = a[index]
        indexed_b = b[index]
        dec_a = BinaryUtils._from_binary_numpy_to_decimal(indexed_a)
        dec_b = BinaryUtils._from_binary_numpy_to_decimal(indexed_b)
        sum_array = BinaryUtils._from_decimal_to_little_endian_binary_numpy(dec_a + dec_b,
                                                                            bits_per_number=len(indexed_a) + 1)

        BinaryUtils.log_generated_sum_sample(indexed_a, indexed_b, sum_array)
        return sum_array

    @staticmethod
    def average_sum_binary_numpy(numbers: List[np.ndarray], index: int) -> np.ndarray:
        # adding 1 for overflow bit
        bits_per_number = len(numbers[0][index]) + 1

        indexed_numbers = [numpy_number[index] for numpy_number in numbers]
        dec_numbers = [BinaryUtils._from_binary_numpy_to_decimal(indexed_number) for indexed_number in indexed_numbers]
        dec_sum_result = sum(dec_numbers)
        average_dec_sum_result = floor(dec_sum_result / len(dec_numbers))

        sum_array = BinaryUtils._from_decimal_to_little_endian_binary_numpy(average_dec_sum_result,
                                                                            bits_per_number=bits_per_number)

        BinaryUtils.log_generated_average_sum_sample(indexed_numbers, sum_array)
        return sum_array

    @staticmethod
    def log_generated_sum_sample(a, b, sum_res, is_verbose=True):
        dec_a = BinaryUtils._from_binary_numpy_to_decimal(a)
        dec_b = BinaryUtils._from_binary_numpy_to_decimal(b)
        dec_sum = BinaryUtils._from_binary_numpy_to_decimal(sum_res)

        bin_a = BinaryUtils._from_binary_numpy_to_binary_str(a)
        bin_b = BinaryUtils._from_binary_numpy_to_binary_str(b)
        bin_sum = BinaryUtils._from_binary_numpy_to_binary_str(sum_res)
        if is_verbose:
            print(f'Generated sample (binary version): {bin_a}+{bin_b}={bin_sum}')
            print(f'Generated sample (decimal version): {dec_a}+{dec_b}={dec_sum}')
        assert dec_a + dec_b == dec_sum, 'Binary arithmetic is broken'

    @staticmethod
    def log_generated_average_sum_sample(numbers: List[np.ndarray], sum_res: np.ndarray, is_verbose=False):
        dec_numbers = [BinaryUtils._from_binary_numpy_to_decimal(number) for number in numbers]
        dec_sum = BinaryUtils._from_binary_numpy_to_decimal(sum_res)

        bin_numbers = [BinaryUtils._from_binary_numpy_to_binary_str(number) for number in numbers]
        bin_sum = BinaryUtils._from_binary_numpy_to_binary_str(sum_res)

        if is_verbose:
            bin_summands_str = '+'.join(bin_numbers)
            print(f'Generated sample (binary version): {bin_summands_str}={bin_sum}')
            dec_summands_str = '+'.join([str(i) for i in dec_numbers])
            print(f'Generated sample (decimal version): {dec_summands_str}={dec_sum}')
        assert floor(sum(dec_numbers)/len(dec_numbers)) == dec_sum, 'Binary arithmetic is broken'
