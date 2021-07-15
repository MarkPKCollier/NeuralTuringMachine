import argparse
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from constants import TMP_ARTIFACTS_PATH


def create_chart(ticks, granularity=1000, type='error', bits_per_number=4, series={}):
    font_settings = {
        'fontname': 'Times New Roman',
        'fontsize': 12
    }

    if type == 'error':
        y_title = 'error'
        chart_title = 'Error per sequence'
        legend_title = 'Error'
    else:  # it is loss
        y_title = 'loss'
        chart_title = 'Total loss'
        legend_title = 'Loss'

    new_ticks = [i for i in ticks if i % granularity == 0]
    filtered_data_df = pd.DataFrame({
        'x': ticks,
        **series
    })
    filtered_data_df = filtered_data_df[filtered_data_df['x'] <= new_ticks[-1]]
    plt.close()

    CB91_Blue = '#2CBDFE'
    CB91_Green = '#47DBCD'
    CB91_Pink = '#F3A0F2'
    CB91_Purple = '#9D2EC5'
    CB91_Violet = '#661D98'
    CB91_Amber = '#F5B14C'

    color_list = [CB91_Blue, CB91_Pink, CB91_Green, CB91_Amber,
                  CB91_Purple, CB91_Violet]
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)

    for key in series:
        plt.plot('x', key, data=filtered_data_df, label=key)

    plt.ylabel(y_title, **font_settings)
    plt.xlabel('training steps', **font_settings)
    plt.locator_params(nbins=5)
    # plt.xticks(new_ticks, **font_settings)
    plt.yticks(**font_settings)
    plt.title(f'NTM training dynamics for MTA \noperator for two assessments, {chart_title}')
    plt.legend()
    return plt


def save_chart(plot, type, bits_per_number):
    try:
        TMP_ARTIFACTS_PATH.mkdir()
    except FileExistsError:
        pass

    path_template = TMP_ARTIFACTS_PATH / f'{bits_per_number}_bit_{type}.eps'
    plot.savefig(path_template, format='eps', dpi=600)


PARSABLE_PATTERN = re.compile(r'''
                        (.*EVAL_PARSABLE:\s)
                        (?P<step>\d+),
                        (?P<error>\d+\.\d+(e[+-]\d+)?),
                        (?P<loss>\d+\.\d+(e[+-]\d+)?).*''', re.VERBOSE)


def get_history(log_path):
    steps = []
    errors = []
    losses = []
    with open(log_path) as f:
        for line in f:
            matched = re.match(PARSABLE_PATTERN, line)
            if not matched:
                continue
            steps.append(int(matched.group('step')))
            errors.append(float(matched.group('error')))
            losses.append(float(matched.group('loss')))
    return steps, errors, losses


def main(args):
    steps, errors, losses = get_history(args.log_path)
    plot = create_chart(series={'Error': errors}, ticks=steps, granularity=args.granularity, type='error',
                        bits_per_number=args.bits_per_number)
    save_chart(plot, type='error', bits_per_number=args.bits_per_number)
    plot = create_chart(series={'Loss': losses}, ticks=steps, granularity=args.granularity, type='loss',
                        bits_per_number=args.bits_per_number)
    save_chart(plot, type='loss', bits_per_number=args.bits_per_number)


def main_mta_all(args):
    mem_128_4_bit_steps, mem_128_4_bit_errors, mem_128_4_bit_losses = get_history(
        './trained_models/average_binary_sum_v1/out.log')
    mem_128_4_bit_steps = [*mem_128_4_bit_steps[9::10], mem_128_4_bit_steps[-1]]
    mem_128_4_bit_errors = [*mem_128_4_bit_errors[9::10], mem_128_4_bit_errors[-1]]
    mem_128_4_bit_losses = [*mem_128_4_bit_losses[9::10], mem_128_4_bit_losses[-1]]
    mem_256_6_bit_steps, mem_256_6_bit_errors, mem_256_6_bit_losses = get_history(
        './trained_models/average_binary_sum_v2/6_bits_256_memory_3_experts_contrib/out.log')
    mem_256_8_bit_steps, mem_256_8_bit_errors, mem_256_8_bit_losses = get_history(
        './trained_models/average_binary_sum_v2/8_bits_256_memory_3_experts_local/out.log')
    mem_512_8_bit_steps, mem_512_8_bit_errors, mem_512_8_bit_losses = get_history(
        './trained_models/average_binary_sum_v2/8_bits_512_memory_3_experts_contrib/out.log')
    mem_256_10_bit_steps, mem_256_10_bit_errors, mem_256_10_bit_losses = get_history(
        './trained_models/average_binary_sum_v2/10_bits_256_memory_3_experts_contrib/out.log')
    series_dict = {
        '4 bits, 128 memory': mem_128_4_bit_errors,
        '6 bits, 256 memory': mem_256_6_bit_errors,
        '8 bits, 256 memory': mem_256_8_bit_errors,
        '8 bits, 512 memory': mem_512_8_bit_errors,
        '10 bits, 256 memory': mem_256_10_bit_errors,
    }
    max_len = max([len(i) for i in series_dict.values()])
    for key in series_dict:
        series_dict[key].extend([np.nan for _ in range(max_len - len(series_dict[key]))])
    plot = create_chart(series=series_dict, ticks=mem_256_8_bit_steps, granularity=args.granularity, type='error',
                        bits_per_number=args.bits_per_number)
    save_chart(plot, type='error', bits_per_number=args.bits_per_number)

    series_dict = {
        '4 bits, 128 memory': mem_128_4_bit_losses,
        '6 bits, 256 memory': mem_256_6_bit_losses,
        '8 bits, 256 memory': mem_256_8_bit_losses,
        '8 bits, 512 memory': mem_512_8_bit_losses,
        '10 bits, 256 memory': mem_256_10_bit_losses,
    }
    max_len = max([len(i) for i in series_dict.values()])
    for key in series_dict:
        series_dict[key].extend([np.nan for _ in range(max_len - len(series_dict[key]))])
    plot = create_chart(series=series_dict, ticks=mem_256_8_bit_steps, granularity=args.granularity, type='loss',
                        bits_per_number=args.bits_per_number)
    save_chart(plot, type='loss', bits_per_number=args.bits_per_number)


def main_mta_2_experts_all(args):
    mem_128_4_bit_steps, mem_128_4_bit_errors, mem_128_4_bit_losses = get_history(
        './trained_models/average_binary_sum_v3/4_bits_128_memory_2_experts_contrib/out.log')
    mem_256_6_bit_steps, mem_256_6_bit_errors, mem_256_6_bit_losses = get_history(
        './trained_models/average_binary_sum_v3/6_bits_256_memory_2_experts_contrib/out.log')
    mem_256_8_bit_steps, mem_256_8_bit_errors, mem_256_8_bit_losses = get_history(
        './trained_models/average_binary_sum_v3/8_bits_256_memory_2_experts_contrib/out.log')
    mem_256_10_bit_steps, mem_256_10_bit_errors, mem_256_10_bit_losses = get_history(
        './trained_models/average_binary_sum_v3/10_bits_256_memory_2_experts_contrib/out.log')
    mem_256_16_bit_steps, mem_256_16_bit_errors, mem_256_16_bit_losses = get_history(
        './trained_models/average_binary_sum_v3/16_bits_256_memory_2_experts_contrib/out.log')
    mem_256_16_bit_steps = [*mem_256_16_bit_steps[:12]]
    mem_256_16_bit_errors = [*mem_256_16_bit_errors[:12]]
    mem_256_16_bit_losses = [*mem_256_16_bit_losses[:12]]

    series_dict = {
        '4 bits, 128 memory': mem_128_4_bit_errors,
        '6 bits, 256 memory': mem_256_6_bit_errors,
        '8 bits, 256 memory': mem_256_8_bit_errors,
        '10 bits, 256 memory': mem_256_10_bit_errors,
        '16 bits, 256 memory': mem_256_16_bit_errors,
    }
    max_len = max([len(i) for i in series_dict.values()])
    for key in series_dict:
        series_dict[key].extend([np.nan for _ in range(max_len - len(series_dict[key]))])
    plot = create_chart(series=series_dict, ticks=mem_256_16_bit_steps, granularity=args.granularity, type='error',
                        bits_per_number=args.bits_per_number)
    save_chart(plot, type='error', bits_per_number=args.bits_per_number)

    series_dict = {
        '4 bits, 128 memory': mem_128_4_bit_losses,
        '6 bits, 256 memory': mem_256_6_bit_losses,
        '8 bits, 256 memory': mem_256_8_bit_losses,
        '10 bits, 256 memory': mem_256_10_bit_losses,
        '16 bits, 256 memory': mem_256_16_bit_losses,
    }
    max_len = max([len(i) for i in series_dict.values()])
    for key in series_dict:
        series_dict[key].extend([np.nan for _ in range(max_len - len(series_dict[key]))])
    plot = create_chart(series=series_dict, ticks=mem_256_16_bit_steps, granularity=args.granularity, type='loss',
                        bits_per_number=args.bits_per_number)
    save_chart(plot, type='loss', bits_per_number=args.bits_per_number)


def main_mta_2_tuple_2_experts_v1_all(args):
    mem_256_104_bit_steps, mem_256_104_bit_errors, mem_256_104_bit_losses = get_history(
        './trained_models/mta_v1/104_bits_256_memory_2_experts_local_full_binary_layout/out.log')

    series_dict = {
        '104 bits, 256 memory': mem_256_104_bit_errors,
    }
    max_len = max([len(i) for i in series_dict.values()])
    for key in series_dict:
        series_dict[key].extend([np.nan for _ in range(max_len - len(series_dict[key]))])
    plot = create_chart(series=series_dict, ticks=mem_256_104_bit_steps, granularity=args.granularity, type='error',
                        bits_per_number=args.bits_per_number)
    save_chart(plot, type='error', bits_per_number=args.bits_per_number)

    series_dict = {
        '104 bits, 256 memory': mem_256_104_bit_losses,
    }
    max_len = max([len(i) for i in series_dict.values()])
    for key in series_dict:
        series_dict[key].extend([np.nan for _ in range(max_len - len(series_dict[key]))])
    plot = create_chart(series=series_dict, ticks=mem_256_104_bit_steps, granularity=args.granularity, type='loss',
                        bits_per_number=args.bits_per_number)
    save_chart(plot, type='loss', bits_per_number=args.bits_per_number)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', required=True, type=str,
                        help='Path to training log')
    parser.add_argument('--granularity', required=False, type=int, default=5000,
                        help='Granularity for ticks')
    parser.add_argument('--bits_per_number', required=True, type=int,
                        help='Bits per number')
    args = parser.parse_args()
    if args.log_path == 'mta_all':
        # main_mta_all(args)
        # main_mta_2_experts_all(args)
        main_mta_2_tuple_2_experts_v1_all(args)
    else:
        main(args)
