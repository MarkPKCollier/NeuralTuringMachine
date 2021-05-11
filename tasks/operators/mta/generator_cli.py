import pickle
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import tensorflow as tf

from tap import Tap

from tasks.operators.mta.generator import generate_batches, load_batches_from_file


class GeneratorCLIArgumentParser(Tap):
    num_batches: int = 1
    batch_size: int = 32
    bits_per_vector: int = 3
    numbers_quantity: int = 2
    two_tuple_weight_precision: int = 1
    two_tuple_alpha_precision: int = 1
    two_tuple_largest_scale_size: int = 5
    serialized_path: str
    mode: str


def main(args: GeneratorCLIArgumentParser):
    if args.mode == 'generate':
        batches = generate_batches(num_batches=args.num_batches,
                                   batch_size=args.batch_size,
                                   bits_per_vector=args.bits_per_vector,
                                   numbers_quantity=args.numbers_quantity,
                                   two_tuple_weight_precision=args.two_tuple_weight_precision,
                                   two_tuple_alpha_precision=args.two_tuple_alpha_precision,
                                   two_tuple_largest_scale_size=args.two_tuple_largest_scale_size)

        path = Path(args.serialized_path)
        try:
            path.parent.mkdir()
        except FileExistsError:
            pass

        with open(path, 'wb') as output:
            pickle.dump(batches, output, pickle.HIGHEST_PROTOCOL)
    elif args.mode == 'load':
        print('Loading serialized batches object')
        batches = load_batches_from_file(Path(args.serialized_path))
        print(f'Loaded {len(batches)} batches, each contains elements of size {batches[0][1].shape} with '
              f'encoded vectors each of length {batches[0][0]}')
    else:
        print(f'Current mode {args.mode} is not supported')


if __name__ == '__main__':
    tf.compat.v1.disable_v2_behavior()
    args = GeneratorCLIArgumentParser().parse_args()
    print(args)
    main(args)
