import numpy as np
from scipy import spatial
import random

snap_boolean = np.vectorize(lambda x: 1.0 if x > 0.5 else 0.0)

class CopyTaskData:
    def generate_batches(self, num_batches, batch_size, bits_per_vector=8, curriculum_point=20, max_seq_len=20,
        curriculum='uniform', pad_to_max_seq_len=False):
        batches = []
        for i in range(num_batches):
            if curriculum == 'deterministic_uniform':
                seq_len = 1 + (i % max_seq_len)
            elif curriculum == 'uniform':
                seq_len = np.random.randint(low=1, high=max_seq_len+1)
            elif curriculum == 'none':
                seq_len = max_seq_len
            elif curriculum in ('naive', 'prediction_gain'):
                seq_len = curriculum_point
            elif curriculum == 'look_back':
                seq_len = curriculum_point if np.random.random_sample() < 0.9 else np.random.randint(low=1, high=curriculum_point+1)
            elif curriculum == 'look_back_and_forward':
                seq_len = curriculum_point if np.random.random_sample() < 0.8 else np.random.randint(low=1, high=max_seq_len+1)
            
            pad_to_len = max_seq_len if pad_to_max_seq_len else seq_len

            def generate_sequence():
                return np.asarray([snap_boolean(np.append(np.random.rand(bits_per_vector), 0)) for _ in range(seq_len)] \
                    + [np.zeros(bits_per_vector+1) for _ in range(pad_to_len - seq_len)])

            inputs = np.asarray([generate_sequence() for _ in range(batch_size)]).astype(np.float32)
            eos = np.ones([batch_size, 1, bits_per_vector + 1])
            output_inputs = np.zeros_like(inputs)

            full_inputs = np.concatenate((inputs, eos, output_inputs), axis=1)

            batches.append((pad_to_len, full_inputs, inputs[:, :, :bits_per_vector]))
        return batches

    def error_per_seq(self, labels, outputs, num_seq):
        outputs[outputs >= 0.5] = 1.0
        outputs[outputs < 0.5] = 0.0
        bit_errors = np.sum(np.abs(labels - outputs))
        return bit_errors/num_seq

class AssociativeRecallData:
    def generate_batches(self, num_batches, batch_size, bits_per_vector=6, curriculum_point=6, max_seq_len=6,
        curriculum='uniform', pad_to_max_seq_len=False):
        NUM_VECTORS_PER_ITEM = 3
        batches = []
        for i in range(num_batches):
            if curriculum == 'deterministic_uniform':
                seq_len = 2 + (i % max_seq_len)
            elif curriculum == 'uniform':
                seq_len = np.random.randint(low=2, high=max_seq_len+1)
            elif curriculum == 'none':
                seq_len = max_seq_len
            elif curriculum in ('naive', 'prediction_gain'):
                seq_len = curriculum_point
            elif curriculum == 'look_back':
                seq_len = curriculum_point if np.random.random_sample() < 0.9 else np.random.randint(low=2, high=curriculum_point+1)
            elif curriculum == 'look_back_and_forward':
                seq_len = curriculum_point if np.random.random_sample() < 0.8 else np.random.randint(low=2, high=max_seq_len+1)
            
            pad_to_len = max_seq_len if pad_to_max_seq_len else seq_len

            def generate_item(seq_len):
                items = [[snap_boolean(np.append(np.random.rand(bits_per_vector), 0)) for _ in range(NUM_VECTORS_PER_ITEM)] for _ in range(seq_len)]

                query_item_num = seq_len = np.random.randint(low=0, high=seq_len-1)
                query_item = items[query_item_num]
                output_item = items[query_item_num+1]

                inputs = [sub_item for item in items for sub_item in item]

                return inputs, query_item, map(lambda sub_item: sub_item[:bits_per_vector], output_item)

            batch_inputs = []
            batch_queries = []
            batch_outputs = []
            for _ in range(batch_size):
                inputs, query_item, output_item = generate_item(seq_len)
                batch_inputs.append(inputs)
                batch_queries.append(query_item)
                batch_outputs.append(output_item)

            batch_inputs = np.asarray(batch_inputs).astype(np.float32)
            batch_queries = np.asarray(batch_queries).astype(np.float32)
            batch_outputs = np.asarray(batch_outputs).astype(np.float32)
            eos = np.ones([batch_size, 1, bits_per_vector + 1])
            output_inputs = np.zeros([batch_size, NUM_VECTORS_PER_ITEM, bits_per_vector + 1])

            if pad_to_max_seq_len:
                full_inputs = np.concatenate((batch_inputs, eos, batch_queries, eos, np.zeros([batch_size, pad_to_len - seq_len, bits_per_vector + 1]) ,output_inputs), axis=1)
            else:
                full_inputs = np.concatenate((batch_inputs, eos, batch_queries, eos, output_inputs), axis=1)

            batches.append((pad_to_len, full_inputs, batch_outputs))
        return batches

    def error_per_seq(self, labels, outputs, num_seq):
        outputs[outputs >= 0.5] = 1.0
        outputs[outputs < 0.5] = 0.0
        bit_errors = np.sum(np.abs(labels - outputs))
        return bit_errors/num_seq

