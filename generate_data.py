import numpy as np
from scipy import spatial
import random

def set_random_seed(seed):
    np.random.seed(seed)

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

class RepeatCopyTaskData:
    def __init__(self, max_seq_len, max_repeats):
        self.max_seq_len = max_seq_len
        self.max_repeats = max_repeats

    def _normalize_num_repeats(self, n):
        return float(n)/self.max_repeats

    def generate_batches(self, num_batches, batch_size, bits_per_vector=8, curriculum_point=20, max_seq_len=20,
        curriculum='uniform', pad_to_max_seq_len=False):
        batches = []
        for i in range(num_batches):
            if curriculum == 'deterministic_uniform':
                seq_len = 1 + (i % self.max_seq_len)
                num_repeats = 1 + (i % self.max_repeats)
            elif curriculum == 'uniform':
                seq_len = np.random.randint(low=1, high=self.max_seq_len+1)
                num_repeats = np.random.randint(low=1, high=self.max_repeats+1)
            elif curriculum == 'none':
                seq_len = self.max_seq_len
                num_repeats = self.max_repeats
            elif curriculum in ('naive', 'prediction_gain_bandit', 'prediction_gain_teacher'):
                seq_len, num_repeats = curriculum_point
            elif curriculum == 'look_back':
                seq_len = curriculum_point[0] if np.random.random_sample() < 0.9 else np.random.randint(low=1, high=curriculum_point[0]+1)
                num_repeats = curriculum_point[1] if np.random.random_sample() < 0.9 else np.random.randint(low=1, high=curriculum_point[1]+1)
            elif curriculum == 'look_back_and_forward':
                seq_len = curriculum_point[0] if np.random.random_sample() < 0.8 else np.random.randint(low=1, high=self.max_seq_len+1)
                num_repeats = curriculum_point[1] if np.random.random_sample() < 0.8 else np.random.randint(low=1, high=self.max_repeats+1)
            
            self.seq_len = seq_len
            self.num_repeats = num_repeats

            pad_to_len = self.max_seq_len if pad_to_max_seq_len else seq_len

            def generate_sequence():
                return np.asarray([snap_boolean(np.append(np.random.rand(bits_per_vector), (0,0))) for _ in range(seq_len)] \
                    + [np.zeros(bits_per_vector+2) for _ in range(pad_to_len - seq_len)])

            inputs = np.asarray([generate_sequence() for _ in range(batch_size)]).astype(np.float32)

            eos = np.ones([batch_size, 1, bits_per_vector + 2])
            eos[:, :, bits_per_vector+1] = self._normalize_num_repeats(num_repeats)
            output_inputs = np.tile(np.zeros_like(inputs), (1, num_repeats, 1))

            full_inputs = np.concatenate((inputs, eos, output_inputs), axis=1)

            targets = np.tile(inputs[:, :, :bits_per_vector], (1, num_repeats, 1))

            batches.append((pad_to_len, full_inputs, targets))
        return batches

    def error_per_seq(self, labels, outputs, num_seq):
        outputs[outputs >= 0.5] = 1.0
        outputs[outputs < 0.5] = 0.0
        bit_errors = np.sum(np.abs(labels - outputs))
        return bit_errors/num_seq

    def last_seq_len(self):
        return self.seq_len

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
            elif curriculum in ('naive', 'prediction_gain_bandit', 'prediction_gain_teacher'):
                seq_len = curriculum_point
            elif curriculum == 'look_back':
                seq_len = curriculum_point if np.random.random_sample() < 0.9 else np.random.randint(low=2, high=curriculum_point+1)
            elif curriculum == 'look_back_and_forward':
                seq_len = curriculum_point if np.random.random_sample() < 0.8 else np.random.randint(low=2, high=max_seq_len+1)
            
            self.seq_len = seq_len

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
            eos[:, :, bits_per_vector] = 1
            output_inputs = np.zeros([batch_size, NUM_VECTORS_PER_ITEM, bits_per_vector + 1])

            if pad_to_max_seq_len:
                full_inputs = np.concatenate(batch_inputs, eos, batch_queries, eos, np.zeros([batch_size, pad_to_len - seq_len, bits_per_vector + 1], output_inputs), axis=1)
            else:
                full_inputs = np.concatenate((batch_inputs, eos, batch_queries, eos, output_inputs), axis=1)

            batches.append((pad_to_len, full_inputs, batch_outputs))
        return batches

    def error_per_seq(self, labels, outputs, num_seq):
        outputs[outputs >= 0.5] = 1.0
        outputs[outputs < 0.5] = 0.0
        bit_errors = np.sum(np.abs(labels - outputs))
        return bit_errors/num_seq

    def last_seq_len(self):
        return self.seq_len

def graph_label_to_one_hot(label):
    res = np.zeros(30)

    if label == -1:
        return res

    hundreds = label/100
    tens = (label % 100)/10
    singles = (label % 100) % 10

    res[hundreds] = 1
    res[tens + 10] = 1
    res[singles + 20] = 1
    return res

def label_from_vectors(vectors):
    return map(lambda batch: np.argmax(batch, axis=-1), np.split(vectors, 9, axis=-1))

def generate_random_graph(num_nodes, k):
    node_labels = 1 + np.random.choice(999, num_nodes, replace=False)
    edge_label_candidates = 1 + np.random.choice(999, num_nodes, replace=False)
    tree = spatial.KDTree(np.random.uniform(size=(num_nodes, 2)))

    graph = {}
    graph_des_vectors = []
    for node_idx in range(num_nodes):
        node_k = k[node_idx]
        _, indexes = tree.query(tree.data[node_idx], k=node_k+1)
        indexes = indexes[1:]
        connected_nodes = map(lambda idx: node_labels[idx], indexes)
        edge_labels = np.random.choice(edge_label_candidates, node_k, replace=False)

        graph[node_labels[node_idx]] = zip(connected_nodes, edge_labels)

        for connected_node, edge_label in zip(connected_nodes, edge_labels):
            graph_des_vectors.append(
                np.concatenate(
                    (graph_label_to_one_hot(node_labels[node_idx]),
                    graph_label_to_one_hot(edge_label),
                    graph_label_to_one_hot(connected_node),
                    np.zeros(2))
                ))

    return graph, graph_des_vectors

def generate_random_graphs(num_graphs, node_range=(3,10), out_degree=(2,4)):
    num_nodes = np.random.randint(low=node_range[0], high=node_range[1]+1)
    k = np.random.randint(low=out_degree[0], high=min(num_nodes-1, out_degree[1])+1, size=num_nodes)

    return [generate_random_graph(num_nodes, k) for _ in range(num_graphs)]

class TraversalData:
    def __init__(self):
        self.lessons = [
            ((3,10), (2,4), (1,1)),
            ((3,10), (2,4), (1,2)),
            ((5,10), (2,4), (1,3)),
            ((5,10), (2,4), (1,4)),
            ((10,15), (2,4), (1,4)),
            ((10,15), (2,4), (1,5)),
            ((10,20), (2,4), (1,5)),
            ((10,20), (2,4), (1,6)),
            ((10,30), (2,4), (1,6)),
            ((10,30), (2,4), (1,7)),
            ((10,30), (2,4), (1,8)),
            ((10,30), (2,4), (1,9)),
            ((10,40), (2,6), (1,10)),
            ((10,40), (2,6), (1,20))
        ]
        self.num_lessons = len(self.lessons)
        self.termination_pattern = np.concatenate((
            graph_label_to_one_hot(0),
            graph_label_to_one_hot(0),
            graph_label_to_one_hot(0)
        ))

    def generate_item(self, graph, graph_des_vectors, path_length):
        random.shuffle(graph_des_vectors)

        path = []
        cur_node = random.choice(graph.keys())
        for _ in range(path_length):
            next_node, edge_label = random.choice(graph[cur_node])
            path.append((cur_node, next_node, edge_label))
            cur_node = next_node

        outputs = map(lambda t:
            np.concatenate((
                graph_label_to_one_hot(t[0]),
                graph_label_to_one_hot(t[1]),
                graph_label_to_one_hot(t[2])
            )),
            path)

        first_query = np.concatenate((
            graph_label_to_one_hot(path[0][0]),
            graph_label_to_one_hot(-1),
            graph_label_to_one_hot(path[0][2]),
            [1, 0]
        ))
        
        other_queries = map(lambda t:
            np.concatenate((
                graph_label_to_one_hot(-1),
                graph_label_to_one_hot(-1),
                graph_label_to_one_hot(t[2]),
                [1, 0]
            )),
            path[1:])

        query = [first_query] + other_queries

        output_inputs = []
        for _ in range(len(outputs)):
            res = np.zeros(92)
            res[-1] = 1
            output_inputs.append(res)

        inputs = graph_des_vectors + query + output_inputs

        return inputs, outputs

    def generate_items(self, num_items, node_range, out_degree, path_length):
        batch_inputs = []
        batch_outputs = []

        path_length = np.random.randint(low=path_length[0], high=path_length[1]+1)
        for graph, graph_des_vectors in generate_random_graphs(num_items, node_range, out_degree):
            inputs, outputs = self.generate_item(graph, graph_des_vectors, path_length)
            batch_inputs.append(inputs)
            batch_outputs.append(outputs)

        return batch_inputs, batch_outputs

    def generate_batches(self, num_batches, batch_size, curriculum_point=1, curriculum='uniform',
        bits_per_vector=None, max_seq_len=None, pad_to_max_seq_len=None):
        batches = []
        for i in range(num_batches):
            if curriculum == 'deterministic_uniform':
                lesson = ((i + 1) % self.num_lessons)
            elif curriculum == 'uniform':
                lesson = np.random.randint(low=1, high=self.num_lessons+1)
            elif curriculum == 'none':
                lesson = self.num_lessons
            elif curriculum in ('naive', 'prediction_gain_bandit', 'prediction_gain_teacher'):
                lesson = curriculum_point
            elif curriculum == 'look_back':
                lesson = curriculum_point if np.random.random_sample() < 0.9 else np.random.randint(low=1, high=curriculum_point+1)
            elif curriculum == 'look_back_and_forward':
                lesson = curriculum_point if np.random.random_sample() < 0.8 else np.random.randint(low=1, high=self.num_lessons+1)
            
            batch_inputs, batch_outputs = self.generate_items(batch_size, *self.lessons[lesson-1])

            batch_outputs = map(lambda outputs: outputs + [self.termination_pattern], batch_outputs) # eos
            max_output_len = max(map(len, batch_outputs))

            batch_inputs = np.asarray(batch_inputs).astype(np.float32)
            batch_outputs = np.asarray(batch_outputs).astype(np.float32)

            batches.append((max_output_len, batch_inputs, batch_outputs))

        return batches

    def error_per_seq(self, labels, outputs, num_seq):
        seq_len = labels.shape[1]

        target_digits = label_from_vectors(labels)
        pred_digits = label_from_vectors(outputs)

        def create_labels(digits):
            labels = []
            for i in range(num_seq):
                tmp = []
                for j in range(seq_len):
                    tmp.append((
                        digits[0][i][j],
                        digits[1][i][j],
                        digits[2][i][j]
                    ))
                labels.append(tmp)
            return labels

        target_source_digits, pred_source_digits = create_labels(target_digits[:3]), create_labels(pred_digits[:3])
        target_dest_digits, pred_dest_digits = create_labels(target_digits[3:6]), create_labels(pred_digits[3:6])
        target_edge_digits, pred_edge_digits = create_labels(target_digits[6:9]), create_labels(pred_digits[6:9])

        errors = 0
        for i in range(num_seq):
            source_digit_same = all(map(lambda t: t[0] == t[1], zip(target_source_digits[i], pred_source_digits[i])))
            if not source_digit_same:
                errors += 1
                continue
            target_digit_same = all(map(lambda t: t[0] == t[1], zip(target_dest_digits[i], pred_dest_digits[i])))
            if not target_digit_same:
                errors += 1
                continue
            target_edge_same = all(map(lambda t: t[0] == t[1], zip(target_edge_digits[i], pred_edge_digits[i])))
            if not target_edge_same:
                errors += 1
                continue

        return errors/float(num_seq)

