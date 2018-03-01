import tensorflow as tf
import numpy as np
from generate_data import CopyTaskData, AssociativeRecallData
from utils import expand, learned_init
from exp3S import Exp3S

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import argparse

parser = argparse.ArgumentParser()

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser.add_argument('--mann', type=str, default='none', help='none | ntm')
parser.add_argument('--num_layers', type=int, default=1)
parser.add_argument('--num_units', type=int, default=100)
parser.add_argument('--num_memory_locations', type=int, default=128)
parser.add_argument('--memory_size', type=int, default=20)
parser.add_argument('--num_read_heads', type=int, default=1)
parser.add_argument('--num_write_heads', type=int, default=1)
parser.add_argument('--conv_shift_range', type=int, default=1, help='only necessary for ntm')
parser.add_argument('--clip_value', type=int, default=20, help='Maximum absolute value of controller and outputs.')
parser.add_argument('--init_mode', type=str, default='learned', help='learned | constant | random')

parser.add_argument('--optimizer', type=str, default='RMSProp', help='RMSProp | Adam')
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--max_grad_norm', type=float, default=50)
parser.add_argument('--num_train_steps', type=int, default=31250)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--eval_batch_size', type=int, default=640)

parser.add_argument('--curriculum', type=str, default='none', help='none | uniform | naive | look_back | look_back_and_forward | prediction_gain')
parser.add_argument('--pad_to_max_seq_len', type=str2bool, default=False)

parser.add_argument('--task', type=str, default='copy', help='copy | associative_recall')
parser.add_argument('--num_bits_per_vector', type=int, default=8)
parser.add_argument('--max_seq_len', type=int, default=20)

parser.add_argument('--verbose', type=str2bool, default=True, help='if true prints lots of feedback')
parser.add_argument('--experiment_name', type=str, required=True)
parser.add_argument('--job-dir', type=str, required=False)
parser.add_argument('--steps_per_eval', type=int, default=200)

args = parser.parse_args()

if args.mann == 'ntm':
    from ntm import NTMCell

if args.verbose:
    import pickle
    HEAD_LOG_FILE = 'head_logs/{0}.p'.format(args.experiment_name)
    GENERALIZATION_HEAD_LOG_FILE = 'head_logs/generalization_{0}.p'.format(args.experiment_name)

class BuildModel(object):
    def __init__(self, max_seq_len, inputs, mode):
        self.max_seq_len = max_seq_len
        self.inputs = inputs
        self.mode = mode
        self._build_model()

    def _build_model(self):
        if args.mann == 'none':
            def single_cell(num_units):
                return tf.contrib.rnn.BasicLSTMCell(num_units, forget_bias=1.0)

            cell = tf.contrib.rnn.OutputProjectionWrapper(
                tf.contrib.rnn.MultiRNNCell([single_cell(args.num_units) for _ in range(args.num_layers)]),
                args.num_bits_per_vector,
                activation=None)

            initial_state = tuple(tf.contrib.rnn.LSTMStateTuple(
                c=expand(tf.tanh(learned_init(args.num_units)), dim=0, N=args.batch_size),
                h=expand(tf.tanh(learned_init(args.num_units)), dim=0, N=args.batch_size))
                for _ in range(args.num_layers))

        elif args.mann == 'ntm':
            cell = NTMCell(args.num_layers, args.num_units, args.num_memory_locations, args.memory_size,
                args.num_read_heads, args.num_write_heads, addressing_mode='content_and_location',
                shift_range=args.conv_shift_range, reuse=False, output_dim=args.num_bits_per_vector,
                clip_value=args.clip_value, init_mode=args.init_mode)

            initial_state = cell.zero_state(args.batch_size, tf.float32)
        
        output_sequence, _ = tf.nn.dynamic_rnn(
            cell=cell,
            inputs=self.inputs,
            time_major=False,
            initial_state=initial_state)

        if args.task == 'copy':
            self.output_logits = output_sequence[:, self.max_seq_len+1:, :]
        elif args.task == 'associative_recall':
            self.output_logits = output_sequence[:, 3*(self.max_seq_len+1)+2:, :]

        if args.task in ('copy', 'associative_recall'):
            self.outputs = tf.sigmoid(self.output_logits)

class BuildTrainModel(BuildModel):
    def __init__(self, max_seq_len, inputs, outputs):
        super(BuildTrainModel, self).__init__(max_seq_len, inputs, tf.contrib.learn.ModeKeys.TRAIN)

        if args.task in ('copy', 'associative_recall'):
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=outputs, logits=self.output_logits)
            self.loss = tf.reduce_sum(cross_entropy)/args.batch_size

        if args.optimizer == 'RMSProp':
            optimizer = tf.train.RMSPropOptimizer(args.learning_rate, momentum=0.9, decay=0.9)
        elif args.optimizer == 'Adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)

        trainable_variables = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, trainable_variables), args.max_grad_norm)
        self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables))

class BuildEvalModel(BuildModel):
    def __init__(self, max_seq_len, inputs, outputs):
        super(BuildEvalModel, self).__init__(max_seq_len, inputs, tf.contrib.learn.ModeKeys.EVAL)

        if args.task in ('copy', 'associative_recall'):
            self.loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=outputs, logits=self.output_logits))/args.batch_size

with tf.variable_scope('root'):
    train_max_seq_len = tf.placeholder(tf.int32)
    train_inputs = tf.placeholder(tf.float32, shape=(args.batch_size, None, args.num_bits_per_vector+1))
    train_outputs = tf.placeholder(tf.float32, shape=(args.batch_size, None, args.num_bits_per_vector))
    train_model = BuildTrainModel(train_max_seq_len, train_inputs, train_outputs)
    initializer = tf.global_variables_initializer()

with tf.variable_scope('root', reuse=True):
    eval_max_seq_len = tf.placeholder(tf.int32)
    eval_inputs = tf.placeholder(tf.float32, shape=(args.batch_size, None, args.num_bits_per_vector+1))
    eval_outputs = tf.placeholder(tf.float32, shape=(args.batch_size, None, args.num_bits_per_vector))
    eval_model = BuildEvalModel(eval_max_seq_len, eval_inputs, eval_outputs)

# training

convergence_on_target_task = None
convergence_on_multi_task = None
performance_on_target_task = None
performance_on_multi_task = None
generalization_from_target_task = None
generalization_from_multi_task = None
if args.task == 'copy':
    data_generator = CopyTaskData()
    target_point = args.max_seq_len
    curriculum_point = 1 if args.curriculum not in ('prediction_gain', 'none') else target_point
    progress_error = 1.0
    convergence_error = 0.1

    if args.curriculum == 'prediction_gain':
        exp3s = Exp3S(args.max_seq_len, 0.001, 0, 0.05)
elif args.task == 'associative_recall':
    data_generator = AssociativeRecallData()
    target_point = args.max_seq_len
    curriculum_point = 2 if args.curriculum not in ('prediction_gain', 'none') else target_point
    progress_error = 1.0
    convergence_error = 0.1

    if args.curriculum == 'prediction_gain':
        exp3s = Exp3S(args.max_seq_len-1, 0.001, 0, 0.05)

sess = tf.Session()
sess.run(initializer)

if args.verbose:
    pickle.dump({target_point: []}, open(HEAD_LOG_FILE, "wb"))
    pickle.dump({}, open(GENERALIZATION_HEAD_LOG_FILE, "wb"))

def run_eval(batches, store_heat_maps=False, generalization_num=None):
    task_loss = 0
    task_error = 0
    num_batches = len(batches)
    for seq_len, inputs, labels in batches:
        task_loss_, outputs = sess.run([eval_model.loss, eval_model.outputs],
            feed_dict={
                eval_inputs: inputs,
                eval_outputs: labels,
                eval_max_seq_len: seq_len
            })

        task_loss += task_loss_
        task_error += data_generator.error_per_seq(labels, outputs, args.batch_size)

    if store_heat_maps:
        if generalization_num is None:
            tmp = pickle.load(open(HEAD_LOG_FILE, "rb"))
            tmp[target_point].append({
                'labels': labels[0],
                'outputs': outputs[0],
                'inputs': inputs[0]
            })
            pickle.dump(tmp, open(HEAD_LOG_FILE, "wb"))
        else:
            tmp = pickle.load(open(GENERALIZATION_HEAD_LOG_FILE, "rb"))
            if tmp.get(generalization_num) is None:
                tmp[generalization_num] = []
            tmp[generalization_num].append({
                'labels': labels[0],
                'outputs': outputs[0],
                'inputs': inputs[0]
            })
            pickle.dump(tmp, open(GENERALIZATION_HEAD_LOG_FILE, "wb"))


    task_loss /= float(num_batches)
    task_error /= float(num_batches)
    return task_loss, task_error

def eval_performance(curriculum_point, store_heat_maps=False):
    # target task
    batches = data_generator.generate_batches(
        (args.eval_batch_size/2)/args.batch_size,
        args.batch_size,
        bits_per_vector=args.num_bits_per_vector,
        curriculum_point=None,
        max_seq_len=args.max_seq_len,
        curriculum='none',
        pad_to_max_seq_len=args.pad_to_max_seq_len
    )

    target_task_loss, target_task_error = run_eval(batches, store_heat_maps=store_heat_maps)

    # multi-task

    batches = data_generator.generate_batches(
        args.eval_batch_size/args.batch_size,
        args.batch_size,
        bits_per_vector=args.num_bits_per_vector,
        curriculum_point=None,
        max_seq_len=args.max_seq_len,
        curriculum='deterministic_uniform',
        pad_to_max_seq_len=args.pad_to_max_seq_len
    )

    multi_task_loss, multi_task_error = run_eval(batches)

    # curriculum point
    if curriculum_point is not None:
        batches = data_generator.generate_batches(
            (args.eval_batch_size/4)/args.batch_size,
            args.batch_size,
            bits_per_vector=args.num_bits_per_vector,
            curriculum_point=curriculum_point,
            max_seq_len=args.max_seq_len,
            curriculum='naive',
            pad_to_max_seq_len=args.pad_to_max_seq_len
        )

        curriculum_point_loss, curriculum_point_error = run_eval(batches)
    else:
        curriculum_point_error = curriculum_point_loss = None

    return target_task_error, target_task_loss, multi_task_error, multi_task_loss, curriculum_point_error, curriculum_point_loss

def eval_generalization():
    res = []
    if args.task == 'copy':
        seq_lens = [40, 60, 80, 100, 120]
    elif args.task == 'associative_recall':
        seq_lens = [7, 8, 9, 10, 11, 12]

    for i in seq_lens:
        batches = data_generator.generate_batches(
            6,
            args.batch_size,
            bits_per_vector=args.num_bits_per_vector,
            curriculum_point=i,
            max_seq_len=args.max_seq_len,
            curriculum='naive',
            pad_to_max_seq_len=False
        )

        loss, error = run_eval(batches, store_heat_maps=args.verbose, generalization_num=i)
        res.append(error)
    return res

for i in range(args.num_train_steps):
    if args.curriculum == 'prediction_gain':
        if args.task == 'copy':
            task = 1 + exp3s.draw_task()
        elif args.task == 'associative_recall':
            task = 2 + exp3s.draw_task()

    seq_len, inputs, labels = data_generator.generate_batches(
        1,
        args.batch_size,
        bits_per_vector=args.num_bits_per_vector,
        curriculum_point=curriculum_point if args.curriculum != 'prediction_gain' else task,
        max_seq_len=args.max_seq_len,
        curriculum=args.curriculum,
        pad_to_max_seq_len=args.pad_to_max_seq_len
    )[0]

    train_loss, _, outputs = sess.run([train_model.loss, train_model.train_op, train_model.outputs],
        feed_dict={
            train_inputs: inputs,
            train_outputs: labels,
            train_max_seq_len: seq_len
        })

    if args.curriculum == 'prediction_gain':
        loss, _ = run_eval([(seq_len, inputs, labels)])
        v = train_loss - loss
        exp3s.update_w(v, seq_len)

    avg_errors_per_seq = data_generator.error_per_seq(labels, outputs, args.batch_size)

    if args.verbose:
        logger.info('Train loss ({0}): {1}'.format(i, train_loss))
        logger.info('curriculum_point: {0}'.format(curriculum_point))
        logger.info('Average errors/sequence: {0}'.format(avg_errors_per_seq))
        logger.info('TRAIN_PARSABLE: {0},{1},{2},{3}'.format(i, curriculum_point, train_loss, avg_errors_per_seq))

    if i % args.steps_per_eval == 0:
        target_task_error, target_task_loss, multi_task_error, multi_task_loss, curriculum_point_error, \
        curriculum_point_loss = eval_performance(curriculum_point if args.curriculum != 'prediction_gain' else None, store_heat_maps=args.verbose)

        if convergence_on_multi_task is None and multi_task_error < convergence_error:
            convergence_on_multi_task = i

        if convergence_on_target_task is None and target_task_error < convergence_error:
            convergence_on_target_task = i

        gen_evaled = False
        if convergence_on_multi_task is not None and (performance_on_multi_task is None or multi_task_error < performance_on_multi_task):
            performance_on_multi_task = multi_task_error
            generalization_from_multi_task = eval_generalization()
            gen_evaled = True

        if convergence_on_target_task is not None and (performance_on_target_task is None or target_task_error < performance_on_target_task):
            performance_on_target_task = target_task_error
            if gen_evaled:
                generalization_from_target_task = generalization_from_multi_task
            else:
                generalization_from_target_task = eval_generalization()

        if curriculum_point_error < progress_error:
            if args.task == 'copy':
                curriculum_point = min(target_point, 2 * curriculum_point)
            elif args.task == 'associative_recall':
                curriculum_point = min(target_point, curriculum_point+1)

        logger.info('----EVAL----')
        logger.info('target task error/loss: {0},{1}'.format(target_task_error, target_task_loss))
        logger.info('multi task error/loss: {0},{1}'.format(multi_task_error, multi_task_loss))
        logger.info('curriculum point error/loss ({0}): {1},{2}'.format(curriculum_point, curriculum_point_error, curriculum_point_loss))
        logger.info('EVAL_PARSABLE: {0},{1},{2},{3},{4},{5},{6},{7}'.format(i, target_task_error, target_task_loss,
            multi_task_error, multi_task_loss, curriculum_point, curriculum_point_error, curriculum_point_loss))

if convergence_on_multi_task is None:
    performance_on_multi_task = multi_task_error
    generalization_from_multi_task = eval_generalization()

if convergence_on_target_task is None:
    performance_on_target_task = target_task_error
    generalization_from_target_task = eval_generalization()

logger.info('----SUMMARY----')
logger.info('convergence_on_target_task: {0}'.format(convergence_on_target_task))
logger.info('performance_on_target_task: {0}'.format(performance_on_target_task))
logger.info('convergence_on_multi_task: {0}'.format(convergence_on_multi_task))
logger.info('performance_on_multi_task: {0}'.format(performance_on_multi_task))

logger.info('SUMMARY_PARSABLE: {0},{1},{2},{3}'.format(convergence_on_target_task, performance_on_target_task,
            convergence_on_multi_task, performance_on_multi_task))

logger.info('generalization_from_target_task: {0}'.format(','.join(map(str, generalization_from_target_task)) if generalization_from_target_task is not None else None))
logger.info('generalization_from_multi_task: {0}'.format(','.join(map(str, generalization_from_multi_task)) if generalization_from_multi_task is not None else None))

