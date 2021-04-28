import os
import argparse
import pickle
import sys

import tensorflow as tf

from freeze import analyze_inputs_outputs
from generate_data import CopyTaskData, AssociativeRecallData
from tasks.arithmetics.binary_average_sum.generator import AverageSumTaskData
from tasks.arithmetics.binary_average_sum.task import AverageSumTask
from tasks.arithmetics.binary_sum.generator import SumTaskData
from tasks.arithmetics.binary_sum.task import SumTask
from tasks.associative_recall.task import AssociativeRecallTask
from tasks.common.errors import UnknownTaskError
from tasks.copy.task import CopyTask
from tasks.operators.mta.task import MTATask
from utils import expand, learned_init, save_session_as_tf_checkpoint, str2bool, logger
from exp3S import Exp3S
from evaluate import run_eval, eval_performance, eval_generalization
import constants


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mann', type=str, default='ntm', help='none | ntm')
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--num_units', type=int, default=100)
    parser.add_argument('--num_memory_locations', type=int, default=128)
    parser.add_argument('--memory_size', type=int, default=20)
    parser.add_argument('--num_read_heads', type=int, default=1)
    parser.add_argument('--num_write_heads', type=int, default=1)
    parser.add_argument('--conv_shift_range', type=int, default=1, help='only necessary for ntm')
    parser.add_argument('--clip_value', type=int, default=20, help='Maximum absolute value of controller and outputs.')
    parser.add_argument('--init_mode', type=str, default='learned', help='learned | constant | random')

    parser.add_argument('--optimizer', type=str, default='Adam', help='RMSProp | Adam')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--max_grad_norm', type=float, default=50)
    parser.add_argument('--num_train_steps', type=int, default=31250)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=640)

    parser.add_argument('--curriculum', type=str, default='none',
                        help='none | uniform | naive | look_back | look_back_and_forward | prediction_gain')
    parser.add_argument('--pad_to_max_seq_len', type=str2bool, default=False)

    parser.add_argument('--task', type=str, default='copy', help='copy | associative_recall',
                        choices=(CopyTask.name, AssociativeRecallTask.name, SumTask.name, AverageSumTask.name,
                                 MTATask.name))
    parser.add_argument('--num_bits_per_vector', type=int, default=8)
    parser.add_argument('--max_seq_len', type=int, default=20)

    parser.add_argument('--verbose', type=str2bool, default=True, help='if true prints lots of feedback')
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--job-dir', type=str, required=False)
    parser.add_argument('--steps_per_eval', type=int, default=200)
    parser.add_argument('--use_local_impl', type=str2bool, default=True,
                        help='whether to use the repos local NTM implementation or the TF contrib version')

    parser.add_argument('--continue_training_from_checkpoint', type=str, required=False,
                        help='Optional. Specifies path to the directory with checkpoint')
    parser.add_argument('--continue_training_from_train_step', type=int, default=0,
                        help='Optional. Specifies train step from which we need to continue training')

    parser.add_argument('--num_experts', type=int, required=False,
                        help='Optional. Specifies number of assessments (numbers) to aggregate: finding average')

    parser.add_argument('--device', type=str, required=False, choices=('cpu', 'gpu'), default='cpu',
                        help='Optional. Specifies number of assessments (numbers) to aggregate: finding average')

    return parser


class BuildModel(object):
    def __init__(self, max_seq_len, inputs):
        self.max_seq_len = max_seq_len
        self.inputs = inputs
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
            if args.use_local_impl:
                cell = NTMCell(
                    controller_layers=args.num_layers,
                    controller_units=args.num_units,
                    memory_size=args.num_memory_locations,
                    memory_vector_dim=args.memory_size,
                    read_head_num=args.num_read_heads,
                    write_head_num=args.num_write_heads,
                    addressing_mode='content_and_location',
                    shift_range=args.conv_shift_range,
                    reuse=False,
                    output_dim=args.num_bits_per_vector,
                    clip_value=args.clip_value,
                    init_mode=args.init_mode
                )
            else:
                def single_cell(num_units):
                    return tf.compat.v1.nn.rnn_cell.BasicLSTMCell(num_units, forget_bias=1.0)

                controller = tf.compat.v1.nn.rnn_cell.MultiRNNCell(
                    [single_cell(args.num_units) for _ in range(args.num_layers)])

                cell = NTMCell(controller, args.num_memory_locations, args.memory_size,
                               args.num_read_heads, args.num_write_heads, shift_range=args.conv_shift_range,
                               output_dim=args.num_bits_per_vector,
                               clip_value=args.clip_value)

        output_sequence, _ = tf.compat.v1.nn.dynamic_rnn(
            cell=cell,
            inputs=self.inputs,
            time_major=False,
            dtype=tf.float32,
            initial_state=initial_state if args.mann == 'none' else None)

        task_to_offset = {
            CopyTask.name: lambda: CopyTask.offset(self.max_seq_len),
            AssociativeRecallTask.name: lambda: AssociativeRecallTask.offset(self.max_seq_len),
            SumTask.name: lambda: SumTask.offset(self.max_seq_len),
            AverageSumTask.name: lambda: AverageSumTask.offset(self.max_seq_len, args.num_experts)
        }
        try:
            where_output_begins = task_to_offset[args.task]()
            self.output_logits = output_sequence[:, where_output_begins:, :]
        except KeyError:
            raise UnknownTaskError(f'No information on output slicing of model for "{args.task}" task')

        # Intentionally put in a map, so that each new task that is added to the library explicitly fails with
        # the message. Otherwise, code fails during the training process with a strange error
        task_to_activation = {
            CopyTask.name: tf.sigmoid,
            AssociativeRecallTask.name: tf.sigmoid,
            SumTask.name: tf.sigmoid,
            AverageSumTask.name: tf.sigmoid,
        }
        try:
            self.outputs = task_to_activation[args.task](self.output_logits)
        except KeyError:
            raise UnknownTaskError(f'No information on activation on model outputs for "{args.task}" task')


class BuildTModel(BuildModel):
    def __init__(self, max_seq_len, inputs, outputs):
        super(BuildTModel, self).__init__(max_seq_len, inputs)

        if is_current_task_supported(args.task):
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=outputs, logits=self.output_logits)
            self.loss = tf.reduce_sum(cross_entropy) / args.batch_size
        else:
            raise UnknownTaskError(f'No information how to calculate loss for {args.task} task')

        if args.optimizer == 'RMSProp':
            optimizer = tf.train.RMSPropOptimizer(args.learning_rate, momentum=0.9, decay=0.9)
        elif args.optimizer == 'Adam':
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=args.learning_rate)

        trainable_variables = tf.compat.v1.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, trainable_variables), args.max_grad_norm)
        self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables))


def is_current_task_supported(task):
    return task in (CopyTask.name, AssociativeRecallTask.name, SumTask.name, AverageSumTask.name)


if __name__ == '__main__':
    args = create_argparser().parse_args()

    if args.mann == 'ntm':
        if args.use_local_impl:
            print('Using local implementation')
            from ntm import NTMCell
        else:
            print('Using contrib implementation')
            from tensorflow.contrib.rnn.python.ops.rnn_cell import NTMCell

    if args.verbose:
        os.makedirs('head_logs', exist_ok=True)
        constants.HEAD_LOG_FILE = 'head_logs/{0}.p'.format(args.experiment_name)
        constants.GENERALIZATION_HEAD_LOG_FILE = 'head_logs/generalization_{0}.p'.format(args.experiment_name)

    tf.compat.v1.disable_v2_behavior()
    if args.device == "gpu":
        device_name = "/gpu:0"
    else:
        device_name = "/cpu:0"
    with tf.device(device_name):
        with tf.compat.v1.variable_scope('root'):
            max_seq_len_placeholder = tf.compat.v1.placeholder(tf.int32)
            inputs_placeholder = tf.compat.v1.placeholder(tf.float32,
                                                          shape=(args.batch_size, None, args.num_bits_per_vector + 1))
            outputs_placeholder = tf.compat.v1.placeholder(tf.float32,
                                                           shape=(args.batch_size, None, args.num_bits_per_vector))
            model = BuildTModel(max_seq_len_placeholder, inputs_placeholder, outputs_placeholder)
            initializer = tf.compat.v1.global_variables_initializer()
    if args.device == "gpu":
        device_name = "/gpu:0"
    else:
        device_name = "/cpu:0"
    with tf.device(device_name):
        with tf.compat.v1.variable_scope('root'):
            max_seq_len_placeholder = tf.compat.v1.placeholder(tf.int32)
            inputs_placeholder = tf.compat.v1.placeholder(tf.float32,
                                                          shape=(args.batch_size, None, args.num_bits_per_vector + 1))
            outputs_placeholder = tf.compat.v1.placeholder(tf.float32,
                                                           shape=(args.batch_size, None, args.num_bits_per_vector))
            model = BuildTModel(max_seq_len_placeholder, inputs_placeholder, outputs_placeholder)
            initializer = tf.global_variables_initializer()

    saver = tf.compat.v1.train.Saver(max_to_keep=10)
    tf.debugging.set_log_device_placement(True)
    sess = tf.compat.v1.Session()
    saver = tf.train.Saver(max_to_keep=10)
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    if not args.continue_training_from_checkpoint:
        print(f'Tensorflow initializing the model')
        sess.run(initializer)
    else:
        latest_checkpoint_path = tf.train.latest_checkpoint(args.continue_training_from_checkpoint)
        print(f'Tensorflow reading {latest_checkpoint_path} checkpoint')
        saver.restore(sess, latest_checkpoint_path)
        print(f'Tensorflow loaded {latest_checkpoint_path} checkpoint')
    tf.compat.v1.get_default_graph().finalize()

    # training
    convergence_on_target_task = None
    convergence_on_multi_task = None
    performance_on_target_task = None
    performance_on_multi_task = None
    generalization_from_target_task = None
    generalization_from_multi_task = None
    multi_task_error = None
    target_task_error = None
    progress_error = None
    convergence_error = None
    target_point = None
    exp3s = None
    data_generator = None
    curriculum_point = None
    task = None

    if args.task == CopyTask.name:
        data_generator = CopyTaskData()
        target_point = args.max_seq_len
        curriculum_point = 1 if args.curriculum not in ('prediction_gain', 'none') else target_point
        progress_error = 1.0
        convergence_error = 0.1

        if args.curriculum == 'prediction_gain':
            exp3s = Exp3S(args.max_seq_len, 0.001, 0, 0.05)
    elif args.task == AssociativeRecallTask.name:
        data_generator = AssociativeRecallData()
        target_point = args.max_seq_len
        curriculum_point = 2 if args.curriculum not in ('prediction_gain', 'none') else target_point
        progress_error = 1.0
        convergence_error = 0.1

        if args.curriculum == 'prediction_gain':
            exp3s = Exp3S(args.max_seq_len - 1, 0.001, 0, 0.05)
    elif args.task == SumTask.name:
        data_generator = SumTaskData()
        target_point = args.max_seq_len
        # TODO: investigate what curriculum point is
        curriculum_point = None  # 1 if args.curriculum not in ('prediction_gain', 'none') else target_point
        progress_error = 1.0
        convergence_error = 0.1
    elif args.task == AverageSumTask.name:
        data_generator = AverageSumTaskData()
        target_point = args.max_seq_len
        # TODO: investigate what curriculum point is
        curriculum_point = None  # 1 if args.curriculum not in ('prediction_gain', 'none') else target_point
        progress_error = 1.0
        convergence_error = 0.1
    else:
        raise UnknownTaskError(f'No information on the way to generate data for {args.task} task')

    if data_generator is None:
        sys.exit(f'Data generation rules for "{args.task}" are not specified')

    if args.verbose:
        pickle.dump({target_point: []}, open(constants.HEAD_LOG_FILE, "wb"))
        pickle.dump({}, open(constants.GENERALIZATION_HEAD_LOG_FILE, "wb"))

    for i in range(args.continue_training_from_train_step + 1, args.num_train_steps):
        if args.curriculum == 'prediction_gain':
            if args.task == CopyTask.name:
                task = 1 + exp3s.draw_task()
            elif args.task == AssociativeRecallTask.name:
                task = 2 + exp3s.draw_task()

        if not is_current_task_supported(args.task):
            raise UnknownTaskError(f'No information on how to properly initiate data generation for {args.task} task')

        generator_args = dict(
            num_batches=1,
            batch_size=args.batch_size,
            bits_per_vector=args.num_bits_per_vector,
            curriculum_point=curriculum_point if args.curriculum != 'prediction_gain' else task,
            max_seq_len=args.max_seq_len,
            curriculum=args.curriculum,
            pad_to_max_seq_len=args.pad_to_max_seq_len
        )

        if args.task == AverageSumTask.name:
            generator_args['numbers_quantity'] = args.num_experts

        seq_len, inputs, labels = data_generator.generate_batches(**generator_args)[0]

        train_loss, _, outputs = sess.run([model.loss, model.train_op, model.outputs],
                                          feed_dict={
                                              inputs_placeholder: inputs,
                                              outputs_placeholder: labels,
                                              max_seq_len_placeholder: seq_len
                                          })

        if args.curriculum == 'prediction_gain':
            loss, _ = run_eval(sess, model, inputs_placeholder, outputs_placeholder, max_seq_len_placeholder,
                               data_generator, args, target_point, labels, outputs, inputs, [(seq_len, inputs, labels)])
            v = train_loss - loss
            exp3s.update_w(v, seq_len)

        avg_errors_per_seq = data_generator.error_per_seq(labels, outputs, args.batch_size)

        if args.verbose:
            logger.info('Train loss ({0}): {1}'.format(i, train_loss))
            logger.info('curriculum_point: {0}'.format(curriculum_point))
            logger.info('Average errors/sequence: {0}'.format(avg_errors_per_seq))
            logger.info('TRAIN_PARSABLE: {0},{1},{2},{3}'.format(i, curriculum_point, train_loss, avg_errors_per_seq))

        if i % args.steps_per_eval == 0:
            should_skip_multi_task = args.task in (SumTask.name, AverageSumTask.name)

            target_task_error, target_task_loss, multi_task_error, multi_task_loss, curriculum_point_error, \
            curriculum_point_loss = eval_performance(sess, data_generator, args, model,
                                                     target_point, labels, outputs, inputs,
                                                     inputs_placeholder, outputs_placeholder, max_seq_len_placeholder,
                                                     curriculum_point if args.curriculum != 'prediction_gain' else None,
                                                     store_heat_maps=args.verbose,
                                                     skip_multi_task=should_skip_multi_task)

            if (convergence_on_multi_task is None and
                    multi_task_error is not None and  # condition inserted due to SumTask
                    multi_task_error < convergence_error):
                convergence_on_multi_task = i

            if convergence_on_target_task is None and target_task_error < convergence_error:
                convergence_on_target_task = i

            gen_evaled = False
            if convergence_on_multi_task is not None and (
                    performance_on_multi_task is None or multi_task_error < performance_on_multi_task):
                performance_on_multi_task = multi_task_error
                generalization_from_multi_task = eval_generalization(sess, model, inputs_placeholder,
                                                                     outputs_placeholder, max_seq_len_placeholder,
                                                                     data_generator, args, target_point, labels,
                                                                     outputs, inputs)
                gen_evaled = True

            if convergence_on_target_task is not None and (
                    performance_on_target_task is None or target_task_error < performance_on_target_task):
                performance_on_target_task = target_task_error
                if gen_evaled:
                    generalization_from_target_task = generalization_from_multi_task
                else:
                    generalization_from_target_task = eval_generalization(sess, model, inputs_placeholder,
                                                                          outputs_placeholder, max_seq_len_placeholder,
                                                                          data_generator, args, target_point, labels,
                                                                          outputs, inputs)

            print(curriculum_point_error)
            print(progress_error)
            if (curriculum_point_error is not None and  # condition inserted due to SumTask
                    curriculum_point_error < progress_error):
                if args.task == CopyTask.name:
                    curriculum_point = min(target_point, 2 * curriculum_point)
                elif args.task == AssociativeRecallTask.name:
                    curriculum_point = min(target_point, curriculum_point + 1)

            save_session_as_tf_checkpoint(sess, saver, str(i), bits_per_number=args.max_seq_len)

            logger.info('----EVAL----')
            logger.info('target task error/loss: {0},{1}'.format(target_task_error, target_task_loss))
            logger.info('multi task error/loss: {0},{1}'.format(multi_task_error, multi_task_loss))
            logger.info('curriculum point error/loss ({0}): {1},{2}'.format(curriculum_point, curriculum_point_error,
                                                                            curriculum_point_loss))
            logger.info('EVAL_PARSABLE: {0},{1},{2},{3},{4},{5},{6},{7}'.format(i, target_task_error, target_task_loss,
                                                                                multi_task_error, multi_task_loss,
                                                                                curriculum_point,
                                                                                curriculum_point_error,
                                                                                curriculum_point_loss))

    if convergence_on_multi_task is None:
        print('In convergence_on_multi_task')
        performance_on_multi_task = multi_task_error
        generalization_from_multi_task = eval_generalization(sess, model, inputs_placeholder, outputs_placeholder,
                                                             max_seq_len_placeholder, data_generator, args,
                                                             target_point, labels, outputs, inputs)

    if convergence_on_target_task is None:
        print('In convergence_on_target_task')
        performance_on_target_task = target_task_error
        generalization_from_target_task = eval_generalization(sess, model, inputs_placeholder, outputs_placeholder,
                                                              max_seq_len_placeholder, data_generator, args,
                                                              target_point, labels, outputs, inputs)

    logger.info('----SUMMARY----')
    logger.info('convergence_on_target_task: {0}'.format(convergence_on_target_task))
    logger.info('performance_on_target_task: {0}'.format(performance_on_target_task))
    logger.info('convergence_on_multi_task: {0}'.format(convergence_on_multi_task))
    logger.info('performance_on_multi_task: {0}'.format(performance_on_multi_task))

    logger.info('SUMMARY_PARSABLE: {0},{1},{2},{3}'.format(convergence_on_target_task, performance_on_target_task,
                                                           convergence_on_multi_task, performance_on_multi_task))

    logger.info('generalization_from_target_task: {0}'.format(
        ','.join(map(str, generalization_from_target_task)) if generalization_from_target_task is not None else None))
    logger.info('generalization_from_multi_task: {0}'.format(
        ','.join(map(str, generalization_from_multi_task)) if generalization_from_multi_task is not None else None))

    logger.info(f'Trained the model after {args.num_train_steps} steps.')

    save_session_as_tf_checkpoint(sess, saver, 'final', bits_per_number=args.max_seq_len)

    inputs, outputs = analyze_inputs_outputs(model.outputs.graph)
    logger.info(f'Model inputs: {inputs}')
    logger.info(f'Model outputs: {outputs}')
