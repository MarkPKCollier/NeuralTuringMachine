import pickle

import constants


def run_eval(sess, model, inputs_placeholder, outputs_placeholder, max_seq_len_placeholder, data_generator, args,
             target_point, labels, outputs, inputs, batches, store_heat_maps=False, generalization_num=None):
    task_loss = 0
    task_error = 0
    num_batches = len(batches)
    for seq_len, inputs, labels in batches:
        task_loss_, outputs = sess.run([model.loss, model.outputs],
                                       feed_dict={
                                           inputs_placeholder: inputs,
                                           outputs_placeholder: labels,
                                           max_seq_len_placeholder: seq_len
                                       })

        task_loss += task_loss_
        task_error += data_generator.error_per_seq(labels, outputs, args.batch_size)

    if store_heat_maps:
        if generalization_num is None:
            tmp = pickle.load(open(constants.HEAD_LOG_FILE, "rb"))
            tmp[target_point].append({
                'labels': labels[0],
                'outputs': outputs[0],
                'inputs': inputs[0]
            })
            pickle.dump(tmp, open(constants.HEAD_LOG_FILE, "wb"))
        else:
            tmp = pickle.load(open(constants.GENERALIZATION_HEAD_LOG_FILE, "rb"))
            if tmp.get(generalization_num) is None:
                tmp[generalization_num] = []
            tmp[generalization_num].append({
                'labels': labels[0],
                'outputs': outputs[0],
                'inputs': inputs[0]
            })
            pickle.dump(tmp, open(constants.GENERALIZATION_HEAD_LOG_FILE, "wb"))

    task_loss /= float(num_batches)
    task_error /= float(num_batches)
    return task_loss, task_error


def eval_performance(sess, data_generator, args, model, target_point, labels, outputs, inputs, inputs_placeholder,
                     outputs_placeholder, max_seq_len_placeholder, curriculum_point, store_heat_maps=False):
    # target task
    batches = data_generator.generate_batches(
        int(int(args.eval_batch_size / 2) / args.batch_size),
        args.batch_size,
        bits_per_vector=args.num_bits_per_vector,
        curriculum_point=None,
        max_seq_len=args.max_seq_len,
        curriculum='none',
        pad_to_max_seq_len=args.pad_to_max_seq_len
    )

    target_task_loss, target_task_error = run_eval(sess, model,
                                                   inputs_placeholder,
                                                   outputs_placeholder,
                                                   max_seq_len_placeholder,
                                                   data_generator,
                                                   args,
                                                   target_point,
                                                   labels,
                                                   outputs,
                                                   inputs,
                                                   batches,
                                                   store_heat_maps=store_heat_maps)

    # multi-task

    batches = data_generator.generate_batches(
        int(args.eval_batch_size / args.batch_size),
        args.batch_size,
        bits_per_vector=args.num_bits_per_vector,
        curriculum_point=None,
        max_seq_len=args.max_seq_len,
        curriculum='deterministic_uniform',
        pad_to_max_seq_len=args.pad_to_max_seq_len
    )

    multi_task_loss, multi_task_error = run_eval(sess, model, inputs_placeholder, outputs_placeholder,
                                                 max_seq_len_placeholder, data_generator, args, target_point, labels,
                                                 outputs, inputs, batches)

    # curriculum point
    if curriculum_point is not None:
        batches = data_generator.generate_batches(
            int(int(args.eval_batch_size / 4) / args.batch_size),
            args.batch_size,
            bits_per_vector=args.num_bits_per_vector,
            curriculum_point=curriculum_point,
            max_seq_len=args.max_seq_len,
            curriculum='naive',
            pad_to_max_seq_len=args.pad_to_max_seq_len
        )

        curriculum_point_loss, curriculum_point_error = run_eval(sess, model, inputs_placeholder, outputs_placeholder,
                                                                 max_seq_len_placeholder, data_generator, args,
                                                                 target_point, labels, outputs, inputs, batches)
    else:
        curriculum_point_error = curriculum_point_loss = None

    return target_task_error, target_task_loss, multi_task_error, multi_task_loss, curriculum_point_error, curriculum_point_loss


def eval_generalization(sess, model, inputs_placeholder, outputs_placeholder, max_seq_len_placeholder, data_generator,
                        args, target_point, labels, outputs, inputs):
    res = []
    seq_lens = []
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

        loss, error = run_eval(sess, model, inputs_placeholder, outputs_placeholder, max_seq_len_placeholder,
                               data_generator, args, target_point, labels, outputs, inputs, batches,
                               store_heat_maps=args.verbose, generalization_num=i)
        res.append(error)
    return res
