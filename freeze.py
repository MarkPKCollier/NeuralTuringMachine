import argparse
import subprocess
from pathlib import Path

import tensorflow as tf


def analyze_inputs_outputs(graph):
    ops = graph.get_operations()
    outputs_set = set(ops)
    inputs = []
    for op in ops:
        if len(op.inputs) == 0 and op.type != 'Const':
            inputs.append(op)
        else:
            for input_tensor in op.inputs:
                if input_tensor.op in outputs_set:
                    outputs_set.remove(input_tensor.op)
    return inputs, list(outputs_set)


def freeze_graph(directory_path):
    root_path = Path(directory_path)
    meta_path = root_path / 'my_model.ckpt.meta'
    frozen_path = root_path / 'frozen_graph.pb'

    output_node_names = ['root/Sigmoid']  # Output nodes

    with tf.Session() as sess:
        # Restore the graph
        saver = tf.train.import_meta_graph(str(meta_path))

        # Load weights
        latest_checkpoint_path = tf.train.latest_checkpoint(str(root_path))
        print(f'Tensorflow reading {latest_checkpoint_path} before freezing')
        saver.restore(sess, latest_checkpoint_path)

        # Freeze the graph
        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            sess.graph_def,
            output_node_names)

        # Save the frozen graph
        with open(frozen_path, 'wb') as f:
            f.write(frozen_graph_def.SerializeToString())


def run_console_tool(tool_arguments):
    python_executable = Path.cwd() / 'venv' / 'bin' / 'python'  # 'python3'
    options = [
        str(python_executable), __file__,
        *tool_arguments
    ]
    print('[SUBPROCESS] {}'.format(' '.join(options)))
    return subprocess.run(options, capture_output=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", default='./trained_models/binary_sum_v1/', type=str,
                        help="Checkpoint model file to import")
    args = parser.parse_args()

    checkpoints_path = Path(args.checkpoint_dir)

    freeze_graph(checkpoints_path)
