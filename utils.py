import logging
from pathlib import Path

import numpy as np
import tensorflow as tf

from infer import test_model
from freeze import freeze_graph, run_console_tool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def expand(x, dim, N):
    return tf.concat([tf.expand_dims(x, dim) for _ in range(N)], axis=dim)


def learned_init(units):
    return tf.squeeze(tf.contrib.layers.fully_connected(tf.ones([1, 1]), units,
        activation_fn=None, biases_initializer=None))


def create_linear_initializer(input_size, dtype=tf.float32):
    stddev = 1.0 / np.sqrt(input_size)
    return tf.truncated_normal_initializer(stddev=stddev, dtype=dtype)


def save_session_as_tf_checkpoint(session, saver, current_stage, bits_per_number):
    model_dir = Path('./models') / f'{current_stage}'
    model_path = model_dir / 'my_model.ckpt'
    saver.save(session, str(model_path))
    logger.info(f'Saved the trained model at step {current_stage}.')
    # freeze_graph(model_dir)
    # err = test_model(model_dir, bits_per_number=bits_per_number)
    # logger.info(f'Tested frozen model at step {current_stage}, error: {err}.')
    tool_arguments = [
        '--checkpoint_dir',
        str(model_dir)
    ]
    res = run_console_tool(tool_arguments)
    logger.info(res)
    logger.info(f'Froze the model at step {current_stage}.')
