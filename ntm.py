# credit: thanks to https://github.com/snowkylin/ntm for the basis of this code
# the major changes made are to make this compatible with the abstract class tf.contrib.rnn.RNNCell
# additionally an LSTM controller is used, a feed-forward controller may be used
# and 2 memory inititialization schemes are offered

import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest
import collections
from utils import expand, learned_init, create_linear_initializer

NTMControllerState = collections.namedtuple('NTMControllerState', ('controller_state', 'read_vector_list', 'w_list', 'M'))

class NTMCell(tf.contrib.rnn.RNNCell):
    def __init__(self, controller_layers, controller_units, memory_size, memory_vector_dim, read_head_num, write_head_num,
                 addressing_mode='content_and_location', shift_range=1, reuse=False, output_dim=None, clip_value=20,
                 init_mode='learned'):
        self.controller_layers = controller_layers
        self.controller_units = controller_units
        self.memory_size = memory_size
        self.memory_vector_dim = memory_vector_dim
        self.read_head_num = read_head_num
        self.write_head_num = write_head_num
        self.addressing_mode = addressing_mode
        self.reuse = reuse
        self.clip_value = clip_value

        def single_cell(num_units):
            return tf.contrib.rnn.BasicLSTMCell(num_units, forget_bias=1.0)

        self.controller = tf.contrib.rnn.MultiRNNCell([single_cell(self.controller_units) for _ in range(self.controller_layers)])

        self.init_mode = init_mode

        self.step = 0
        self.output_dim = output_dim
        self.shift_range = shift_range

        self.o2p_initializer = create_linear_initializer(self.controller_units)
        self.o2o_initializer = create_linear_initializer(self.controller_units + self.memory_vector_dim * self.read_head_num)

    def __call__(self, x, prev_state):
        prev_read_vector_list = prev_state.read_vector_list

        controller_input = tf.concat([x] + prev_read_vector_list, axis=1)
        with tf.variable_scope('controller', reuse=self.reuse):
            controller_output, controller_state = self.controller(controller_input, prev_state.controller_state)

        num_parameters_per_head = self.memory_vector_dim + 1 + 1 + (self.shift_range * 2 + 1) + 1
        num_heads = self.read_head_num + self.write_head_num
        total_parameter_num = num_parameters_per_head * num_heads + self.memory_vector_dim * 2 * self.write_head_num
        with tf.variable_scope("o2p", reuse=(self.step > 0) or self.reuse):
            parameters = tf.contrib.layers.fully_connected(
                controller_output, total_parameter_num, activation_fn=None,
                weights_initializer=self.o2p_initializer)
            parameters = tf.clip_by_value(parameters, -self.clip_value, self.clip_value)
        head_parameter_list = tf.split(parameters[:, :num_parameters_per_head * num_heads], num_heads, axis=1)
        erase_add_list = tf.split(parameters[:, num_parameters_per_head * num_heads:], 2 * self.write_head_num, axis=1)

        prev_w_list = prev_state.w_list
        prev_M = prev_state.M
        w_list = []
        for i, head_parameter in enumerate(head_parameter_list):
            k = tf.tanh(head_parameter[:, 0:self.memory_vector_dim])
            beta = tf.nn.softplus(head_parameter[:, self.memory_vector_dim])
            g = tf.sigmoid(head_parameter[:, self.memory_vector_dim + 1])
            s = tf.nn.softmax(
                head_parameter[:, self.memory_vector_dim + 2:self.memory_vector_dim + 2 + (self.shift_range * 2 + 1)]
            )
            gamma = tf.nn.softplus(head_parameter[:, -1]) + 1
            with tf.variable_scope('addressing_head_%d' % i):
                w = self.addressing(k, beta, g, s, gamma, prev_M, prev_w_list[i])
            w_list.append(w)

        # Reading (Sec 3.1)

        read_w_list = w_list[:self.read_head_num]
        read_vector_list = []
        for i in range(self.read_head_num):
            read_vector = tf.reduce_sum(tf.expand_dims(read_w_list[i], dim=2) * prev_M, axis=1)
            read_vector_list.append(read_vector)

        # Writing (Sec 3.2)

        write_w_list = w_list[self.read_head_num:]
        M = prev_M
        for i in range(self.write_head_num):
            w = tf.expand_dims(write_w_list[i], axis=2)
            erase_vector = tf.expand_dims(tf.sigmoid(erase_add_list[i * 2]), axis=1)
            add_vector = tf.expand_dims(tf.tanh(erase_add_list[i * 2 + 1]), axis=1)
            M = M * (tf.ones(M.get_shape()) - tf.matmul(w, erase_vector)) + tf.matmul(w, add_vector)

        if not self.output_dim:
            output_dim = x.get_shape()[1]
        else:
            output_dim = self.output_dim
        with tf.variable_scope("o2o", reuse=(self.step > 0) or self.reuse):
            NTM_output = tf.contrib.layers.fully_connected(
                tf.concat([controller_output] + read_vector_list, axis=1), output_dim, activation_fn=None,
                weights_initializer=self.o2o_initializer)
            NTM_output = tf.clip_by_value(NTM_output, -self.clip_value, self.clip_value)

        self.step += 1
        return NTM_output, NTMControllerState(
            controller_state=controller_state, read_vector_list=read_vector_list, w_list=w_list, M=M)

    def addressing(self, k, beta, g, s, gamma, prev_M, prev_w):

        # Sec 3.3.1 Focusing by Content

        # Cosine Similarity

        k = tf.expand_dims(k, axis=2)
        inner_product = tf.matmul(prev_M, k)
        k_norm = tf.sqrt(tf.reduce_sum(tf.square(k), axis=1, keep_dims=True))
        M_norm = tf.sqrt(tf.reduce_sum(tf.square(prev_M), axis=2, keep_dims=True))
        norm_product = M_norm * k_norm
        K = tf.squeeze(inner_product / (norm_product + 1e-8))                   # eq (6)

        # Calculating w^c

        K_amplified = tf.exp(tf.expand_dims(beta, axis=1) * K)
        w_c = K_amplified / tf.reduce_sum(K_amplified, axis=1, keep_dims=True)  # eq (5)

        if self.addressing_mode == 'content':                                   # Only focus on content
            return w_c

        # Sec 3.3.2 Focusing by Location

        g = tf.expand_dims(g, axis=1)
        w_g = g * w_c + (1 - g) * prev_w                                        # eq (7)

        s = tf.concat([s[:, :self.shift_range + 1],
                       tf.zeros([s.get_shape()[0], self.memory_size - (self.shift_range * 2 + 1)]),
                       s[:, -self.shift_range:]], axis=1)
        t = tf.concat([tf.reverse(s, axis=[1]), tf.reverse(s, axis=[1])], axis=1)
        s_matrix = tf.stack(
            [t[:, self.memory_size - i - 1:self.memory_size * 2 - i - 1] for i in range(self.memory_size)],
            axis=1
        )
        w_ = tf.reduce_sum(tf.expand_dims(w_g, axis=1) * s_matrix, axis=2)      # eq (8)
        w_sharpen = tf.pow(w_, tf.expand_dims(gamma, axis=1))
        w = w_sharpen / tf.reduce_sum(w_sharpen, axis=1, keep_dims=True)        # eq (9)

        return w

    def zero_state(self, batch_size, dtype):
        with tf.variable_scope('init', reuse=self.reuse):
            read_vector_list = [expand(tf.tanh(learned_init(self.memory_vector_dim)), dim=0, N=batch_size)
                for i in range(self.read_head_num)]

            w_list = [expand(tf.nn.softmax(learned_init(self.memory_size)), dim=0, N=batch_size)
                for i in range(self.read_head_num + self.write_head_num)]

            controller_init_state = self.controller.zero_state(batch_size, dtype)

            if self.init_mode == 'learned':
                M = expand(tf.tanh(
                    tf.reshape(
                        learned_init(self.memory_size * self.memory_vector_dim),
                        [self.memory_size, self.memory_vector_dim])
                    ), dim=0, N=batch_size)
            elif self.init_mode == 'random':
                M = expand(
                    tf.tanh(tf.get_variable('init_M', [self.memory_size, self.memory_vector_dim],
                        initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5))),
                    dim=0, N=batch_size)
            elif self.init_mode == 'constant':
                M = expand(
                    tf.get_variable('init_M', [self.memory_size, self.memory_vector_dim],
                        initializer=tf.constant_initializer(1e-6)),
                    dim=0, N=batch_size)

            return NTMControllerState(
                controller_state=controller_init_state,
                read_vector_list=read_vector_list,
                w_list=w_list,
                M=M)

    @property
    def state_size(self):
        return NTMControllerState(
            controller_state=self.controller.state_size,
            read_vector_list=[self.memory_vector_dim for _ in range(self.read_head_num)],
            w_list=[self.memory_size for _ in range(self.read_head_num + self.write_head_num)],
            M=tf.TensorShape([self.memory_size * self.memory_vector_dim]))

    @property
    def output_size(self):
        return self.output_dim
