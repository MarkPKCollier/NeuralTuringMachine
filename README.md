# Neural Turing Machine

This repository contains a Tensorflow implementation of a Neural Turing Machine and the Copy and Associative Recall tasks from the [original paper](https://arxiv.org/abs/1410.5401).

The implementation is based on: https://github.com/snowkylin/ntm but contains some substantial modifications. Most importantly, I backpropagate through the initialization of the memory contents and find this works much better than constant or random memory initialization. Additionally the NTMCell implements the [Tensorflow RNNCell interface](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/RNNCell) so can be used directly with [tf.nn.dynamic_rnn](https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn), etc. I never see loss go to NaN as some other implementations report (although convergence is slow and unstable on some training runs).

I replicated the hyperparameters from the [original paper](https://arxiv.org/abs/1410.5401) for the 2 tasks:

- Memory Size: 128 X 20
- Controller: LSTM - 100 units
- Optimizer: RMSProp - learning rate = 10^-4

The Copy task network was trained on sequences of length sampled from Uniform(1,20) with 8-dimensional random bit vectors. The Associative Recall task network was trained on sequences with the number of items sampled from Uniform(2,6) each item consisted of 3 6-dimensional random bit vectors.

#### Example performance of NTM on Copy task with sequence length = 20 (output is perfect):
![Neural Turing Machine Copy Task - Seq len=20](/img/copy_ntm_20_0.png)

#### Example performance of NTM on Copy task with sequence length = 40 (network only trained on sequences of length up to 20 - performance degrades on example after 36th input):
![Neural Turing Machine Copy Task - Seq len=40](/img/copy_ntm_40_1.png)

#### Example performance of NTM on Associative Recall task with 6 items (output is perfect):
![Neural Turing Machine Associate Recall Task - Seq len=6 items](/img/associative_recall_ntm_6_0.png)

#### Example performance of NTM on Associative Recall task with 12 items (despite only being trained on sequences of up to 6 items to network generalizes perfectly to 12 items):
![Neural Turing Machine Associate Recall Task - Seq len=12 items](/img/associative_recall_ntm_12_0.png)

In order to interpret how the NTM used its external memory I trained a network with 32 memory locations on the Copy task and graphed the read and write head address locations over time.

As you can see from the below graphs, the network first writes the sequence to memory and then reads it back in the same order it wrote it to memory. This uses both the content and location based addressing capabilities of the NTM. The pattern of writes followed by reads is what we would expect of a reasonable solution to the Copy task.

#### Write head locations of NTM with 32 memory locations trained on Copy task:
![Write head locations of NTM with 32 memory locations trained on Copy task](/img/ntm_copy_write_head.png)

#### Read head locations of NTM with 32 memory locations trained on Copy task:
![Read head locations of NTM with 32 memory locations trained on Copy task](/img/ntm_copy_read_head.png)

I also compared the learning curves on 3 training runs (with different random seeds) of my Neural Turing Machine implementation to the [reference Differentiable Neural Computer implementation](https://github.com/deepmind/dnc) and the [Tensorflow BasicLSTM cell](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/BasicLSTMCell).

As you can see from the below graphs on 2 out 3 training runs for both the Copy and Associative Recall tasks the NTM performs comparably to the results in the NTM paper (although there is some instability after convergence on one of the training runs). My NTM implementation is slow to converge on 1 out of the 3 training runs - I am unclear as to why this is the case, perhaps it is our implementation e.g. parameter initialization.

#### Comparison of learning curves of NTM, DNC and LSTM on 3 training runs on the Copy task each:
![Comparison of learning curves of NTM, DNC and LSTM on 3 training runs on the Copy task each](/img/copy_task_archiecture_comparison.png)

#### Comparison of learning curves of NTM, DNC and LSTM on 3 training runs on the Associative Recall task each:
![Comparison of learning curves of NTM, DNC and LSTM on 3 training runs on the Associative Recall task each](/img/associative_recall_archiecture_comparison.png)
