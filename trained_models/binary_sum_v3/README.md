# Training information

## General

|Training aspect | Description |
|:--|:--|
|Date| 2021-03-11|
|Task| sum (16-bit binary numbers)|
|Error per sequence (start)| 8.390625 |
|Loss function value (start)| 11.78230562210083 |
|Error per sequence| 8.58125 |
|Loss function value| 11.793183040618896|
|Training iterations| 1000000 |

## Training command

```bash
python run_tasks.py --experiment_name experiment --verbose no \
                      --num_train_steps 1000000 --steps_per_eval 1000 --use_local_impl yes \
                      --curriculum none \
                      --num_bits_per_vector 3 --max_seq_len 16 --task sum
```

## Inference command

```bash
python infer.py --frozen_model_filename trained_models/binary_sum_v3/frozen_graph.pb \
                --bits_per_number 16
```

## Logs

[Logs](./out.log)
