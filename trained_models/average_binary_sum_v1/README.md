# Training information

## General

|Training aspect | Description |
|:--|:--|
|Date| 2021-03-12|
|Task| average sum (4-bit binary numbers)|
|Error per sequence (start)| 1.7875 |
|Loss function value (start)| 2.6775789499282836 |
|Error per sequence| 0.0 |
|Loss function value| 0.006404294399544597|
|Training iterations| 50000 |
|Additional parameters| `num_experts=3` |

## Training command

```bash
python run_tasks.py --experiment_name experiment --verbose no \
                      --num_train_steps 100000 --steps_per_eval 1000 --use_local_impl yes \
                      --curriculum none \
                      --num_bits_per_vector 3 \
                      --max_seq_len 4 --task average_sum \
                      --num_experts 3
```

## Inference command

```bash
python infer.py --frozen_model_filename trained_models/average_binary_sum_v1/frozen_graph.pb \
                --bits_per_number 4 --num_experts 3
```

## Logs

[Logs](./out.log)
