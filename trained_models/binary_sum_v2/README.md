# Training information

## General

|Training aspect | Description |
|:--|:--|
|Date| 2021-02-23|
|Task| sum (10-bit binary numbers)|
|Error per sequence (start)| 5.384375|
|Loss function value (start)| 7.627603244781494 |
|Error per sequence| 0.0 |
|Loss function value| 0.0007314569724258035 |
|Training iterations| 115000 |

## Training command

```bash
python run_tasks.py --experiment_name experiment --verbose no \
                      --num_train_steps 1000000 --steps_per_eval 1000 --use_local_impl no \
                      --curriculum none \
                      --num_bits_per_vector 3 --max_seq_len 10  --task sum
```

## Inference command

```bash
python infer.py --frozen_model_filename trained_models/binary_sum_v2/frozen_graph.pb \
                --bits_per_number 10
```

## Logs

[Logs](./out.log)
