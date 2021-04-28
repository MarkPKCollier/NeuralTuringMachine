# Training information

## General

|Training aspect | Description |
|:--|:--|
|Date| 2021-04-26|
|Task| average sum (10-bit binary numbers)|
|Error per sequence (start)| 3.346875 |
|Loss function value (start)| 4.891600847244263 |
|Error per sequence| 0.0 |
|Loss function value| 0.0017 |
|Training iterations| 70000 |
|Additional parameters| `num_experts=2` |

## Training command

```bash
python run_tasks.py --experiment_name experiment --verbose no \
                    --num_train_steps 1000000 --steps_per_eval 10000 \
                    --use_local_impl no --curriculum none --num_bits_per_vector 3 \
                    --num_memory_locations 256 --max_seq_len 10 \
                    --task average_sum --num_experts 2
```

## Inference command

**N/A**

## Logs

[Logs](./out.log)
