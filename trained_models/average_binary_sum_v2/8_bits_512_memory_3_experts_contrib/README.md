# Training information

## General

|Training aspect | Description |
|:--|:--|
|Date| 2021-04-12|
|Task| average sum (8-bit binary numbers)|
|Error per sequence (start)| 2.8875 |
|Loss function value (start)| 4.151074671745301 |
|Error per sequence| 0.95 |
|Loss function value| 1.4034770607948304 |
|Training iterations| 440000 |
|Additional parameters| `num_experts=3` |

## Training command

```bash
python run_tasks.py --experiment_name experiment --verbose no \
                    --num_train_steps 1000000 --steps_per_eval 10000 \
                    --use_local_impl no --curriculum none --num_bits_per_vector 3 \
                    --num_memory_locations 512 --max_seq_len 8 \
                    --task average_sum --num_experts 3
```

## Inference command

**N/A**

## Logs

[Logs](./out.log)
