# Training information

## General

|Training aspect | Description |
|:--|:--|
|Date| 2021-04-20|
|Task| average sum (6-bit binary numbers)|
|Error per sequence (start)| 1.91875 |
|Loss function value (start)| 2.7651373147964478 |
|Error per sequence| 0.0 |
|Loss function value| 0.0016367931966669858 |
|Training iterations| 30000 |
|Additional parameters| `num_experts=2` |

## Training command

```bash
python run_tasks.py --experiment_name experiment --verbose no \
                    --num_train_steps 1000000 --steps_per_eval 10000 \
                    --use_local_impl no --curriculum none --num_bits_per_vector 3 \
                    --num_memory_locations 256 --max_seq_len 6 \
                    --task average_sum --num_experts 2
```

## Inference command

**N/A**

## Logs

[Logs](./out.log)
