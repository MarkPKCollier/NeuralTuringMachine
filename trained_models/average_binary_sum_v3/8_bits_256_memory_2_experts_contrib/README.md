# Training information

## General

|Training aspect | Description |
|:--|:--|
|Date| 2021-04-22|
|Task| average sum (8-bit binary numbers)|
|Error per sequence (start)| 4.284375 |
|Loss function value (start)| 5.5712730407714846 |
|Error per sequence| 0.0 |
|Loss function value| 0.0006552457401994616 |
|Training iterations| 110000 |
|Additional parameters| `num_experts=2` |

## Training command

```bash
python run_tasks.py --experiment_name experiment --verbose no \
                    --num_train_steps 1000000 --steps_per_eval 10000 \
                    --use_local_impl no --curriculum none --num_bits_per_vector 3 \
                    --num_memory_locations 256 --max_seq_len 8 \
                    --task average_sum --num_experts 2
```

## Inference command

**N/A**

## Logs

[Logs](./out.log)
