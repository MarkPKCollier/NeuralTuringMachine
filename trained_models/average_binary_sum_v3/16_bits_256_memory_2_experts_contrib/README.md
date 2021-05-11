# Training information

## General

|Training aspect | Description |
|:--|:--|
|Date| 2021-05-05|
|Task| average sum (16-bit binary numbers)|
|Error per sequence (start)| 7.96875 |
|Loss function value (start)| 11.695734119415283 |
|Error per sequence| 5.578125 |
|Loss function value| 7.814896202087402 |
|Training iterations| 760000 |
|Additional parameters| `num_experts=2` |

## Training command

```bash
python run_tasks.py --experiment_name experiment --verbose no \
                    --num_train_steps 1000000 --steps_per_eval 10000 \
                    --use_local_impl yes --curriculum none --num_bits_per_vector 3 \
                    --num_memory_locations 256 --max_seq_len 16 \
                    --task average_sum --num_experts 2
```

## Inference command

**N/A**

## Logs

[Logs](./out.log)
