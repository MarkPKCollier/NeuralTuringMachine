# Training information

## General

|Training aspect | Description |
|:--|:--|
|Date| 2021-05-11|
|Task| mta (full binary layout)|
|Error per sequence (start)| 10.0875 |
|Loss function value (start)| -1.163021695613861 |
|Error per sequence| 12.431 |
|Loss function value| -31.342 |
|Training iterations| 42000 |
|Additional parameters| `num_experts=2` |

## Training command

```bash
python run_tasks.py --experiment_name experiment --verbose no \
                    --num_train_steps 1000000 --steps_per_eval 1000 \
                    --use_local_impl yes --curriculum none --device cpu --num_bits_per_vector 3 \
                    --num_memory_locations 256 --max_seq_len 4 \
                    --task mta --num_experts 2
```

## Inference command

**N/A**

## Logs

[Logs](./out.log)
