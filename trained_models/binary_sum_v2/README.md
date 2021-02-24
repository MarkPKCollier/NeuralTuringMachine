# Training information

## General

Date: 2021-02-23
Task: sum (binary numbers)

## CLI command

```bash
python run_tasks.py --experiment_name experiment --verbose no \
                      --num_train_steps 1000000 --steps_per_eval 1000 --use_local_impl no \
                      --curriculum none \
                      --num_bits_per_vector 3 --max_seq_len 10  --task sum
```

## Logs

[Logs](./out.log)
