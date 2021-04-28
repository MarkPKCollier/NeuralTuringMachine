# Training information

## General

Date: 2021-02-17
Task: sum (binary numbers)

## Training command

```bash
python3 run_tasks.py --experiment_name experiment --verbose no \
                      --num_train_steps 30000 --steps_per_eval 1000 --use_local_impl yes \
                      --curriculum none \
                      --num_bits_per_vector 3 --max_seq_len 4 --task sum
```

## Logs

[Logs](./out.log)
