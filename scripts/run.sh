# make the script verbose
set -x

# create a models directory
mkdir models

# get RAM information
free

# get CPU information
lscpu

# get GPU information
lspci -nn | grep '\[03'

# run the training
python3 run_tasks.py --experiment_name experiment --verbose no \
                      --num_train_steps 1000000 --steps_per_eval 10000 --use_local_impl yes \
                      --curriculum none --device gpu\
                      --num_bits_per_vector 3 --num_memory_locations 128\
                      --max_seq_len 4 --task average_sum \
                      --num_experts 3
