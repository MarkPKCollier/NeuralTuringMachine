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
python run_tasks.py --experiment_name experiment --verbose no \
                      --num_train_steps 1000000 --steps_per_eval 1000 --use_local_impl yes \
                      --curriculum none --device cpu\
                      --num_bits_per_vector 3 --num_memory_locations 256\
                      --max_seq_len 4 --task mta \
                      --num_experts 2
