# make the script verbose
set -x

# Watching the training:
#
# tail -f ~/projects/tf1-approved-NeuralTuringMachine/out.log
# GPU load
# nvidia-smi --query-gpu=utilization.gpu --format=csv --loop=1
# watch -n0.1 nvidia-smi

# Running the training:

pushd ~/NeuralTuringMachine/
#source venv/bin/activate
rm -rf models/*
rm -rf out.log
popd

pushd ~/NeuralTuringMachine/
nohup bash scripts/run.sh > out.log &
popd
