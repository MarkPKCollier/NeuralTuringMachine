# make the script verbose
set -x

WHERE_TO_KEEP=~/Downloads/new_model/
TRAINING_STEP=110000
rm -rf ${WHERE_TO_KEEP}
mkdir -p ${WHERE_TO_KEEP}
scp -r root@161.97.147.172:~/NeuralTuringMachine/models/${TRAINING_STEP}/ ${WHERE_TO_KEEP}
