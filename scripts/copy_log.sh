# make the script verbose
set -x

WHERE_TO_KEEP=~/Downloads/new_log/
rm -rf ${WHERE_TO_KEEP}
mkdir -p ${WHERE_TO_KEEP}
scp -r root@161.97.147.172:~/NeuralTuringMachine/out.log ${WHERE_TO_KEEP}
