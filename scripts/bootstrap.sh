# make the script verbose
set -x

# Accessing the machine:
#
# ssh root@188.166.169.120
# sudo apt-get update
# sudo apt install git
# git clone https://github.com/demid5111/NeuralTuringMachine
# cd NeuralTuringMachine/
# git checkout feature/demidovs/binary-average-accumulator-dataset-generator
# bash scripts/bootstrap.s

# Preparing fresh machine for the first usage:

sudo apt-get --yes update
sudo update-pciids
sudo apt-get --yes upgrade
sudo apt-get --yes install software-properties-common
sudo add-apt-repository --yes ppa:deadsnakes/ppa
sudo apt-get --yes install python3.7 htop
sudo apt-get --yes install python3-pip
python3.7 -m pip install virtualenv
python3.7 -m virtualenv venv
source venv/bin/activate

git submodule update --init --recursive
python -m pip install -r requirements.txt
python -m pip install -r tasks/operators/tpr_toolkit/requirements.txt
