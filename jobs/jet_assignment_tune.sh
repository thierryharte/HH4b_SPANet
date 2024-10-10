#!/bin/bash
SPANET_DIR=$HOME/SPANet
HH4b_SPANET_DIR=$HOME/HH4b_SPANet
NUM_GPU=1

# Create venv in local job dir
python -m venv myenv --system-site-packages
source myenv/bin/activate

# Install SPANet in virtual environment
cd $SPANET_DIR
pip install torch===2.2.2
pip install -e .

# Install ttHbb_SPANet in virtual environment
cd $HH4b_SPANET_DIR
pip install "ray[tune]" hyperopt

export SEED=$3


# Launch training
if [ $# -eq 3 ]; then
    python -m spanet.tune \
           -o $1 \
           -n $2 \
           --log_dir $2\
           -g $NUM_GPU
else
    python -m spanet.tune \
           -o $1 \
           -n $2 \
           --log_dir $2\
           -g $NUM_GPU\
           "${@:4}"
fi
