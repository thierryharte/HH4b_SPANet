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
# pip install torch===2.0.1
pip install -e .

# Install ttHbb_SPANet in virtual environment
cd $HH4b_SPANET_DIR
# pip install -e .

export SEED=$3


# Launch training
if [ $# -eq 3 ]; then
    python -m spanet.train \
           --options_file $1 \
           -n $2 \
           --log_dir $2\
           --time_limit 07:00:00:00\
           --gpus $NUM_GPU
elif [ $# -eq 4 ]; then
    python -m spanet.train \
           --options_file $1 \
           -n $2 \
           --log_dir $2\
           --time_limit 07:00:00:00\
           --gpus $NUM_GPU\
           "${@:4}"
fi
# elif [ $# -eq 5 ]; then
#     python -m spanet.train \
#            --options_file $1 \
#            -n $2 \
#            --log_dir $2\
#            --checkpoint $4\
#            --time_limit 07:00:00:00\
#            --gpus $NUM_GPU\
#            "${@:5}"