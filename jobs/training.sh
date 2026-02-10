#!/usr/bin/bash

export SEED=$3
export NUM_GPU=$4
export SPANET_MAIN_DIR=$5
export SPANET_ENV_DIR=$6
export HOME=$7


echo $SPANET_MAIN_DIR
echo $SPANET_ENV_DIR
echo $HOME

SPANET_DIR=$SPANET_MAIN_DIR/SPANet
HH4b_SPANET_DIR=$SPANET_MAIN_DIR/HH4b_SPANet

echo "$(python --version)"

# Create venv in local job dir
source $SPANET_ENV_DIR/bin/activate

cd $HH4b_SPANET_DIR
echo $HH4b_SPANET_DIR

echo "$(python --version)"


sleep 20m


# Launch training
if [ $# -eq 7 ]; then
    python -m spanet.train \
           --options_file $1 \
           -n $2 \
           --log_dir $2\
           --time_limit 07:00:00:00\
           --gpus $NUM_GPU
else
    python -m spanet.train \
           --options_file $1 \
           -n $2 \
           --log_dir $2\
           --time_limit 07:00:00:00\
           --gpus $NUM_GPU\
           "${@:8}"
fi
