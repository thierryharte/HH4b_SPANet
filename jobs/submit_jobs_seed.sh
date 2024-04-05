#!/bin/bash

source $HOME/condor_env/bin/activate
echo $HOME

OPTION=$1
# get the seeds boundaries from the command line
SEED_START=$2
SEED_END=$3
dir_name=$(basename $OPTION .json)

ADDITIONAL_ARGS=""
if [ $# -gt 3 ]; then
    ADDITIONAL_ARGS="--args ${@:4}"
fi

# if SEED_START is not None, do the loop over the seeds
if [ ! -z "$SEED_END" ]; then
    # loop over the seeds
    for ((SEED=$SEED_START; SEED<=$SEED_END; SEED++))
    do
        # submit the job
        python3 scripts/submit_to_condor.py --cfg jobs/config/jet_assignment_deep_network.yaml -of $OPTION -l out_spanet_outputs/out_$dir_name/out_seed_trainings_$SEED --seed $SEED --basedir $HOME/HH4b_SPANet $ADDITIONAL_ARGS
    done
else
    for ((i=0; i<$SEED_START; i++))
    do
        # submit the job
        python3 scripts/submit_to_condor.py --cfg jobs/config/jet_assignment_deep_network.yaml -of $OPTION -l out_spanet_outputs/out_$dir_name/out_no_seed_trainings_$i --basedir $HOME/HH4b_SPANet $ADDITIONAL_ARGS
    done
fi
