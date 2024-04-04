#!/bin/bash

source $HOME/condor_env/bin/activate

# get the seeds boundaries from the command line
SEED_START=$1
SEED_END=$2

# loop over the seeds
for ((SEED=$SEED_START; SEED<=$SEED_END; SEED++))
do
    # submit the job
    python3 scripts/submit_to_condor.py --cfg jobs/config/jet_assignment_deep_network.yaml -of options_files/hh4b_5jets_ATLAS_ptreg.json -l $HOME/HH4b_SPANet/out_seed_trainings_$SEED --seed $SEED --basedir $HOME/HH4b_SPANet
done
