#!/bin/bash
import os
import subprocess
import sys

OPTION = sys.argv[1]
SEED_START = int(sys.argv[2])
SEED_END = sys.argv[3] if len(sys.argv) > 3 else None
dir_name = os.path.splitext(OPTION)[0]

ADDITIONAL_ARGS = ""
if len(sys.argv) > 4:
    ADDITIONAL_ARGS = "--args '{}'".format(' '.join(sys.argv[4:]))

print(ADDITIONAL_ARGS)

a = '--args "{}"'.format(' '.join(sys.argv[4:]))
print(a)

if SEED_END is not None:
    SEED_END = int(SEED_END)
    for SEED in range(SEED_START, SEED_END + 1):
        cmd = 'python3 scripts/submit_to_condor.py --cfg jobs/config/jet_assignment_deep_network.yaml -of {} -l out_spanet_outputs/out_{}/out_seed_trainings_{} --seed {} --basedir {}/HH4b_SPANet --args "{}"'.format(OPTION, dir_name, SEED, SEED, os.getenv("HOME"), ' '.join(sys.argv[4:]))
        subprocess.run(cmd, shell=True)
else:
    for i in range(SEED_START):
        cmd = 'python3 scripts/submit_to_condor.py --cfg jobs/config/jet_assignment_deep_network.yaml -of {} -l out_spanet_outputs/out_{}/out_no_seed_trainings_{} --basedir {}/HH4b_SPANet {}'.format(OPTION, dir_name, i, os.getenv("HOME"), ADDITIONAL_ARGS)
        subprocess.run(cmd, shell=True)

OPTION=$1
# get the seeds boundaries from the command line
SEED_START=$2
SEED_END=$3
dir_name=$(basename $OPTION .json)

ADDITIONAL_ARGS=""
if [ $# -gt 3 ]; then
    ADDITIONAL_ARGS="--args '${@:4}'"
fi
echo $ADDITIONAL_ARGS

a='--args "${@:4}"'
echo $a

# if SEED_START is not None, do the loop over the seeds
if [ ! -z "$SEED_END" ]; then
    # loop over the seeds
    for ((SEED=$SEED_START; SEED<=$SEED_END; SEED++))
    do
        # submit the job
        python3 scripts/submit_to_condor.py --cfg jobs/config/jet_assignment_deep_network.yaml -of $OPTION -l out_spanet_outputs/out_$dir_name/out_seed_trainings_$SEED --seed $SEED --basedir $HOME/HH4b_SPANet --args "${@:4}"
    done
else
    for ((i=0; i<$SEED_START; i++))
    do
        # submit the job
        python3 scripts/submit_to_condor.py --cfg jobs/config/jet_assignment_deep_network.yaml -of $OPTION -l out_spanet_outputs/out_$dir_name/out_no_seed_trainings_$i --basedir $HOME/HH4b_SPANet $ADDITIONAL_ARGS
    done
fi
