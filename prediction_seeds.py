import os
import subprocess
import sys
import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    "-f", "--file", type=str, required=True, help="Training for prediction"
)
arg_parser.add_argument(
    "-n", "--name", type=str, required=True, help="Name of the new file"
)
arg_parser.add_argument(
    "-t", "--test", type=str, required=True, help="Test file for the evaluation"
)
arg_parser.add_argument(
    "-s", "--seeds", type=str, required=False, default=None, help="start:end"
)
args = arg_parser.parse_args()

if args.seeds:
    seed_start = int(args.seeds.split(":")[0])
    seed_end = int(args.seeds.split(":")[1])


for seed in range(seed_start, seed_end+1):
    cmd= "python -m spanet.predict {}/out_seed_trainings_{}/version_0 prediction_seeds/{}_seed_{}.h5 -tf {} --gpu".format(
        args.file , seed, args.name, seed, args.test)
    subprocess.run(cmd, shell=True)