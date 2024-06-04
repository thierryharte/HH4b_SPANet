import os
import subprocess
import sys
import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    "-c", "--config", type=str, required=True, help="Path to the config file"
)
arg_parser.add_argument(
    "-o", "--option", type=str, required=True, help="Path to the option file"
)

arg_parser.add_argument(
    "-n", "--num", type=int, required=False, default=0, help="number of trainings"
)

arg_parser.add_argument(
    "-a",
    "--add_args",
    type=str,
    required=False,
    default=None,
    help="additional arguments",
)
arg_parser.add_argument(
    "--suffix",
    type=str,
    required=False,
    default="",
    help="directory name",
)
args = arg_parser.parse_args()

# get the file name without extension and use it as the directory name
dir_name = os.path.splitext(os.path.basename(args.option))[0] + args.suffix

add_args = f'--args "{args.add_args}"' if args.add_args else ""

print(add_args)

for i in range(args.num):
    cmd = "python3 scripts/submit_to_condor_classification.py  --cfg {} -of {} -l out_spanet_outputs/out_{}/out_training_{} --basedir {}/HH4b_SPANet {}".format(
        args.config,args.option, dir_name, i, os.getenv("HOME"), add_args
    )
    subprocess.run(cmd, shell=True)
    
