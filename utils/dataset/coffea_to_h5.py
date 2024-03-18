import subprocess
import argparse
import os


parser = argparse.ArgumentParser(
    description="Convert awkward ntuples in coffea files to h5 files."
)
parser.add_argument("-i", "--input", type=str, required=True, help="Input coffea file")
parser.add_argument("-o", "--output", type=str, default="", help="Output directory")
parser.add_argument(
    "-c",
    "--cat",
    type=str,
    default="4b_region",
    required=False,
    help="Event category",
)
parser.add_argument(
    "-f",
    "--frac-train",
    type=float,
    default=0.8,
    help="Fraction of events to use for training",
)
parser.add_argument(
    "-d",
    "--dataset",
    type=str,
    default="GluGlutoHHto4B",
    help="Dataset name",
)

args = parser.parse_args()

out_dir = args.output if args.output else os.path.dirname(args.input)

coffea_to_parquet = f"python coffea_to_parquet.py -i {args.input} -o {os.path.dirname(args.input)} -c {args.cat}"
subprocess.run(coffea_to_parquet, shell=True)

parquet_to_h5 = f"python parquet_to_h5.py -i {os.path.dirname(args.input)}/{args.dataset}.parquet -o {out_dir} -f {args.frac_train}"
subprocess.run(parquet_to_h5, shell=True)
