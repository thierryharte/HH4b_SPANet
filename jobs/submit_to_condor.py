import os
import argparse

import htcondor

from omegaconf import OmegaConf

parser = argparse.ArgumentParser()
parser.add_argument("--cfg", type=str, required=True)
parser.add_argument(
    "-of",
    "--options_file",
    type=str,
    default=None,
    help="JSON file with option overloads.",
    required=True,
)
parser.add_argument(
    "-l",
    "--log_dir",
    type=str,
    default=None,
    help="Output directory for the checkpoints and tensorboard logs. Default to current directory.",
    required=True,
)
parser.add_argument(
    "-cf",
    "--checkpoint",
    type=str,
    default=None,
    help="Optional checkpoint to load the training state from. "
    "Fully restores model weights and optimizer state.",
)
parser.add_argument("--dry", action="store_true")
parser.add_argument("--interactive", action="store_true")
parser.add_argument("--ngpu", type=int, default=1)
parser.add_argument("--ncpu", type=int, default=3)
parser.add_argument("--good-gpus", action="store_true")
parser.add_argument("--seed", type=int, default=None, help="Random seed")
parser.add_argument("--outputdir", type=str, default=None, help="Output directory")
parser.add_argument("--args", default="", type=str, help="additional args")
args = parser.parse_args()

basedir = f"{os.path.dirname(os.path.abspath(__file__))}/../"
print(f"basedir {basedir}")

homedir = os.environ["HOME"]
spanet_main_dir = os.environ.get("SPANET_MAIN_DIR", os.environ["HOME"])
spanet_env_dir = os.environ.get("SPANET_ENV_DIR", os.environ["HOME"])
print("homedir: ", homedir)
print("spanet_main_dir: ", spanet_main_dir)
print("spanet_env_dir: ", spanet_env_dir)

print("\nargs:", args.args)
interactive = args.interactive
if interactive:
    print("interactive mode")

singularity_bindings = (
    "/afs, " "/eos/user/t/tharte/, " "/eos/user/m/mmalucch/spanet_infos/",
    "/etc/sysconfig/ngbauth-submit, " "${XDG_RUNTIME_DIR}",
)

col = htcondor.Collector()
credd = htcondor.Credd()
credd.add_user_cred(htcondor.CredTypes.Kerberos, None)

cfg = OmegaConf.load(args.cfg)
outputdir = basedir if not args.outputdir else args.outputdir
model = cfg["model"]
job_flavour = cfg["job_flavour"]
ngpu = cfg["ngpu"]
ncpu = cfg["ncpu"]

# Override defaults by command line arguments
if args.ngpu != parser.get_default("ngpu"):
    ngpu = args.ngpu
if args.ncpu != parser.get_default("ncpu"):
    ncpu = args.ncpu

print("Initializing job submission script...", end="\n\n")
sub = htcondor.Submit()

if interactive:
    sub["InteractiveJob"] = True

if model in ["training","model_tune"]:
    sub["Executable"] = f"{basedir}/jobs/{model}.sh"
    sub["arguments"] = (
        f"-o {basedir}/{args.options_file} -n {outputdir}/{args.log_dir} -s {args.seed} -g {ngpu} -m {spanet_main_dir} -e {spanet_env_dir} -H {homedir} -- {args.args}"
    )
    sub["Output"] = f"{basedir}/{args.log_dir}/{model}-$(ClusterId).$(ProcId).out"
    sub["Error"] = f"{basedir}/{args.log_dir}/{model}-$(ClusterId).$(ProcId).err"
    sub["Log"] = f"{basedir}/{args.log_dir}/{model}-$(ClusterId).log"
    sub["MY.SendCredential"] = True
    sub["MY.SingularityImage"] = (
        '"/cvmfs/unpacked.cern.ch/registry.hub.docker.com/cmsml/cmsml:latest"'
    )
    sub["+JobFlavour"] = f'"{job_flavour}"'
    sub["environment"] = (
        f'SINGULARITY_BIND_EXPR="{singularity_bindings}", '
        f'KRB5CCNAME="FILE:${{XDG_RUNTIME_DIR}}/krb5cc"'
    )
    sub["MY.SingularityUseGPU"] = True
else:
    raise ValueError(f"Model {model} not implemented")

# Load checkpoint
if args.checkpoint:
    sub["arguments"] += f" {args.checkpoint}"

# GPU and CPU requirements
sub["request_cpus"] = f"{args.ncpu}"
sub["request_gpus"] = f"{args.ngpu}"

if args.good_gpus:
    sub["requirements"] = (
        'regexp("A100", TARGET.GPUs_DeviceName) || regexp("V100", TARGET.GPUs_DeviceName)'
    )

print("Submission parameters:")
print(sub, end="\n\n")

print("Creating log folders...")
for folder in ["Output", "Error", "Log"]:
    os.makedirs(os.path.dirname(sub[folder]), exist_ok=True)

if not args.dry:
    print("Starting Condor scheduler...")
    client = htcondor.Schedd()
    result = client.submit(sub, count=1)

    print(f"Submitted {result.num_procs()} job(s) to {result.cluster()}")
