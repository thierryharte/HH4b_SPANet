# HH4b_SPANet
Repository for development of a signal vs background classifier in the HH4b analysis based on SPANet.

## Running SPANet within the `cmsml` docker container

In order to use the SPANet package we use the prebuilt **apptainer** image for machine learning applications in CMS, [`cmsml`](https://hub.docker.com/r/cmsml/cmsml).

First we activate the apptainer environment on **lxplus** with the following command:

```bash
apptainer shell -B /afs -B /eos --nv /cvmfs/unpacked.cern.ch/registry.hub.docker.com/cmsml/cmsml:latest
```

All the common packages for machine learning applications are now available in a singularity shell.
We can proceed by installing `SPANet` in a virtual environment inside this Docker container:
```bash
# Clone locally the SPANet repository
git clone git@github.com:Alexanders101/SPANet.git
cd SPANet

# Create a local virtual environment using the packages defined in the apptainer image
python -m venv --system-site-packages myenv

# Activate the environment
source myenv/bin/activate

# Install in EDITABLE mode
pip install -e .
```

The next time the user enters in the apptainer the virtual environment needs to be activated.
```bash
#Enter the image
apptainer shell -B /afs -B /eos --nv /cvmfs/unpacked.cern.ch/registry.hub.docker.com/cmsml/cmsml:latest

# Activate the virtual environment
cd SPANet
source myenv/bin/activate
```

To check that SPANet is correctly installed in the environment, run the following command:
```bash
python -m spanet.train --help
```

## Dataset creation

### Coffea to H5 conversion
In order to create the `.h5` dataset from the `.coffea` output file, one can use the following command:
```bash
cd HH4b_SPANet
python utils/dataset/coffea_to_h5.py -i input.coffea 
```

### Coffea to Parquet conversion
In order to create the `.parquet` dataset from the `.coffea` output file, one can use the following command:
```bash
cd HH4b_SPANet
python utils/dataset/coffea_to_parquet.py -i input.coffea -o output_folder
```

The script will produce an output file for each sample in the `.parquet` format, saved in the folder `output_folder`.

### Parquet to H5 conversion
Once the `.parquet` file is saved, the `.h5` file in the SPANet format can be produced using the following command:
```bash
python utils/dataset/parquet_to_h5.py -i input.parquet -o output.h5
```

## Train SPANet model for jet assignment
In order to train the SPANet model for jet assignment, run the following command:
```bash
python -m spanet.train -of options_file --gpus 1
```

## Compute predictions
In order to compute the predictions from a previously trained SPANet model, one has to run the following command:
```bash
python -m spanet.predict $LOG_DIRECTORY predicitons.h5 -tf input.h5 --gpu
```
where `$LOG_DIRECTORY` is the output folder where the checkpoints of the trained SPANet model are saved, `predicitons.h5` is the customizable name of the output file containing the predictions and `input.h5` is the input `.h5` file in SPANet format. With the `--gpu` flag one can profit from the available GPUs.

## Train on HTCondor
In order to train the SPANet model on HTCondor, one can use the following command:
```bash
python jobs/submit_jobs_seed.py -o <options_files/option_file.json> -c <jobs/config/config.yaml> -s <start_seed>:<end_seed> -a <"additional arguments to pass to spanet.train"> --suffix <directory_suffix>
```