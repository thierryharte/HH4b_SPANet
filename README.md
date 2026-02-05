# HH4b_SPANet
Repository with [SPANet]([url](https://github.com/matteomalucchi/SPANet)) configuration for HH4b analysis. Originally forked from https://github.com/mmarchegiani/ttHbb_SPANet.

## Running SPANet within the `cmsml` docker container

In order to use the SPANet package we use the prebuilt **apptainer** image for machine learning applications in CMS, [`cmsml`](https://hub.docker.com/r/cmsml/cmsml).

First we activate the apptainer environment on **lxplus** with the following command (Some of the paths may have to be edited according the the personal setup):

```bash
apptainer shell -B /afs -B /eos/user/t/tharte -B /eos/user/m/mmalucch/spanet_inputs -B /etc/sysconfig/ngbauth-submit -B ${XDG_RUNTIME_DIR} --env KRB5CCNAME="FILE:${XDG_RUNTIME_DIR}/krb5cc" --nv /cvmfs/unpacked.cern.ch/registry.hub.docker.com/cmsml/cmsml:latest
```

All the common packages for machine learning applications are now available in a singularity shell.
We can proceed by installing `SPANet` in a virtual environment inside this Docker container:
```bash
# Clone locally the SPANet repository
git clone git@github.com:matteomalucchi/SPANet.git
cd SPANet

# Create a local virtual environment using the packages defined in the apptainer image
python -m venv --system-site-packages myenv

# Activate the environment
source myenv/bin/activate

# Install in EDITABLE mode
pip install -e .
```

For the script to run the efficiency evaluation plots, a different environment is required.
This repo does not have a direct way to install. An example of such an environement is here. It might be more useful to start a new environment and just add the missing packages until the respective script runs. A better list is to be created.:
<details>
    <summary>Installed packages (more than required)</summary>
    
    absl-py==2.1.0
    aiohappyeyeballs==2.4.3
    aiohttp==3.10.10
    aiosignal==1.3.1
    alabaster==0.7.12
    amqp==5.1.1
    annotated-types==0.7.0
    antlr4-python3-runtime==4.9.3
    appdirs==1.4.4
    argcomplete==1.12.0
    argparse-manpage==4.7
    astroid==2.11.6
    asttokens==2.0.5
    async-timeout==4.0.3
    attrs==20.3.0
    auth-get-sso-cookie==2.3.0
    autopage==0.5.2
    awkward-cpp==42
    awkward==2.7.1
    babel==2.9.1
    backcall==0.1.0
    bcrypt==3.2.2
    beautifulsoup4==4.10.0
    black==24.10.0
    boost-histogram==1.5.0
    build==0.10.0
    cached-property==1.5.2
    cachetools==5.5.0
    certifi==2023.5.7
    certmgr-client==1.19.0
    cffi==1.14.5
    chardet==4.0.0
    click==8.1.7
    cliff==4.0.0
    cloud-init==24.4
    cloudpickle==3.1.0
    cmd2==2.4.2
    coffea==2024.11.0
    collectd-cvmfs==1.3.4
    collectd-dnf==1.1.1
    collectd-heartbeat==1.4.0
    collectd-monit-alarm-actuator==1.1.1
    collectd-monit-alarm-handler==1.2.1
    collectd-processcount==2.6.1
    collectd-puppet==2.0.1
    collectd-systemd==0.0.1
    coloredlogs==15.0.1
    conda-package-handling==1.7.3
    conda==4.14.0
    configobj==5.0.6
    contourpy==1.3.0
    correctionlib==2.6.4
    cramjam==2.8.4
    cryptography==36.0.1
    cycler==0.11.0
    cython==0.29.37
    cytoolz==0.11.2
    dask-awkward==2024.12.1
    dask-histogram==2024.9.1
    dask==2024.8.0
    dbus-python==1.2.18
    debtcollector==2.5.0
    decorator==4.4.2
    defusedxml==0.7.1
    dill==0.3.5.1
    distlib==0.3.2
    distrdf==6.36.4
    distro==1.5.0
    dnspython==2.6.1
    docopt==0.6.2
    docutils==0.16
    dogpile.cache==1.1.8
    elasticsearch==7.17.4
    eventlet==0.33.3
    events==0.4
    executing==0.8.2
    extras==1.0.0
    fail2ban==1.1.0
    fasteners==0.17.3
    filelock==3.7.1
    fixtures==3.0.0
    flake8==7.1.1
    flatbuffers==24.3.25
    fonttools==4.55.0
    frozendict==1.2
    frozenlist==1.4.1
    fsspec-xrootd==0.4.0
    fsspec==2024.9.0
    fts3==3.14.2
    future==0.18.3
    futurist==2.4.1
    gfal2-util==1.9.0
    gpg==1.15.1
    greenlet==3.1.1
    grpcio==1.66.2
    gssapi==1.6.9
    h5py==3.12.1
    hist==2.8.0
    histoprint==2.5.0
    htcondor==24.0.7
    htgettoken==2.0
    httplib2==0.20.3
    humanfriendly==10.0
    idna==2.10
    imagesize==1.2.0
    impacket==0.10.0
    importlib-metadata==8.5.0
    importlib-resources==6.4.5
    influxdb==5.3.1
    iotop==0.6
    ipython==8.5.0
    ipywidgets==8.1.5
    iso8601==1.0.2
    isort==5.10.1
    jedi==0.18.1
    jeepney==0.8.0
    jinja2==3.0.1
    jira==3.5.0
    jmespath==0.9.4
    joblib==1.4.2
    jsonpatch==1.21
    jsonpointer==2.0
    jsonschema==3.2.0
    jupyterlab-widgets==3.0.13
    kerberos==1.3.0
    keyring==21.8.0
    keystoneauth1==5.0.1
    kiwisolver==1.4.1
    koji==1.35.2
    kombu==5.2.4
    landbtools==24.4.4.post2
    lazy-object-proxy==1.7.1
    ldap3==2.8.1
    libcomps==0.1.18
    lightning-utilities==0.11.7
    llvmlite==0.43.0
    locket==1.0.0
    lxml==4.6.5
    lz4==4.3.3
    m2crypto==0.38.0
    mako==1.1.4.dev0
    markdown-it-py==3.0.0
    markdown==3.7
    markupsafe==3.0.2
    matplotlib-inline==0.1.5
    matplotlib==3.9.3
    mccabe==0.7.0
    mdmm==0.1.3
    mdurl==0.1.2
    megabus==2.1.0
    monotonic==1.5
    mplhep-data==0.0.4
    mplhep==0.3.55
    mpmath==1.3.0
    msgpack==1.0.3
    multidict==6.1.0
    munch==2.5.0
    mypy-extensions==1.0.0
    nc-heartbeat==0.4.1
    netaddr==0.10.1
    netifaces==0.10.6
    nftables==0.1
    numba==0.60.0
    numpy==2.0.2
    oauthlib==3.1.1
    olefile==0.46
    omegaconf==2.3.0
    onnxruntime==1.19.2
    opensearch-py==3.0.0
    openstacksdk==0.101.0
    opt-einsum==3.4.0
    oracledb==1.2.2
    os-client-config==2.1.0
    os-service-types==1.7.0
    osc-lib==2.6.2
    osc-placement==4.0.0
    oslo.concurrency==5.0.1
    oslo.config==9.0.0
    oslo.context==5.0.0
    oslo.i18n==5.1.0
    oslo.log==5.0.0
    oslo.messaging==14.0.3
    oslo.metrics==0.5.1
    oslo.middleware==5.0.0
    oslo.serialization==5.0.0
    oslo.service==3.0.0
    oslo.utils==6.0.2
    osprofiler==3.4.3
    packaging==24.2
    pandas==2.2.3
    paramiko==2.12.0
    parso==0.8.3
    partd==1.4.2
    paste==3.5.0
    pastedeploy==2.1.1
    pathspec==0.12.1
    pbr==5.10.0
    pcp==5.0
    pexpect==4.8.0
    pickleshare==0.7.5
    pillow==10.0.1
    pip==25.1.1
    platformdirs==2.5.4
    ply==3.11
    podman-compose==1.5.0
    prettytable==0.7.2
    prometheus-client==0.21.0
    prompt-toolkit==3.0.38
    propcache==0.2.0
    protobuf==5.29.1
    psutil==5.8.0
    ptyprocess==0.6.0
    pure-eval==0.2.2
    py-cpuinfo==8.0.0
    pyarrow==18.0.0
    pyasn1-modules==0.2.8
    pyasn1==0.4.8
    pycairo==1.20.1
    pycodestyle==2.12.1
    pycosat==0.6.3
    pycparser==2.20
    pycryptodomex==3.20.0
    pycurl==7.43.0.6
    pydantic-core==2.23.4
    pydantic==2.9.2
    pyflakes==3.2.0
    pygments==2.18.0
    pygobject==3.40.1
    pyinotify==0.9.6
    pylint==2.13.7
    pymysql==0.10.1
    pynacl==1.4.0
    pyngus==2.3.0
    pynvim==0.5.2
    pyopencl==2022.3.1
    pyopenssl==21.0.0
    pyparsing==2.4.7
    pyperclip==1.8.0
    pyphonebook==2.2.0
    pyproject-hooks==1.0.0
    pyqt5-sip==12.11.1
    pyqt5==5.15.9
    pyrsistent==0.17.3
    pyserial==3.4
    pysocks==1.7.1
    python-barbicanclient==5.4.0
    python-cinderclient==9.1.1
    python-collectd-certificate==0.1.2
    python-collectd-sssd-func==0.1.0
    python-dateutil==2.9.0.post0
    python-dotenv==0.19.2
    python-gitlab==3.9.0
    python-glanceclient==4.1.0
    python-heatclient==3.1.0
    python-ironic-inspector-client==4.8.0
    python-ironicclient==5.0.1
    python-keystoneclient==5.0.1
    python-ldap==3.4.3
    python-linux-procfs==0.7.3
    python-magic==0.4.27
    python-magnumclient==4.0.0
    python-manilaclient==4.1.3
    python-mistralclient==4.5.0
    python-neutronclient==8.1.0
    python-novaclient==18.1.0
    python-octaviaclient==3.1.1
    python-openstackclient==6.0.1-1.1
    python-qpid-proton==0.35.0
    python-swiftclient==4.1.0
    pytools==2022.1.14
    pytorch-lightning==2.4.0
    pytz==2021.1
    pyudev==0.22.0
    pyxattr==0.7.2
    pyyaml==5.4.1
    repoze.lru==0.7
    requests-gssapi==1.2.3
    requests-kerberos==0.12.0
    requests-oauthlib==1.3.0
    requests-toolbelt==0.9.1
    requests==2.25.1
    requestsexceptions==1.4.0
    rfc3986==1.5.0
    rich==13.9.4
    root==6.36.4
    routes==2.5.1
    rpm==4.16.1.3
    ruamel.yaml.clib==0.2.7
    ruamel.yaml==0.16.6
    ruff==0.12.2
    s3cmd==2.4.0
    scikit-learn==1.5.2
    scipy==1.13.1
    secretstorage==3.3.1
    selinux==3.6
    sepolicy==3.6
    setools==4.4.4
    setuptools==53.0.0
    simplejson==3.17.6
    six==1.15.0
    snowballstemmer==1.9.0
    soupsieve==2.4.1
    sphinx==3.4.3
    sphinxcontrib-applehelp==1.0.2
    sphinxcontrib-devhelp==1.0.2
    sphinxcontrib-htmlhelp==1.0.3
    sphinxcontrib-jsmath==1.0.1
    sphinxcontrib-qthelp==1.0.3
    sphinxcontrib-serializinghtml==1.1.4
    sqlalchemy==1.4.45
    sssdconfig==2.9.6
    stack-data==0.5.0
    statsd==3.2.1
    stevedore==4.0.2
    stomp.py==7.0.0
    suds==1.1.2
    sympy==1.13.1
    systemd-python==234
    teigi==4.31.1
    tempita==0.5.2
    tenacity==8.2.3
    tensorboard-data-server==0.7.2
    tensorboard==2.18.0
    testtools==2.5.0
    threadpoolctl==3.5.0
    toml==0.10.2
    tomli==2.0.1
    toolz==1.0.0
    tqdm==4.66.5
    traitlets==5.1.1
    typed-ast==1.5.4
    typing-extensions==4.12.2
    tzdata==2024.2
    uhi==0.5.0
    uproot==5.5.1
    urllib-gssapi==1.0.2
    urllib3==1.26.5
    vector==1.5.2
    vine==5.0.0
    virtualenv==20.21.1
    warlock==1.3.3
    wcwidth==0.2.5
    webob==1.8.7
    werkzeug==3.0.4
    wheel==0.36.2
    widgetsnbextension==4.0.13
    wrapt==1.14.1
    xattr==0.10.0
    xlrd==2.0.1
    xxhash==3.5.0
    yappi==1.3.6
    yarl==1.15.3
    zipp==3.20.2
    
</details>

The next time the user enters in the apptainer the virtual environment needs to be activated.
```bash
#Enter the image
apptainer shell -B /afs -B /eos/user/t/tharte -B /eos/user/m/mmalucch/spanet_inputs -B /etc/sysconfig/ngbauth-submit -B ${XDG_RUNTIME_DIR} --env KRB5CCNAME="FILE:${XDG_RUNTIME_DIR}/krb5cc" --nv /cvmfs/unpacked.cern.ch/registry.hub.docker.com/cmsml/cmsml:latest

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

# Explicit example
python3 /work/tharte/HH4b_SPANet/utils/dataset/coffea_to_h5.py -i $1/output_all.coffea -o $TMPDIR/ -r -c 4b_region
```
This code can also be run in a slurm job using a small script. The script has to be adapted a bit and has hardcoded paths. But the idea is, to have such a script in the folder, where you run it and then you can just execute: `sbatch coffea_to_h5.sh`:
```
#!/bin/bash
#
#SBATCH --mem=6000M
#SBATCH --job-name=coffea_to_h5
#SBATCH --account=t3
#SBATCH --time 01:00:00
#SBATCH --cpus-per-task 24
#SBATCH -o %x-%j.out    # replace default slurm-SLURM_JOB_ID.out; %x is a job-name (or script name when there is no job-name)
#SBATCH -e %x-%j.err    # replace default slurm-SLURM_JOB_ID.err

echo HOME: $HOME 
echo USER: $USER 
echo SLURM_JOB_ID: $SLURM_JOB_ID
echo HOSTNAME: $HOSTNAME

# each worker node has local /scratch space to be used during job run
mkdir -p /scratch/$USER/${SLURM_JOB_ID}
export TMPDIR=/scratch/$USER/${SLURM_JOB_ID}

echo InputFileDir: $1/output_all.coffea

python /work/tharte/HH4b_SPANet/utils/dataset/coffea_to_h5.py -i $1/output_all.coffea -o $TMPDIR/ -r -c inclusive_region
cp $TMPDIR/* $1

rm -rf $TMPDIR
```
TODO: This script will be made more agnostic and put into the repo for easier use.


### Coffea to Parquet conversion
In order to create the `.parquet` dataset from the `.coffea` output file, one can use the following command:
```bash
cd HH4b_SPANet
python utils/dataset/coffea_to_parquet.py -i input.coffea -o output_folder

# Explicit example
python3 /work/tharte/HH4b_SPANet/utils/dataset/coffea_to_parquet.py -i ./output_all.coffea -o . -c 4b_region
```

The script will produce an output file for each sample in the `.parquet` format, saved in the folder `output_folder`.

### Parquet to H5 conversion
Once the `.parquet` file is saved, the `.h5` file in the SPANet format can be produced using the following command:
```bash
python utils/dataset/parquet_to_h5.py -i input.parquet -o output.h5

# Explicit example (saving to a scratch partition)
python3 /work/tharte/HH4b_SPANet/utils/dataset/parquet_to_h5.py -i ./*.parquet -o /scratch/<user>/166814/ -f 0.8
```

## Train SPANet model for jet assignment
In order to train the SPANet model for jet assignment, run the following command:
```bash
python -m spanet.train -of options_file --gpus 1
```

## Train on HTCondor
In order to train the SPANet model on HTCondor, one can use the following command:
```bash
python jobs/submit_jobs_seed.py -o <options_files/option_file.json> -c <jobs/config/config.yaml> -s <start_seed>:<end_seed> -a <"additional arguments to pass to spanet.train"> --suffix <directory_suffix>
```

## Compute predictions
In order to compute the predictions from a previously trained SPANet model, one has to run the following command:
```bash
python -m spanet.predict $LOG_DIRECTORY predicitons.h5 -tf input.h5 --gpu
```
where `$LOG_DIRECTORY` is the output folder where the checkpoints of the trained SPANet model are saved, `predicitons.h5` is the customizable name of the output file containing the predictions and `input.h5` is the input `.h5` file in SPANet format. With the `--gpu` flag one can profit from the available GPUs.

## Convert to onnx for further use
To use the model afterwards with different frameworks, the easiest way is to have an `onnx` file. This can be exported easily like this:
```
python -m spanet.export <path_to_training>/out_seed_trainings_100/version_0/ <output_file_name>.onnx --gpu
```

## Run the efficiency script
This repo contains also a script to determine the pairing efficiency of the models. It runs on the files gained from `spanet.predict`.

To add a new model, you just have to add a sub-dictionary to the `spanet_dict`:
```
f'{spanet_dir}spanet_hh4b_5jets_300_ptreg_loose_s100_btag_wp.h5',
    'true': '5_jets_pt_true_wp_allklambda',
    'label': 'SPANet btag WP',
    'color': 'orange'},
```

And then in the target folder, you can run this:
```
python3 ~/public/Software/HH4b_SPANet/utils/performance/efficiency_studies.py -pd . -k
# Alternatively just run on the data samples, to analyse the mass sculpting:
python3 ~/public/Software/HH4b_SPANet/utils/performance/efficiency_studies.py -pd . -d
```

