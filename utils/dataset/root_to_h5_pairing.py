import numpy as np
import os
import h5py
import argparse
#import awkward as ak
import ROOT as root
import uproot

FACTOR = 0.8
PAD_VALUE = 9999
PAD_VALUE_ROOT = -10
shuffidx = [] # Will be filled, as soon as the length of the data is known (at the first shuffling).

parser = argparse.ArgumentParser(
    description="Convert ntuples in root files to h5 files."
)
# get multiple files
parser.add_argument(
    "-bi", "--binput", required=True, help="Input background root file", nargs="+"
)
parser.add_argument(
    "-si", "--sinput", required=True, help="Input signal root file", nargs="+"
)
parser.add_argument("-o", "--output", type=str, default="", help="Output directory (will receive a suffix for training and validation)")
parser.add_argument("-n", "--name", type=str, default="", help="Output file name")
parser.add_argument("-s", "--seed", type=str, default=None, help="Seed for random shuffling")


args = parser.parse_args()
SEED = args.seed if args.seed is not None else 666


def shuffle(data):
    global shuffidx
    if len(shuffidx)<1:
        shuffidx = np.random.RandomState(seed=SEED).permutation(len(full_data["weight"]))
    data = data[shuffidx]
    return data

def create_targets(file_train,file_valid, jets_prov):
    higgs_targets = {1: ["b1", "b2"], 2: ["b3", "b4"]}
    for j in [1, 2]:
        index_b1=np.full(len(jets_prov), 0+(j-1)*2)
        index_b2=np.full(len(jets_prov), 1+(j-1)*2)
        cutoff = int(len(index_b1)*FACTOR)
        #For training dataset
        file_train.create_dataset(
            f"TARGETS/h{j}/{higgs_targets[j][0]}",
            np.shape(index_b1[:cutoff]),
            dtype="int64",
            data=index_b1[:cutoff],
        )
        file_train.create_dataset(
            f"TARGETS/h{j}/{higgs_targets[j][1]}",
            np.shape(index_b2[:cutoff]),
            dtype="int64",
            data=index_b2[:cutoff],
        )
        #For validation dataset
        file_valid.create_dataset(
            f"TARGETS/h{j}/{higgs_targets[j][0]}",
            np.shape(index_b1[cutoff:]),
            dtype="int64",
            data=index_b1[cutoff:],
        )
        file_valid.create_dataset(
            f"TARGETS/h{j}/{higgs_targets[j][1]}",
            np.shape(index_b2[cutoff:]),
            dtype="int64",
            data=index_b2[cutoff:],
        )

def create_inputs(file_train, file_valid, category, subcategory, name, var_array):
    var_array = shuffle(var_array)
    if name == "MASK":
        var_array = var_array.astype("bool")
        print(type(var_array[0][0]))
        var_type = "bool"
    elif name == "signal":
        var_type = "int64"
    else:
        var_type = "float32"

    if subcategory is not None:
        infile_string = f"{category}/{subcategory}/{name}"
    else:    
        infile_string = f"{category}/{name}"
    if name == "weight":
        infile_string = f"WEIGHTS/{name}"
    print(infile_string) # DEBUG
    # first you create the variable and then the dataset
    cutoff = int(len(var_array)*FACTOR)
    var_ds = file_train.create_dataset(infile_string, np.shape(var_array[:cutoff]), dtype=var_type, data=var_array[:cutoff])
    var_ds = file_valid.create_dataset(infile_string, np.shape(var_array[cutoff:]), dtype=var_type, data=var_array[cutoff:])

def create_jets(file_train,file_valid, names, values):
    cat = "INPUTS"
    subcat = "Jet"
    mname, mvals = create_mask(values)
    names.append(mname)
    values = np.vstack((values,mvals[None]))
    for i in range(len(names)):
        # Messed up direction, now have to transpose:
        val_t = np.array(values[i]).T
        val_t[val_t==-10] = PAD_VALUE
        create_inputs(file_train,file_valid,cat, subcat,names[i],val_t)

def create_mask(values):
    mask = ~(values[0] == PAD_VALUE_ROOT)
    return "MASK", mask 

def get_name(name):
    if name=="DeltaRjj":
        return "dR"
    elif name=="DeltaPhijj":
        return "dPhi"
    elif name=="DeltaEtajj":
        return "dEta"
    elif name=="DeltaR":
        return "dR"
    elif name=="wp1_DeltaR":
        return "wp1_dR"
    elif name=="wp2_DeltaR":
        return "wp2_dR"
    elif name=="DeltaPhi":
        return "dPhi"
    elif name=="DeltaEta":
        return "dEta"
    elif name=="minDeltaR_Higgjj":
        return "dR_min"
    elif name=="maxDeltaR_Higgjj":
        return "dR_max"
    elif name=="CosThetaStar":
        return "cos_theta_star"
    elif name=="wp1_CosThetaStar":
        return "wp1_cos_theta_star"
    elif name=="wp2_CosThetaStar":
        return "wp2_cos_theta_star"
    else:
        return name

def convert_name(name):
    splitname = name.split("_")
    # Define which category the parameter fits in
    if splitname[0] in ["weight", "weight_dnn"]:
        category="WEIGHTS"
        subcategory=None
        name=splitname[0]
        return category, subcategory, name
    elif splitname[0] == "signal":
        category="CLASSIFICATIONS"
        subcategory="EVENT"
        name=splitname[0]
        return category, subcategory, name
    else:
        category="INPUTS"
    # Define the subcategory and the name
    if splitname[0]=="hh":
        subcategory="HH"
        if splitname[1]=="vec":
            if splitname[-2] in ["wp1","wp2"]:
                name=get_name(f"{splitname[-2]}_{splitname[-1]}")
            else:
                name=get_name(splitname[-1])
        elif splitname[1] in ["wp1","wp2"]:
            name=get_name(f"{splitname[1]}_{splitname[2]}")
        else:
            name=get_name(splitname[1])
    elif splitname[0] == "higgs1":
        if splitname[1] == "wp1":
            subcategory="HiggsLeadingWp1"
        elif splitname[1] == "wp2":
            subcategory="HiggsLeadingWp2"
        else:
            if splitname[-2] in ["jet1","jet2"]:
                subcategory=f"h1j{splitname[-2][3]}_JETS"
            else:
                subcategory="HiggsLeading"
        name=get_name(splitname[-1])
    elif splitname[0] == "higgs2":
        if splitname[1] == "wp1":
            subcategory="HiggsSubLeadingWp1"
        elif splitname[1] == "wp2":
            subcategory="HiggsSubLeadingWp2"
        else:
            if splitname[-2] in ["jet1","jet2"]:
                subcategory=f"h2j{splitname[-2][3]}_JETS"
            else:
                subcategory="HiggsSubLeading"
        name=get_name(splitname[-1])
    elif splitname[0] == "add":
        if splitname[-2] == "Higgs1":
            subcategory="HiggsLeading"
        elif splitname[-2] == "Higgs2":
            subcategory="HiggsSubLeading"
        else:
            subcategory="j1pt_JETS"
        name=get_name(splitname[-1])
    elif "higgs1" in splitname:
        subcategory="HiggsLeading"
        name=get_name(name)
    elif "higgs2" in splitname:
        subcategory="HiggsSubLeading"
        name=get_name(name)
    else:
        subcategory="Event"
        name=get_name(name)
    return category, subcategory, name


main_dir = args.output if args.output else os.path.dirname(os.environ.get("PWD"))
os.makedirs(main_dir, exist_ok=True)
dfs = []
sigval = []
#read root tree convert it to numpy
print("Background samples")
for filename in list(args.binput):
    with uproot.open(filename) as file:
        tree = file['tree']
        dfs.append(tree.arrays(library="np"))
        print(len(dfs[-1]["weight"]))
        sigval.append(0)
print("Signal samples")
for filename in list(args.sinput):
    with uproot.open(filename) as file:
        tree = file['tree']
        dfs.append(tree.arrays(library="np"))
        print(len(dfs[-1]["weight"]))
        sigval.append(1)
        
# get if the file is sig and bkg and put the flag in the dataset
# then concatenate the two sets.
# there are a few annoying things to consider, so it looks longer than it actually is.
jets_pt = [0,0,0,0,0]
jets_eta = [0,0,0,0,0]
jets_phi = [0,0,0,0,0]
jets_mass = [0,0,0,0,0]
weights = []
name_to_index = {"h1j1": 0, "h1j2":1, "h2j1": 2, "h2j2": 3, "j1pt": 4} 
file_train = h5py.File(f"{main_dir}/{args.name}_train.h5", "w")
file_valid = h5py.File(f"{main_dir}/{args.name}_testing.h5", "w")
full_data = None
for i, dataset in enumerate(dfs):
    if not full_data:
        full_data=dataset
        full_data["signal"]=np.full(len(dataset["weight"]),sigval[i])
    else:
        for key, value in full_data.items():
            if key=="signal":
                full_data[key] = np.concatenate((full_data[key],np.full(len(dataset["weight"]),sigval[i])),0)
            elif key=="weight_dnn":
                if not key in dataset.keys():
                    full_data[key] = np.concatenate((full_data[key],np.full(len(dataset["weight"]),1,)),0) #Hacking 1 into weight_dnn for the non-dnn data. (will be multiplied anyway, so 1 does not change anything).
                else:
                    full_data[key] = np.concatenate((full_data[key],dataset[key]),0)
            elif not key in dataset.keys(): # Hacking, there seem to be parameters that only exist in one of the sets... data from the second set that is not available in the first set is ignored... (TODO: fixthis)
                full_data[key] = np.concatenate((full_data[key],np.full(len(dataset["weight"]),0)),0)
            else:
                full_data[key] = np.concatenate((full_data[key],dataset[key]),0)

# Now run everything over the large dataset
# First create "targets" (h1->b1,b2 and h2->b3,b4)
create_targets(
    file_train,
    file_valid,
    full_data["weight"]
)
for key, values in full_data.items():
    cat, subcat, name = convert_name(key)
    if subcat is None:
        weights.append(values)
    elif subcat[-4:] == "JETS":
        if subcat[0:4] == "j1pt":
            create_inputs(
                    file_train,
                    file_valid,
                    "INPUTS","FifthJet",name,
                    values
                    )
        if name == "pt":
            jets_pt[name_to_index[subcat[0:4]]] = values
        elif name == "eta":
            jets_eta[name_to_index[subcat[0:4]]] = values
        elif name == "phi":
            jets_phi[name_to_index[subcat[0:4]]] = values
        elif name == "mass":
            jets_mass[name_to_index[subcat[0:4]]] = values
        else:
            print("jet_category not found")
        continue
    #else: 
    #    create_inputs(
    #        file_train,
    #        file_valid,
    #        cat, subcat, name,
    #        values
    #    )
create_inputs(file_train,file_valid,"WEIGHTS",None,"weight",weights[0]*weights[1])
## HAndle jets
create_jets(file_train,file_valid, ["pt", "eta", "phi", "mass"],np.array([jets_pt, jets_eta, jets_phi, jets_mass]))


with open(f"{main_dir}/parameters.txt", "w") as f:
    for key in dfs[0].keys():
        f.write(f"{key}\n")

#create_add_variables(file_out, events)
