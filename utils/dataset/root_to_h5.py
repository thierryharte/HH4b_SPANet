import numpy as np
import os
import h5py
import argparse
#import awkward as ak
import ROOT as root
import uproot

parser = argparse.ArgumentParser(
    description="Convert ntuples in root files to h5 files."
)
# get multiple files
parser.add_argument(
    "-i", "--input", required=True, help="Input root file", nargs="+"
)
parser.add_argument("-o", "--output", type=str, default="", help="Output directory")
parser.add_argument("-n", "--name", type=str, default="", help="Output file name", nargs="+")


args = parser.parse_args()


def create_targets(file, jets_prov):
    higgs_targets = {1: ["b1", "b2"], 2: ["b3", "b4"]}
    for j in [1, 2]:
        print(np.shape(jets_prov))
        index_b=np.full(len(jets_prov), -1)
        file.create_dataset(
            f"TARGETS/h{j}/{higgs_targets[j][0]}",
            np.shape(index_b),
            dtype="int64",
            data=index_b,
        )
        file.create_dataset(
            f"TARGETS/h{j}/{higgs_targets[j][1]}",
            np.shape(index_b),
            dtype="int64",
            data=index_b,
        )

def create_inputs(file, category, subcategory, name, var_array):
    if subcategory is not None:
        infile_string = f"{category}/{subcategory}/{name}"
    else:    
        infile_string = f"{category}/{name}"
    if name == "weight":
        infile_string = f"WEIGHTS/{name}"
    print(infile_string) # DEBUG
    # first you create the variable and then the dataset
    var_ds = file.create_dataset(infile_string, np.shape(var_array), dtype="float32", data=var_array)

def create_jets(file, names, values):
    cat = "INPUTS"
    subcat = "JET"
    for i in range(len(names)):
        # Messed up direction, now have to transpose:
        val_t = np.array(values[i]).T.tolist()
        create_inputs(file,cat, subcat,names[i],val_t)

#def create_add_variables(file, events):
#    mask_ds = file.create_dataset(
#        "INPUTS/Jet/MASK", np.shape(events.mask), dtype="bool", data=events.mask
#    )
#
#    weight_array = events.weight
#    weight_ds = file.create_dataset(
#        "WEIGHTS/weight",
#        np.shape(weight_array),
#        dtype="float32",
#        data=weight_array,
#    )
#    sb_array = events.sb
#    sb_ds = file.create_dataset(
#        "CLASSIFICATIONS/EVENT/signal",
#        np.shape(sb_array),
#        dtype="int64",
#        data=sb_array,
#    )

def convert_name(name):
    splitname = name.split("_")
    # Define which category the parameter fits in
    if splitname[0] in ["weight", "weight_dnn"]:
        category="WEIGHTS"
        subcategory=None
        name=splitname[0]
        return category, subcategory, name
    else:
        category="INPUTS"
    # Define the subcategory and the name
    if splitname[0]=="hh":
        subcategory="HH"
        if splitname[1]=="vec":
            if splitname[-2] in ["wp1","wp2"]:
                name=f"{splitname[-2]}_{splitname[-1]}"
            else:
                name=splitname[-1]
        elif splitname[1] in ["wp1","wp2"]:
            name=f"{splitname[1]}_{splitname[2]}"
        else:
            name=splitname[1]
    elif splitname[0] == "higgs1":
        if splitname[1] == "wp1":
            subcategory="LEADING_HIGGS_WP1"
        elif splitname[1] == "wp2":
            subcategory="LEADING_HIGGS_WP2"
        else:
            if splitname[-2] in ["jet1","jet2"]:
                print(f"h1j{splitname[-2][3]}_JETS")
                subcategory=f"h1j{splitname[-2][3]}_JETS"
            else:
                subcategory="LEADING_HIGGS"
        name=splitname[-1]
    elif splitname[0] == "higgs2":
        if splitname[1] == "wp1":
            subcategory="SUBLEADING_HIGGS_WP1"
        elif splitname[1] == "wp2":
            subcategory="SUBLEADING_HIGGS_WP2"
        else:
            if splitname[-2] in ["jet1","jet2"]:
                subcategory=f"h2j{splitname[-2][3]}_JETS"
            else:
                subcategory="SUBLEADING_HIGGS"
        name=splitname[-1]
    elif splitname[0] == "add":
        if splitname[-2] == "Higgs1":
            subcategory="LEADING_HIGGS"
        elif splitname[-2] == "Higgs2":
            subcategory="SUBLEADING_HIGGS"
        else:
            subcategory="j1pt_JETS"
        name=splitname[-1]
    elif "higgs1" in splitname:
        subcategory="LEADING_HIGGS"
        name=name
    elif "higgs2" in splitname:
        subcategory="SUBLEADING_HIGGS"
        name=name
    else:
        subcategory="Event"
        name=name
    return category, subcategory, name


main_dir = args.output if args.output else os.path.dirname(os.environ.get("PWD"))
os.makedirs(main_dir, exist_ok=True)
dfs = []
#read root tree convert it to numpy
for filename in list(args.input):
    with uproot.open(filename) as file:
        tree = file['tree']
        dfs.append(tree.arrays(library="np"))
        
# get if the file is sig and bkg and put the flag in the dataset
jets_pt = [0,0,0,0,0]
jets_eta = [0,0,0,0,0]
jets_phi = [0,0,0,0,0]
jets_mass = [0,0,0,0,0]
weights = []
name_to_index = {"h1j1": 0, "h1j2":1, "h2j1": 2, "h2j2": 3, "j1pt": 4} 
for i, dataset in enumerate(dfs):
    file_out = h5py.File(f"{main_dir}/{list(args.name)[i]}", "w")
    print(f"{main_dir}/{list(args.name)[i]}")
    print(type(file_out))
    create_targets(
        file_out,
        dataset["weight"]
    )
    for key, values in dataset.items():
        cat, subcat, name = convert_name(key)
        if subcat is None:
            weights.append(values)
        elif subcat[-4:] == "JETS":
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
        else: 
            create_inputs(
                file_out,
                cat, subcat, name,
                values
            )
    create_inputs(file_out,"WEIGHTS",None,"weight",weights[0]*weights[1])
    create_jets(file_out, ["pt", "eta", "phi", "mass"],[jets_pt, jets_eta, jets_phi, jets_mass])
## HAndle jets


with open(f"{main_dir}/parameters.txt", "w") as f:
    for key in dfs[0].keys():
        f.write(f"{key}\n")

#create_add_variables(file_out, events)
