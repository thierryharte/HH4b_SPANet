import awkward as ak
import numba
import numpy as np
import pandas as pd
import awkward as ak
import os
import h5py
import vector
import argparse
from multiprocessing import Pool
import functools

vector.register_numba()
vector.register_awkward()


PAD_VALUE = 0

parser = argparse.ArgumentParser(
    description="Convert awkward ntuples in parquet files to h5 files."
)
parser.add_argument("-i", "--input", type=str, required=True, help="Input parquet file")
parser.add_argument("-o", "--output", type=str, default="", help="Output directory")
parser.add_argument(
    "-f",
    "--frac-train",
    type=float,
    default=0.8,
    help="Fraction of events to use for training",
)


args = parser.parse_args()


filename = f"{args.input}"
main_dir = args.output if args.output else os.path.dirname(filename)
os.makedirs(main_dir, exist_ok=True)
df = ak.from_parquet(filename)


def create_groups(file):
    file.create_group("TARGETS/h1")  # higgs 1 -> b1 b2
    file.create_group("TARGETS/h2")  # higgs 2 -> b3 b4
    file.create_group("INPUTS")
    # file.create_group("INPUTS/Source")
    # file.create_group("INPUTS/ht")
    return file


def create_targets(file, particle, jets, filename, max_num_jets):
    indices = ak.local_index(jets)
    higgs_targets = {1: ["b1", "b2"], 2: ["b3", "b4"]}

    for j in [1, 2]:
        if particle == f"h{j}":
            mask = jets.prov == j  # H->b1b2
            indices_prov = ak.fill_none(ak.pad_none(indices[mask], 2), -1)
            print(filename, particle, indices_prov)

            index_b1 = indices_prov[:, 0]
            index_b2 = indices_prov[:, 1]

            index_b1 = ak.where(index_b1 < max_num_jets, index_b1, -1)
            index_b2 = ak.where(index_b2 < max_num_jets, index_b2, -1)

            file.create_dataset(
                f"TARGETS/h{j}/{higgs_targets[j][0]}",
                np.shape(index_b1),
                dtype="int64",
                data=index_b1,
            )
            file.create_dataset(
                f"TARGETS/h{j}/{higgs_targets[j][1]}",
                np.shape(index_b2),
                dtype="int64",
                data=index_b2,
            )


def create_inputs(file, jets, max_num_jets, global_fifth_jet):
    pt_array = ak.to_numpy(
        ak.fill_none(ak.pad_none(jets.pt, max_num_jets, clip=True), PAD_VALUE)
    )
    pt_ds = file.create_dataset(
        "INPUTS/Jet/pt", np.shape(pt_array), dtype="float32", data=pt_array
    )

    mask = ~(pt_array == PAD_VALUE)
    mask_ds = file.create_dataset(
        "INPUTS/Jet/MASK", np.shape(mask), dtype="bool", data=mask
    )

    ptPnetRegNeutrino_array = ak.to_numpy(
        ak.fill_none(
            ak.pad_none(jets.ptPnetRegNeutrino, max_num_jets, clip=True), PAD_VALUE
        )
    )
    ptPnetRegNeutrino_ds = file.create_dataset(
        "INPUTS/Jet/ptPnetRegNeutrino",
        np.shape(ptPnetRegNeutrino_array),
        dtype="float32",
        data=ptPnetRegNeutrino_array,
    )

    phi_array = ak.to_numpy(
        ak.fill_none(ak.pad_none(jets.phi, max_num_jets, clip=True), PAD_VALUE)
    )
    phi_ds = file.create_dataset(
        "INPUTS/Jet/phi", np.shape(phi_array), dtype="float32", data=phi_array
    )

    eta_array = ak.to_numpy(
        ak.fill_none(ak.pad_none(jets.eta, max_num_jets, clip=True), PAD_VALUE)
    )
    eta_ds = file.create_dataset(
        "INPUTS/Jet/eta", np.shape(eta_array), dtype="float32", data=eta_array
    )

    btag = ak.to_numpy(
        ak.fill_none(ak.pad_none(jets.btag, max_num_jets, clip=True), PAD_VALUE)
    )
    btag_ds = file.create_dataset(
        "INPUTS/Jet/btag", np.shape(btag), dtype="float32", data=btag
    )

    mass_array = ak.to_numpy(
        ak.fill_none(ak.pad_none(jets.mass, max_num_jets, clip=True), PAD_VALUE)
    )
    mass_ds = file.create_dataset(
        "INPUTS/Jet/mass", np.shape(mass_array), dtype="float32", data=mass_array
    )

    # create new global variables for the fifth jet (if it exists) otherwise fill with PAD_VALUE
    if global_fifth_jet is not None:
        pt_array_5 = ak.to_numpy(
            ak.fill_none(ak.pad_none(global_fifth_jet.pt, 5, clip=True), PAD_VALUE)[
                :, 4
            ]
        )
        pt_ds_5 = file.create_dataset(
            "INPUTS/FifthJet/pt", np.shape(pt_array_5), dtype="float32", data=pt_array_5
        )

        ptPnetRegNeutrino_array_5 = ak.to_numpy(
            ak.fill_none(
                ak.pad_none(global_fifth_jet.ptPnetRegNeutrino, 5, clip=True), PAD_VALUE
            )[:, 4]
        )
        ptPnetRegNeutrino_ds_5 = file.create_dataset(
            "INPUTS/FifthJet/ptPnetRegNeutrino",
            np.shape(ptPnetRegNeutrino_array_5),
            dtype="float32",
            data=ptPnetRegNeutrino_array_5,
        )

        phi_array_5 = ak.to_numpy(
            ak.fill_none(ak.pad_none(global_fifth_jet.phi, 5, clip=True), PAD_VALUE)[
                :, 4
            ]
        )
        phi_ds_5 = file.create_dataset(
            "INPUTS/FifthJet/phi",
            np.shape(phi_array_5),
            dtype="float32",
            data=phi_array_5,
        )

        eta_array_5 = ak.to_numpy(
            ak.fill_none(ak.pad_none(global_fifth_jet.eta, 5, clip=True), PAD_VALUE)[
                :, 4
            ]
        )
        eta_ds_5 = file.create_dataset(
            "INPUTS/FifthJet/eta",
            np.shape(eta_array_5),
            dtype="float32",
            data=eta_array_5,
        )

        btag_5 = ak.to_numpy(
            ak.fill_none(ak.pad_none(global_fifth_jet.btag, 5, clip=True), PAD_VALUE)[
                :, 4
            ]
        )
        btag_ds_5 = file.create_dataset(
            "INPUTS/FifthJet/btag", np.shape(btag_5), dtype="float32", data=btag_5
        )

        mass_array_5 = ak.to_numpy(
            ak.fill_none(ak.pad_none(global_fifth_jet.mass, 5, clip=True), PAD_VALUE)[
                :, 4
            ]
        )
        mass_ds_5 = file.create_dataset(
            "INPUTS/FifthJet/mass",
            np.shape(mass_array_5),
            dtype="float32",
            data=mass_array_5,
        )


file_dict = {
    0: "output_JetGood_train.h5",
    1: "output_JetGood_test.h5",
    2: "output_JetGoodHiggs_train.h5",
    3: "output_JetGoodHiggs_test.h5",
}


# create the test and train datasets
# and create differnt datasetse with jetGood and jetGoodHiggs

jets_good = df.JetGood
jets_good_higgs = df.JetGoodHiggs

jets_list = []
max_num_jets_list = []
for i, jets_all in enumerate([jets_good, jets_good_higgs]):
    print(f"Creating dataset for {'JetGood' if i == 0 else 'JetGoodHiggs'}")
    n_events = len(jets_all)
    print(f"Number of events: {n_events}")
    idx_train_max = int(np.ceil(n_events * args.frac_train))
    print(f"Number of events for training: {idx_train_max}")
    print(f"Number of events for testing: {n_events - idx_train_max}")
    jets_train = jets_all[:idx_train_max]
    jets_test = jets_all[idx_train_max:]

    for jets in [jets_train, jets_test]:
        jets_list.append(jets)
        max_num_jets_list.append(5 if i == 0 else 4)


def add_info_to_file(input_to_file):
    k, jets = input_to_file
    print(f"Adding info to file {file_dict[k]}")
    file_out = h5py.File(f"{main_dir}/{file_dict[k]}", "w")
    file_out = create_groups(file_out)
    print("max_num_jets", max_num_jets_list[k])
    global_fifth_jet = None
    if file_dict[k] == "output_JetGoodHiggs_train.h5":
        global_fifth_jet = jets_list[0]
    elif file_dict[k] == "output_JetGoodHiggs_test.h5":
        global_fifth_jet = jets_list[1]
    create_inputs(file_out, jets, max_num_jets_list[k], global_fifth_jet)
    create_targets(file_out, "h1", jets, file_dict[k], max_num_jets_list[k])
    create_targets(file_out, "h2", jets, file_dict[k], max_num_jets_list[k])
    print("Completed file ", file_dict[k])
    file_out.close()


with Pool(4) as p:
    p.map(add_info_to_file, enumerate(jets_list))
