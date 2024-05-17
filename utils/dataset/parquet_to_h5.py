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


PAD_VALUE = 9999

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
parser.add_argument(
    "-n",
    "--num-jets",
    type=int,
    default=5,
    help="Number of JetGood to use in the dataset",
)
parser.add_argument(
    "-s",
    "--no-shuffle",
    action="store_true",
    default=False,
    help="Do not shuffle the dataset",
)

args = parser.parse_args()


btag_wp = [0.0499, 0.2605, 0.6915]

xsec_qcd_tot = (
    1.968e06
    + 1.000e05
    + 1.337e04
    + 3.191e03
    + 8.997e02
    + 3.695e02
    + 1.272e02
    + 2.514e01
)


def create_groups(file):
    file.create_group("TARGETS/h1")  # higgs 1 -> b1 b2
    file.create_group("TARGETS/h2")  # higgs 2 -> b3 b4
    file.create_group("INPUTS")

    return file


def create_targets(file, particle, jets, filename, max_num_jets):
    indices = ak.local_index(jets)
    higgs_targets = {1: ["b1", "b2"], 2: ["b3", "b4"]}

    for j in [1, 2]:
        if particle == f"h{j}":
            if ak.all(jets.prov == -1):
                index_b1 = ak.full_like(jets.pt[:, 0], 0)
                index_b2 = ak.full_like(jets.pt[:, 0], 0)
                print(filename, particle, index_b1, index_b2)
            else:
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


def create_inputs(file, jets, max_num_jets, global_fifth_jet, events):
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
    # compute the cos and sin of phi
    cos_phi = ak.to_numpy(
        ak.fill_none(ak.pad_none(np.cos(jets.phi), max_num_jets, clip=True), PAD_VALUE)
    )

    cos_phi_ds = file.create_dataset(
        "INPUTS/Jet/cosPhi", np.shape(cos_phi), dtype="float32", data=cos_phi
    )

    sin_phi = ak.to_numpy(
        ak.fill_none(ak.pad_none(np.sin(jets.phi), max_num_jets, clip=True), PAD_VALUE)
    )
    sin_phi_ds = file.create_dataset(
        "INPUTS/Jet/sinPhi", np.shape(sin_phi), dtype="float32", data=sin_phi
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

    btag_wp_array = ak.to_numpy(
        ak.fill_none(
            ak.pad_none(
                ak.where(btag > btag_wp[0], 1, 0)
                + ak.where(btag > btag_wp[1], 1, 0)
                + ak.where(btag > btag_wp[2], 1, 0),
                max_num_jets,
                clip=True,
            ),
            PAD_VALUE,
        )
    )
    btag_wp_ds = file.create_dataset(
        "INPUTS/Jet/btag_wp_bit",
        np.shape(btag_wp_array),
        dtype="int32",
        data=btag_wp_array,
    )

    mass_array = ak.to_numpy(
        ak.fill_none(ak.pad_none(jets.mass, max_num_jets, clip=True), PAD_VALUE)
    )
    mass_ds = file.create_dataset(
        "INPUTS/Jet/mass", np.shape(mass_array), dtype="float32", data=mass_array
    )

    kl_array = ak.to_numpy(events.kl)
    kl_ds = file.create_dataset(
        "INPUTS/Event/kl", np.shape(kl_array), dtype="float32", data=kl_array
    )

    weight_array = ak.to_numpy(
        events.weight / (xsec_qcd_tot if "QCD" in args.input else 1)
    )
    weight_ds = file.create_dataset(
        "INPUTS/Event/weight",
        np.shape(weight_array),
        dtype="float32",
        data=weight_array,
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
                ak.pad_none(global_fifth_jet.ptPnetRegNeutrino, 5, clip=True),
                PAD_VALUE,
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

        cos_phi_5 = ak.to_numpy(
            ak.fill_none(
                ak.pad_none(np.cos(global_fifth_jet.phi), 5, clip=True),
                PAD_VALUE,
            )[:, 4]
        )
        cos_phi_ds_5 = file.create_dataset(
            "INPUTS/FifthJet/cosPhi",
            np.shape(cos_phi_5),
            dtype="float32",
            data=cos_phi_5,
        )

        sin_phi_5 = ak.to_numpy(
            ak.fill_none(
                ak.pad_none(np.sin(global_fifth_jet.phi), 5, clip=True),
                PAD_VALUE,
            )[:, 4]
        )
        sin_phi_ds_5 = file.create_dataset(
            "INPUTS/FifthJet/sinPhi",
            np.shape(sin_phi_5),
            dtype="float32",
            data=sin_phi_5,
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

        btag_wp_array_5 = ak.to_numpy(
            ak.fill_none(
                ak.pad_none(
                    ak.where(global_fifth_jet.btag > btag_wp[0], 1, 0)
                    + ak.where(global_fifth_jet.btag > btag_wp[1], 1, 0)
                    + ak.where(global_fifth_jet.btag > btag_wp[2], 1, 0),
                    5,
                    clip=True,
                ),
                PAD_VALUE,
            )[:, 4]
        )
        btag_wp_ds_5 = file.create_dataset(
            "INPUTS/FifthJet/btag_wp_bit",
            np.shape(btag_wp_array_5),
            dtype="int32",
            data=btag_wp_array_5,
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
    create_inputs(
        file_out, jets, max_num_jets_list[k], global_fifth_jet, events_list[k]
    )
    create_targets(file_out, "h1", jets, file_dict[k], max_num_jets_list[k])
    create_targets(file_out, "h2", jets, file_dict[k], max_num_jets_list[k])
    print("Completed file ", file_dict[k])
    file_out.close()


filename = f"{args.input}"
main_dir = args.output if args.output else os.path.dirname(filename)
os.makedirs(main_dir, exist_ok=True)
df = ak.from_parquet(filename)


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
events_list = []
max_num_jets_list = []
n_events = len(jets_good)
idx = np.random.RandomState(seed=42).permutation(n_events)
for i, jets_all in enumerate([jets_good, jets_good_higgs]):
    events_all = df.event
    print(f"Creating dataset for {'JetGood' if i == 0 else 'JetGoodHiggs'}")
    print(f"Number of events: {n_events}")
    idx_train_max = int(np.ceil(n_events * args.frac_train))
    print(f"Number of events for training: {idx_train_max}")
    print(f"Number of events for testing: {n_events - idx_train_max}")

    if not args.no_shuffle:
        jets_all = jets_all[idx]
        events_all = events_all[idx]

    jets_train = jets_all[:idx_train_max]
    jets_test = jets_all[idx_train_max:]
    events_train = events_all[:idx_train_max]
    events_test = events_all[idx_train_max:]

    for jets, ev in zip([jets_train, jets_test], [events_train, events_test]):
        jets_list.append(jets)
        events_list.append(ev)
        max_num_jets_list.append(args.num_jets if i == 0 else 4)


with Pool(4) as p:
    p.map(add_info_to_file, enumerate(jets_list))
