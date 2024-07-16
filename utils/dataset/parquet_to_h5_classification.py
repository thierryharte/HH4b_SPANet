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
import sys
vector.register_numba()
vector.register_awkward()
import psutil
from rich.progress import track

num_threads = 1
# os.environ["OMP_NUM_THREADS"] = f"{num_threads}"
# print("THREADS", os.environ.get("OMP_NUM_THREADS"), flush=True)
# os.environ["MKL_NUM_THREADS"] = str(num_threads)
# os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)

import onnxruntime
sess_opts = onnxruntime.SessionOptions()
sess_opts.execution_mode  = onnxruntime.ExecutionMode.ORT_PARALLEL
sess_opts.inter_op_num_threads = num_threads #parallelize the call
sess_opts.intra_op_num_threads = num_threads 

print("THREADS", sess_opts.inter_op_num_threads, sess_opts.intra_op_num_threads, flush=True)

from prediction_selection import *


# pass arguments to rum the code
# to use them do: args.name of the arg
# the input file will be a parquet file that we access by doing filename = f"{args.input}"
# df = ak.from_parquet(filename) : this is an awkard lisr


PAD_VALUE = 9999
NUMBER_JETS_SPANET = 5

parser = argparse.ArgumentParser(
    description="Convert awkward ntuples in parquet files to h5 files."
)
# get multiple files
parser.add_argument(
    "-i", "--input", required=True, help="Input parquet file", nargs="+"
)
parser.add_argument("-o", "--output", type=str, default="", help="Output directory")
parser.add_argument(
    "-t", "--spanet-training", type=str, default="/work/mmalucch/out_hh4b_5jets_ATLAS_ptreg_c0_lr1e4_wp0_noklininp_oc_300e_kl3p5.onnx", help="Spanet training"
)


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
parser.add_argument(
    "-c",
    "--classification",
    action="store_true",
    default=False,
    help="Dataset is for classification",
)

args = parser.parse_args()

# low, medium and tight WP
btag_wp = [0.0499, 0.2605, 0.6915]

# load the model
session = onnxruntime.InferenceSession(
    args.spanet_training,
    providers=onnxruntime.get_available_providers(),
    sess_opts=sess_opts,
)
# name of the inputs and outputs of the model
input_name = [input.name for input in session.get_inputs()]
output_name = [output.name for output in session.get_outputs()]

input_shape = [input.shape for input in session.get_inputs()]
output_shape = [output.shape for output in session.get_outputs()]

# set the names of the groups in the h5 out file (used in add_info_to_file)


def create_groups(file):
    file.create_group("TARGETS/h1")  # higgs 1 -> b1 b2
    file.create_group("TARGETS/h2")  # higgs 2 -> b3 b4
    file.create_group("INPUTS")

    return file


# four vector for only fully matched events
def jet_four_vector_fully_matched(jet):
    jet_prov_unflat = ak.unflatten(jet.prov, len(jet))
    jet_pt_unflat = ak.unflatten(jet.pt, len(jet))
    jet_eta_unflat = ak.unflatten(jet.eta, len(jet))
    jet_phi_unflat = ak.unflatten(jet.phi, len(jet))
    jet_mass_unflat = ak.unflatten(jet.mass, len(jet))
    jet_btag_unflat = ak.unflatten(jet.btag, len(jet))

    count_ones = ak.sum(jet_prov_unflat == 1, axis=1)
    count_twos = ak.sum(jet_prov_unflat == 2, axis=1)

    mask_fully_matched = (count_ones == 2) & (count_twos == 2)

    jet_fully_matched = ak.zip(
        {
            "pt": jet_pt_unflat[mask_fully_matched],
            "eta": jet_eta_unflat[mask_fully_matched],
            "phi": jet_phi_unflat[mask_fully_matched],
            "mass": jet_mass_unflat[mask_fully_matched],
            "btag": jet_btag_unflat[mask_fully_matched],
            "prov": jet_prov_unflat[mask_fully_matched],
        },
        with_name="Momentum4D",
    )
    print(jet_fully_matched)
    return jet_fully_matched


def get_awkward_array_shape(arr):
    if len(arr) == 0:
        return (0,)
    shape = []
    while isinstance(arr, ak.Array):
        shape.append(len(arr))
        arr = arr[0]
    return tuple(shape)


def reconstruct_higgs(jet_collection, idx_collection):
    # print("jet coll", jet_collection)
    # print("indx coll", idx_collection)

    # print("len(idx_collection)", len(idx_collection))
    # print("len jet coll indx", len(jet_collection[np.arange(len(idx_collection)), idx_collection[:, 0, 0]]))
    # print("len jet coll indx", np.arange(len(idx_collection)))

    # sum_j=jet_collection[:,0]+ jet_collection[:,1]

    # print(sum_j.pt)

    # jet_collection[np.arange(len(idx_collection)), idx_collection[:, 0, 0]]
    #  the first part is the event, and the second theindeex of the jeet in the event

    # print("event 8", jet_collection[7,4].pt)
    # print("event 8", len(jet_collection[7,:].pt))

    idx_1 = idx_collection[:, 0, 0]
    idx_2 = idx_collection[:, 0, 1]
    idx_3 = idx_collection[:, 1, 0]
    idx_4 = idx_collection[:, 1, 1]
    lenght = np.arange(len(idx_collection))

    higgs_1 = (
        jet_collection[np.arange(len(idx_collection)), idx_collection[:, 0, 0]]
        + jet_collection[np.arange(len(idx_collection)), idx_collection[:, 0, 1]]
    )
    print("h1", higgs_1)
    higgs_1_unflat = ak.unflatten(higgs_1, 1)
    print("h1_unflat", higgs_1_unflat)

    higgs_2 = (
        jet_collection[np.arange(len(idx_collection)), idx_collection[:, 1, 0]]
        + jet_collection[np.arange(len(idx_collection)), idx_collection[:, 1, 1]]
    )
    higgs_2_unflat = ak.unflatten(higgs_1, 1)
    print("h2", higgs_2)
    print("h1_unflat", higgs_2_unflat)

    higgs_leading_index = ak.where(higgs_1.pt > higgs_2.pt, 0, 1)
    print(higgs_leading_index)

    higgs_lead = ak.where(higgs_leading_index == 0, higgs_1, higgs_2)
    higgs_sub = ak.where(higgs_leading_index == 0, higgs_2, higgs_1)

    higgs_leading_index_expanded = higgs_leading_index[
        :, np.newaxis, np.newaxis
    ] * np.ones((2, 2))
    print("higgs_leading_index_expanded", higgs_leading_index_expanded)
    idx_ordered = ak.where(
        higgs_leading_index_expanded == 0, idx_collection, idx_collection[:, ::-1]
    )

    print("idx_ordered", idx_ordered)

    return higgs_lead, higgs_sub, idx_ordered
    # return sum_j, higgs_1, idx_ordered


def create_targets(file, particle, jets, filename, max_num_jets):
    indices = ak.local_index(jets)
    higgs_targets = {1: ["b1", "b2"], 2: ["b3", "b4"]}

    for j in [1, 2]:
        if particle == f"h{j}":
            if ak.all(jets.prov == -1):
                if args.classification:
                    index_b1 = ak.full_like(jets.pt[:, 0], 0 + (j - 1) * 2)
                    index_b2 = ak.full_like(jets.pt[:, 0], 1 + (j - 1) * 2)
                else:
                    index_b1 = ak.full_like(jets.pt[:, 0], 0)
                    index_b2 = ak.full_like(jets.pt[:, 0], 0)
                # print(filename, particle, index_b1, index_b2)
            else:
                mask = jets.prov == j  # H->b1b2
                indices_prov = ak.fill_none(ak.pad_none(indices[mask], 2), -1)
                # print(filename, particle, indices_prov)

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


def jet_four_vector(jets_list):

    jet_4vector = []
    # k, jets_list = input_to_file
    jet_prov_unflat = jets_list.prov
    jet_pt_unflat = jets_list.pt
    jet_eta_unflat = jets_list.eta
    jet_phi_unflat = jets_list.phi
    jet_mass_unflat = jets_list.mass
    jet_btag_unflat = jets_list.btag

    jet = ak.zip(
        {
            "pt": jet_pt_unflat,
            "eta": jet_eta_unflat,
            "phi": jet_phi_unflat,
            "mass": jet_mass_unflat,
            "btag": jet_btag_unflat,
            "prov": jet_prov_unflat,
        },
        with_name="Momentum4D",
    )
    print("jet_4vector 1", jet, flush=True)
    # jet_4vector.append(jet)
    # print("jet_4vector 2",jet_4vector)
    return jet


def get_pairing_information(jets, file):
    ptPnetRegNeutrino_array = np.log(
        ak.to_numpy(
            ak.fill_none(
                ak.pad_none(jets.pt, NUMBER_JETS_SPANET, clip=True),
                PAD_VALUE,
            )
        )
        + 1
    )
    ptPnetRegNeutrino = ptPnetRegNeutrino_array.astype("float32")

    eta_array = ak.to_numpy(
        ak.fill_none(ak.pad_none(jets.eta, NUMBER_JETS_SPANET, clip=True), PAD_VALUE)
    )
    eta = eta_array.astype("float32")

    phi_array = ak.to_numpy(
        ak.fill_none(ak.pad_none(jets.phi, NUMBER_JETS_SPANET, clip=True), PAD_VALUE)
    )

    phi = phi_array.astype("float32")

    btag_array = ak.to_numpy(
        ak.fill_none(ak.pad_none(jets.btag, NUMBER_JETS_SPANET, clip=True), PAD_VALUE)
    )

    btag = btag_array.astype("float32")

    mask = ~(phi_array == PAD_VALUE)
    # we have the inputs in which we will evaluate our model
    inputs = np.stack((ptPnetRegNeutrino, eta, phi, btag), axis=-1)
    # inputs_complete = {input_name[0]: inputs, input_name[1]: mask}
    # print(inputs.shape)
    # print(mask.shape)

    # # evalutaion of the model on these inputs
    batch_size=6000
    nbatches=int(len(jets)/batch_size)
    print("nbatches", nbatches)
    total_outputs=[]
    total_prob_h1=[]
    total_prob_h2=[]
    # for i in track(range(nbatches), "Inference"):
    for i in range(nbatches):
        #for i in range(nbatches):
        # if i> 10: break
        #print(f"EVAL {i}")
        start = i*batch_size
        if i < (nbatches-1):
            stop = start+batch_size
            # print("stop 1", stop)
        else:
            stop = len(jets)
            # print("stop 2", stop)
        # print("inputs[: , start:stop]", inputs[start:stop])
        # print("shape inputs[: , start:stop]", inputs[start:stop].shape)
        # print("mask[start:stop]", mask[start:stop])
        # print("shape mask[start:stop]", mask[start:stop].shape)
        inputs_complete = {input_name[0]: inputs[start:stop], input_name[1]: mask[start:stop]}
        outputs = (session.run( output_name, inputs_complete))
        prob_h1= outputs[0]
        # print("prob_h1 type", type(prob_h1))
        prob_h2= outputs[1]
        # print("outputs", (outputs))
        # for j in outputs[0]:
        #     if j.shape != (5,5):
        #         print("outputs",j.shape)
        # print("len outputs",len(outputs))
        total_prob_h1.append(prob_h1)
        total_prob_h2.append(prob_h2)
        #print(outputs)
    
        
    # print(total_prob_h1)
    # print("total outputs", total_outputs)
    print("len total proba h1", len(total_prob_h1), flush= True)
    print("len total proba h2", len(total_prob_h2), flush= True)
    
    
    proba_h1=np.concatenate(total_prob_h1, axis=0)
    proba_h2=np.concatenate(total_prob_h2, axis=0)
    
    print("prob h1",proba_h1.shape)
    print("prob h2",proba_h2.shape)
    # outputs = session.run(output_name, inputs_complete)
    print("file", file)

    # extract the best jet assignment from
    # the predicted probabilities
    assignment_probability = np.stack((proba_h1, proba_h2), axis=0)
    # print("\nassignment_probability", assignment_probability)
    # swap axis

    # print("matrix proba",outputs[0])
    predictions_best = np.swapaxes(extract_predictions(assignment_probability), 0, 1)

    # get the probabilities of the best jet assignment
    num_events = assignment_probability.shape[1]
    print("num_events", num_events)
    range_num_events = np.arange(num_events)
    best_pairing_probabilities = np.ndarray((2, num_events))
    for i in range(2):
        best_pairing_probabilities[i] = assignment_probability[
            i,
            range_num_events,
            predictions_best[:, i, 0],
            predictions_best[:, i, 1],
        ]
    best_pairing_probabilities_sum = np.sum(best_pairing_probabilities, axis=0)
    print("\nbest_pairing_probabilities_sum", best_pairing_probabilities_sum)
    print("\nbest_pairing_probabilities_sum", len(best_pairing_probabilities_sum))

    # set to zero the probabilities of the best jet assignment, the symmetrization and the same jet assignment on the other target
    for j in range(2):
        for k in range(2):
            assignment_probability[
                j,
                range_num_events,
                predictions_best[:, j, k],
                predictions_best[:, j, 1 - k],
            ] = 0
            assignment_probability[
                1 - j,
                range_num_events,
                predictions_best[:, j, k],
                predictions_best[:, j, 1 - k],
            ] = 0

    # print("\nassignment_probability new", assignment_probability)
    # extract the second best jet assignment from
    # the predicted probabilities
    # swap axis
    predictions_second_best = np.swapaxes(
        extract_predictions(assignment_probability), 0, 1
    )

    # get the probabilities of the second best jet assignment
    second_best_pairing_probabilities = np.ndarray((2, num_events))
    for i in range(2):
        second_best_pairing_probabilities[i] = assignment_probability[
            i,
            range_num_events,
            predictions_second_best[:, i, 0],
            predictions_second_best[:, i, 1],
        ]
    second_best_pairing_probabilities_sum = np.sum(
        second_best_pairing_probabilities, axis=0
    )
    # print(
    #     "\nsecond_best_pairing_probabilities_sum",
    #     second_best_pairing_probabilities_sum,
    # )

    difference = best_pairing_probabilities_sum - second_best_pairing_probabilities_sum

    return (
        difference,
        predictions_best,
        best_pairing_probabilities_sum,
        second_best_pairing_probabilities_sum,
    )


def create_inputs(file, jets, jet_4vector, max_num_jets, global_fifth_jet, events):

    # pt_array = ak.to_numpy(
    #     ak.fill_none(ak.pad_none(jets.pt, max_num_jets, clip=True), PAD_VALUE)
    # )
    # pt_ds = file.create_dataset(
    #     "INPUTS/Jet/pt", np.shape(pt_array), dtype="float32", data=pt_array
    # )

    # first you create the variable and then the dataset
    ptPnetRegNeutrino_array = ak.to_numpy(
        ak.fill_none(
            ak.pad_none(jets.pt, max_num_jets, clip=True), PAD_VALUE
        )
    )
    ptPnetRegNeutrino_ds = file.create_dataset(
        "INPUTS/Jet/ptPnetRegNeutrino",
        np.shape(ptPnetRegNeutrino_array),
        dtype="float32",
        data=ptPnetRegNeutrino_array,
    )

    mask = ~(ptPnetRegNeutrino_array == PAD_VALUE)
    mask_ds = file.create_dataset(
        "INPUTS/Jet/MASK", np.shape(mask), dtype="bool", data=mask
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
        dtype="int64",
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

    if args.classification:
        weight_array = ak.to_numpy(events.weight)
        weight_ds = file.create_dataset(
            "WEIGHTS/weight",
            np.shape(weight_array),
            dtype="float32",
            data=weight_array,
        )

        sb_array = ak.to_numpy(events.sb)
        sb_ds = file.create_dataset(
            "CLASSIFICATIONS/EVENT/signal",
            np.shape(sb_array),
            dtype="int64",
            data=sb_array,
        )

        # using the 4 vector function
        pt_4vector = jet_4vector.pt
        print("4vector pt", pt_4vector)

        ht_array = ak.sum(jet_4vector.pt, axis=1)
        print("ht_array", ht_array)
        ht_ds = file.create_dataset(
            "INPUTS/Event/HT",
            np.shape(ht_array),
            dtype="int64",
            data=ht_array,
        )
        
        #TODO: check  the shape

        o, JetGood2 = ak.unzip(
            ak.cartesian(
                [jet_4vector, jet_4vector],
                nested=True,
            )
        )

        # print(JetGood2)
        dR = jet_4vector.deltaR(JetGood2)
        print("dr", dR)
        # remove dR between the same jets
        dR = ak.mask(dR, dR > 0)
        print("dr post mask", dR)
        # flatten the last 2 dimension of the dR array  to get an array for each event
        dR = ak.flatten(dR, axis=2)
        print("dr post flat", dR, flush= True)
        dR_min = ak.min(dR, axis=1)

        print("dR_min", dR_min, flush=True)

        dR_max = ak.max(dR, axis=1)

        print("dR_max", dR_max)

        dR_min_ds = file.create_dataset(
            "INPUTS/Event/dR_min",
            np.shape(dR_min),
            dtype="float32",
            data=dR_min,
        )

        dR_max_ds = file.create_dataset(
            "INPUTS/Event/dR_max",
            np.shape(dR_max),
            dtype="float32",
            data=dR_max,
        )

        pt = jet_4vector.pt
        eta = jet_4vector.eta
        phi = jet_4vector.phi
        btag = jet_4vector.btag

        (
            difference,
            predictions_best,
            best_pairing_probabilities_sum,
            second_best_pairing_probabilities_sum,
        ) = get_pairing_information(jets, file)

        difference = ak.to_numpy(difference)

        difference_ds = file.create_dataset(
            "INPUTS/Event/Probability_difference",
            np.shape(difference),
            dtype="float32",
            data=difference,
        )

        HiggsLeading, HiggsSubLeading, pairing_predictions_ordered = reconstruct_higgs(
            jet_4vector, predictions_best
        )

        # print("higgs_1", higgs_1)
        # print("higgs_2", higgs_2)
        print("Higgs Leading", HiggsLeading)

        HiggsLeading_pt = HiggsLeading.pt
        HiggsLeading_pt_ds = file.create_dataset(
            "INPUTS/HiggsLeading/pt",
            np.shape(HiggsLeading_pt),
            dtype="float32",
            data=HiggsLeading_pt,
        )

        HiggsLeading_eta = HiggsLeading.eta
        HiggsLeading_eta_ds = file.create_dataset(
            "INPUTS/HiggsLeading/eta",
            np.shape(HiggsLeading_eta),
            dtype="float32",
            data=HiggsLeading_eta,
        )

        HiggsLeading_phi = HiggsLeading.phi
        HiggsLeading_phi_ds = file.create_dataset(
            "INPUTS/HiggsLeading/phi",
            np.shape(HiggsLeading_phi),
            dtype="float32",
            data=HiggsLeading_phi,
        )

        HiggsLeading_mass = HiggsLeading.mass
        HiggsLeading_mass_ds = file.create_dataset(
            "INPUTS/HiggsLeading/mass",
            np.shape(HiggsLeading_mass),
            dtype="float32",
            data=HiggsLeading_mass,
        )

        HiggsLeading_costheta = abs(np.cos(HiggsLeading.theta))
        HiggsLeading_costheta_ds = file.create_dataset(
            "INPUTS/HiggsLeading/cos_theta",
            np.shape(HiggsLeading_costheta),
            dtype="float32",
            data=HiggsLeading_costheta,
        )

        # self.events["HiggsSubLeading"] = ak.with_field(
        #         self.events.HiggsSubLeading,
        #         self.events.JetGood[
        #             np.arange(len(pairing_predictions_ordered)),
        #             pairing_predictions_ordered[:, 1, 0],
        #         ].delta_r(
        #             self.events.JetGood[
        #                 np.arange(len(pairing_predictions_ordered)),
        #                 pairing_predictions_ordered[:, 1, 1],
        #             ]
        #         ),

        # mask_prov_1= jet_4vector.prov==1
        # jet_fully_matched=jet_4vector[mask_prov_1]

        indx_1_0 = pairing_predictions_ordered[:, 0, 0]
        indx_1_1 = pairing_predictions_ordered[:, 0, 1]

        jet_h1_0 = jet_4vector[np.arange(len(predictions_best)), indx_1_0]
        jet_h1_1 = jet_4vector[np.arange(len(predictions_best)), indx_1_1]

        print("jet 1 ", jet_h1_0)
        print("jet 2 ", jet_h1_1)

        HiggsLeading_dR = jet_h1_0.deltaR(jet_h1_1)
        print("HiggsLeading_dR", HiggsLeading_dR)

        # HiggsLeading_dR= jet_4vector[:,pairing_predictions_ordered[:, 0, 0]].deltaR(jet_4vector[:,pairing_predictions_ordered[:, 0, 1]])
        HiggsLeading_dR_ds = file.create_dataset(
            "INPUTS/HiggsLeading/dR",
            np.shape(HiggsLeading_dR),
            dtype="float32",
            data=HiggsLeading_dR,
        )

        HiggsSubLeading_pt = HiggsSubLeading.pt
        HiggsSubLeading_pt_ds = file.create_dataset(
            "INPUTS/HiggsSubLeading/pt",
            np.shape(HiggsSubLeading_pt),
            dtype="float32",
            data=HiggsSubLeading_pt,
        )

        HiggsSubLeading_eta = HiggsSubLeading.eta
        HiggsSubLeading_eta_ds = file.create_dataset(
            "INPUTS/HiggsSubLeading/eta",
            np.shape(HiggsSubLeading_eta),
            dtype="float32",
            data=HiggsSubLeading_eta,
        )

        HiggsSubLeading_phi = HiggsSubLeading.phi
        HiggsSubLeading_phi_ds = file.create_dataset(
            "INPUTS/HiggsSubLeading/phi",
            np.shape(HiggsSubLeading_phi),
            dtype="float32",
            data=HiggsSubLeading_phi,
        )

        HiggsSubLeading_mass = HiggsSubLeading.mass
        HiggsSubLeading_mass_ds = file.create_dataset(
            "INPUTS/HiggsSubLeading/mass",
            np.shape(HiggsSubLeading_mass),
            dtype="float32",
            data=HiggsSubLeading_mass,
        )

        HiggsSubLeading_costheta = abs(np.cos(HiggsSubLeading.theta))
        HiggsSubLeading_costheta_ds = file.create_dataset(
            "INPUTS/HiggsSubLeading/cos_theta",
            np.shape(HiggsSubLeading_costheta),
            dtype="float32",
            data=HiggsSubLeading_costheta,
        )

        indx_2_0 = pairing_predictions_ordered[:, 1, 0]
        indx_2_1 = pairing_predictions_ordered[:, 1, 1]

        jet_h2_0 = jet_4vector[np.arange(len(predictions_best)), indx_2_0]
        jet_h2_1 = jet_4vector[np.arange(len(predictions_best)), indx_2_1]

        print("jet 1 ", jet_h2_0)
        print("jet 2 ", jet_h2_1)

        HiggsSubLeading_dR = jet_h2_0.deltaR(jet_h2_1)
        print("HiggsSubLeading_dR", HiggsSubLeading_dR)

        # HiggsSubLeading_dR= jet_4vector[:,pairing_predictions_ordered[:, 1, 0]].deltaR(jet_4vector[:,pairing_predictions_ordered[:, 1, 1]])
        HiggsSubLeading_dR_ds = file.create_dataset(
            "INPUTS/HiggsSubLeading/dR",
            np.shape(HiggsSubLeading_dR),
            dtype="float32",
            data=HiggsSubLeading_dR,
        )

        HH = HiggsLeading + HiggsSubLeading

        HH_pt = HH.pt
        HH_pt_ds = file.create_dataset(
            "INPUTS/HH/pt",
            np.shape(HH_pt),
            dtype="float32",
            data=HH_pt,
        )

        HH_eta = HH.eta
        HH_eta_ds = file.create_dataset(
            "INPUTS/HH/eta",
            np.shape(HH_eta),
            dtype="float32",
            data=HH_eta,
        )

        HH_phi = HH.phi
        HH_phi_ds = file.create_dataset(
            "INPUTS/HH/phi",
            np.shape(HH_phi),
            dtype="float32",
            data=HH_phi,
        )

        HH_mass = HH.mass
        HH_mass_ds = file.create_dataset(
            "INPUTS/HH/mass",
            np.shape(HH_mass),
            dtype="float32",
            data=HH_mass,
        )

        HH_dR = HiggsLeading.deltaR(HiggsSubLeading)
        HH_dR_ds = file.create_dataset(
            "INPUTS/HH/dR",
            np.shape(HH_dR),
            dtype="float32",
            data=HH_dR,
        )

        HH_cos = abs(np.cos(HH.theta))
        HH_cos_ds = file.create_dataset(
            "INPUTS/HH/cos_theta_star",
            np.shape(HH_cos),
            dtype="float32",
            data=HH_cos,
        )

        HH_dEta = HiggsLeading.eta - HiggsSubLeading.eta
        HH_dEta_ds = file.create_dataset(
            "INPUTS/HH/dEta",
            np.shape(HH_dEta),
            dtype="float32",
            data=HH_dEta,
        )

        HH_dPhi = HiggsLeading.deltaphi(HiggsSubLeading)
        HH_dPhi_ds = file.create_dataset(
            "INPUTS/HH/dPhi",
            np.shape(HH_dPhi),
            dtype="float32",
            data=HH_dPhi,
        )

        # predictions_best= ak.to_numpy(predictions_best)
        best_pairing_probabilities_sum = ak.to_numpy(best_pairing_probabilities_sum)
        second_best_pairing_probabilities_sum = ak.to_numpy(
            second_best_pairing_probabilities_sum
        )

        # predictions_ds= file.create_dataset(
        #     "INPUTS/Event/Predictions_best",
        #     np.shape(predictions_best),
        #     dtype="float32",
        #     data=predictions_best,
        # )

        best_pairing_probabilities_sum_ds = file.create_dataset(
            "INPUTS/Event/Best_pairing_probabilities_sum",
            np.shape(best_pairing_probabilities_sum),
            dtype="float32",
            data=best_pairing_probabilities_sum,
        )

        second_best_pairing_probabilities_sum_ds = file.create_dataset(
            "INPUTS/Event/Second_best_pairing_probabilities",
            np.shape(second_best_pairing_probabilities_sum),
            dtype="float32",
            data=second_best_pairing_probabilities_sum,
        )
        
        r_HH= np.sqrt((HiggsLeading_mass -125)**2 +(HiggsSubLeading_mass -120)**2)
        mask_sr= r_HH < 30

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
                ak.pad_none(global_fifth_jet.pt, 5, clip=True),
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
            dtype="int64",
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
    # since the input an enumerate object, k corresponds to the index and jets to the actual list
    k, jets = input_to_file
    print(f"Adding info to file {file_dict[k]}")
    file_out = h5py.File(f"{main_dir}/{file_dict[k]}", "w")
    file_out = create_groups(file_out)
    print("max_num_jets", max_num_jets_list[k])
    global_fifth_jet = None
    if file_dict[k] == "output_JetGoodHiggs_train.h5":
        # jets_list[0] is the list of jets for jet good ( so contains a fifth jet) for the train dataset
        global_fifth_jet = jets_list[0]
    elif file_dict[k] == "output_JetGoodHiggs_test.h5":
        # jets_list[1] is the list of jets for jet good ( so contains a fifth jet) for the test dataset
        global_fifth_jet = jets_list[1]

    jet_4vector = jet_four_vector(jets)

    # evaluate the model for the events
    # predictions_best,  best_pairing_probabilities_sum , second_best_pairing_probabilities_sum= get_pairing_information(jets,file_out)

    # print("best", predictions_best)
    # print("bes sum", best_pairing_probabilities_sum)
    # print("second best", second_best_pairing_probabilities_sum)

    create_inputs(
        file_out,
        jets,
        jet_4vector,
        max_num_jets_list[k],
        global_fifth_jet,
        events_list[k],
    )
    create_targets(file_out, "h1", jets, file_dict[k], max_num_jets_list[k])
    create_targets(file_out, "h2", jets, file_dict[k], max_num_jets_list[k])
    print("Completed file ", file_dict[k])
    file_out.close()


main_dir = args.output if args.output else os.path.dirname(args.input[0])
os.makedirs(main_dir, exist_ok=True)
dfs = []
for filename in list(args.input):
    dfs.append(ak.from_parquet(filename))
    print(dfs[-1].event.sb)
    

df = ak.concatenate(dfs)
print("df", df)
print("df shape", df.event.sb)
# df= ['JetGood', 'JetGoodHiggs', 'JetGoodHiggsMatched', 'JetGoodMatched', 'event']


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

# print("df=", df.fields)
# print("jetgood",jets_good.fields)
# print("jet good higgs",jets_good_higgs.type)

jets_list = []
events_list = []
max_num_jets_list = []
n_events = len(jets_good)

# Randomly permute a sequence, or return a permuted range. in this case we randomly permute the number of events
# I think we fix the seed so that the permutation is the same in jet_good and jet_good_higgs but not sure
idx = np.random.RandomState(seed=42).permutation(n_events)
for i, jets_all in enumerate([jets_good, jets_good_higgs]):
    events_all = df.event
    print("events_all",events_all)
    print("jets_all",jets_all)
    print(f"Creating dataset for {'JetGood' if i == 0 else 'JetGoodHiggs'}")
    print(f"Number of events: {n_events}")
    # The ceil of the scalar x is the smallest integer i, such that i >= x
    idx_train_max = int(np.ceil(n_events * args.frac_train))
    print(f"Number of events for training: {idx_train_max}")
    print(f"Number of events for testing: {n_events - idx_train_max}")

    # i believe here we shuffle the indices
    if not args.no_shuffle:
        jets_all = jets_all[idx]
        events_all = events_all[idx]

    jets_train = jets_all[:idx_train_max]
    jets_test = jets_all[idx_train_max:]
    events_train = events_all[:idx_train_max]
    events_test = events_all[idx_train_max:]

    for jets, ev in zip([jets_train, jets_test], [events_train, events_test]):
        # list of shuffled jets i believe in the end there are 4 bc there is test and train for jet good and jet good higgs
        jets_list.append(jets)
        # lkist of the number of events
        events_list.append(ev)
        max_num_jets_list.append(args.num_jets if i == 0 else 4)

# no estoy segura porque hay un 4 pero este pool y map permite de ejectutar en paralelo la funcion add info a las distintas jet_lists
# with Pool(4) as p:
#     p.map(add_info_to_file, enumerate(jets_list))


# with Pool(4) as p:
#     p.map(add_info_to_file, [(3, jets_list[3])])

# jet_list=[]
# for i in range (1):
#     jet_list.append(jets_list[i])

# # jet_list.append(jets_list[1])

# print("jet_list", jet_list, flush=True)
    
# for number, jet in enumerate(jet_list):
#     add_info_to_file((number, jet))
    
add_info_to_file((1, jets_list[1]))

# enumerate jet lists tienen 4 , dos pare traiin/test de jet good higgs y otros dos test/train de jet good

# results=enumerate(jets_list)
# print("results",list(results))


# print(jets_list[0].type)
# print(jets_list[1].type)
# print(len(jets_list))

