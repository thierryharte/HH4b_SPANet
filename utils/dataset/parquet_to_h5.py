# sbatch -p short --time 00:20:00 --account=t3 --mem 15gb --cpus-per-task=32 --wrap="python parquet_to_h5_classification.py -i /work/ramella/parquet_files/full_dataset/DATA_JetMET_JMENano_2b_region.parquet  /work/ramella/parquet_files/GluGlutoHHto4B_4b_region.parquet  -o  /work/ramella/h5_files/full_dataset/ -c"

# safe resources for signal region:
# for 2b total dataset test--> 3.5h, 15Gb
# for 2b total dataset train--> 7h, 20Gb
# for reduced test --> 1h, 3Gb
# for reduced train --> 3h, 3Gb


import awkward as ak
import numpy as np
import os
import h5py
import vector
import argparse
import onnxruntime
from prediction_selection import extract_predictions

vector.register_numba()
vector.register_awkward()

# pass arguments to run the code
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
    "-t",
    "--spanet-training",
    type=str,
    default="/work/mmalucch/out_hh4b_5jets_ATLAS_ptreg_c0_lr1e4_wp0_noklininp_oc_300e_kl3p5.onnx",
    help="Spanet training",
)
parser.add_argument(
    "-w",
    "--reweight",
    action="store_true",
    default=False,
    help="Reweight the different regions to have same sum",
)
parser.add_argument(
    "-f",
    "--frac-train",
    type=float,
    default=0.8,
    help="Fraction of events to use for training",
)
parser.add_argument(
    "--num-jets",
    type=int,
    default=5,
    help="Number of JetGood to use in the dataset",
)
parser.add_argument(
    "-m",
    "--max-events",
    type=int,
    default=-1,
    help="Maximum number of events to process",
)
parser.add_argument(
    "-b",
    "--batch-size",
    type=int,
    default=6000,
    help="Batch size for the model",
)
parser.add_argument(
    "-n",
    "--num-threads",
    type=int,
    default=1,
    help="Number of threads to use",
)
parser.add_argument(
    "--type",
    default=-1,
    type=int,
    help="Type of the dataset (0=JetGood_train, 1=JetGood_test, 2=JetGoodHiggs_train, 3=JetGoodHiggs_test)",
)
parser.add_argument(
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
parser.add_argument(
    "-s",
    "--signal",
    action="store_true",
    default=False,
    help="Mask for signal region",
)
parser.add_argument(
    "-r",
    "--random_pt",
    action="store_true",
    default=False,
    help="Applying a random weight to pT to reduce mass dependence",
)

args = parser.parse_args()
print(args)

if args.classification or args.signal:

    sess_opts = onnxruntime.SessionOptions()
    sess_opts.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
    sess_opts.inter_op_num_threads = args.num_threads  # parallelize the call
    sess_opts.intra_op_num_threads = args.num_threads
    print(
        "THREADS",
        sess_opts.inter_op_num_threads,
        sess_opts.intra_op_num_threads,
        flush=True,
    )

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


def get_awkward_array_shape(arr):
    if len(arr) == 0:
        return (0,)
    shape = []
    while isinstance(arr, ak.Array):
        shape.append(len(arr))
        arr = arr[0]
    return tuple(shape)


def reconstruct_higgs(jet_collection, idx_collection):

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


def create_targets(file, particle, jets_prov, filename, max_num_jets):
    indices = ak.local_index(jets_prov)
    higgs_targets = {1: ["b1", "b2"], 2: ["b3", "b4"]}

    print("PREDICTIONS BEST")
    print(jets_prov)

    for j in [1, 2]:
        if particle == f"h{j}":
            if ak.all(jets_prov == -1):
                if args.classification:
                    index_b1 = ak.full_like(jets_prov[:, 0], 0 + (j - 1) * 2)
                    index_b2 = ak.full_like(jets_prov[:, 0], 1 + (j - 1) * 2)
                else:
                    index_b1 = ak.full_like(jets_prov[:, 0], 0)
                    index_b2 = ak.full_like(jets_prov[:, 0], 0)
                # print(filename, particle, index_b1, index_b2)1
            else:
                mask = jets_prov == j  # H->b1b2
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


# four vector for only fully matched events
def jet_four_vector_fully_matched(jet):
    jet_prov_unflat = jet.prov
    jet_pt_unflat = jet.pt
    jet_eta_unflat = jet.eta
    jet_phi_unflat = jet.phi
    jet_mass_unflat = jet.mass
    jet_btag_unflat = jet.btag
    jet_btag_wp_unflat = jet.btag_wp
    jet_btag_3wp_unflat = ak.where(jet.btag_wp > 2.5, 2, jet.btag_wp)

    print(jet_prov_unflat)
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
            "btag_wp": jet_btag_wp_unflat[mask_fully_matched],
            "btag_3wp": jet_btag_3wp_unflat[mask_fully_matched],
            "prov": jet_prov_unflat[mask_fully_matched],
        },
        with_name="Momentum4D",
    )
    print(jet_fully_matched)
    return jet_fully_matched


def jet_four_vector(jets_list):

    # k, jets_list = input_to_file
    jet_prov_unflat = jets_list.prov
    jet_pt_unflat = jets_list.pt
    jet_eta_unflat = jets_list.eta
    jet_phi_unflat = jets_list.phi
    jet_mass_unflat = jets_list.mass
    jet_btag_unflat = jets_list.btag
    jet_btag_wp_unflat = jets_list.btag_wp
    jet_btag_3wp_unflat = ak.where(jets_list.btag_wp > 2.5, 2, jets_list.btag_wp)

    jet = ak.zip(
        {
            "pt": jet_pt_unflat,
            "eta": jet_eta_unflat,
            "phi": jet_phi_unflat,
            "mass": jet_mass_unflat,
            "btag": jet_btag_unflat,
            "btag_wp": jet_btag_wp_unflat,
            "btag_3wp": jet_btag_3wp_unflat,
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

    btag_wp_array = ak.to_numpy(
        ak.fill_none(ak.pad_none(jets.btag_wp, NUMBER_JETS_SPANET, clip=True), PAD_VALUE)
    )

    btag_wp = btag_wp_array.astype("int64")

    btag_3wp_array = ak.to_numpy(
        ak.fill_none(ak.pad_none(ak.where(jets.btag_wp > 2.5, 2, jets.btag_wp), NUMBER_JETS_SPANET, clip=True), PAD_VALUE)
    )

    btag_3wp = btag_3wp_array.astype("int64")

    mask = ~(phi_array == PAD_VALUE)
    # we have the inputs in which we will evaluate our model
    inputs = np.stack((ptPnetRegNeutrino, eta, phi, btag, btag_wp), axis=-1)
    # inputs_complete = {input_name[0]: inputs, input_name[1]: mask}
    # print(inputs.shape)
    # print(mask.shape)

    # # evalutaion of the model on these inputs
    batch_size = args.batch_size
    nbatches = int(len(jets) / batch_size) if int(len(jets) / batch_size) > 0 else 1
    print("nbatches", nbatches)
    total_prob_h1 = []
    total_prob_h2 = []
    # for i in track(range(nbatches), "Inference"):
    for i in range(nbatches):
        if i % 1 == 0:
            print(
                f"Batch {i}/{nbatches}  //  percentage {i/nbatches*100} %", flush=True
            )

        start = i * batch_size
        if i < (nbatches - 1):
            stop = start + batch_size
            # print("stop 1", stop)
        else:
            stop = len(jets)
        inputs_complete = {
            input_name[0]: inputs[start:stop],
            input_name[1]: mask[start:stop],
        }
        outputs = session.run(output_name, inputs_complete)
        prob_h1 = outputs[0]
        prob_h2 = outputs[1]
        total_prob_h1.append(prob_h1)
        total_prob_h2.append(prob_h2)

    # print(total_prob_h1)
    # print("total outputs", total_outputs)
    print("len total proba h1", len(total_prob_h1), flush=True)
    print("len total proba h2", len(total_prob_h2), flush=True)

    prob_h1 = np.concatenate(total_prob_h1, axis=0)
    prob_h2 = np.concatenate(total_prob_h2, axis=0)

    print("prob h1", prob_h1.shape)
    print("prob h2", prob_h2.shape)
    # outputs = session.run(output_name, inputs_complete)
    print("file", file)

    # extract the best jet assignment from
    # the predicted probabilities
    assignment_probability = np.stack((prob_h1, prob_h2), axis=0)
    # print("\nassignment_probability", assignment_probability)

    # print("matrix proba",outputs[0])
    # swap axis
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


def add_fields(collection, fields=["pt", "eta", "phi", "mass"], four_vec=True):
    if four_vec:
        fields_dict = {field: getattr(collection, field) for field in fields}
        collection = ak.zip(
            fields_dict,
            with_name="PtEtaPhiMLorentzVector",
        )
    else:
        for field in fields:
            collection = ak.with_field(collection, getattr(collection, field), field)

    return collection


def reconstruct_higgs_from_provenance(jets):
    print(jets.prov)
    print(len(jets))
    print(len(jets.prov))

    jet_higgs1 = jets[jets.prov == 1]
    jet_higgs2 = jets[jets.prov == 2]

    jet_higgs1 = jet_higgs1[ak.argsort(jet_higgs1.pt, axis=1, ascending=False)]
    jet_higgs2 = jet_higgs2[ak.argsort(jet_higgs2.pt, axis=1, ascending=False)]
    print(jet_higgs1.prov)
    print(jet_higgs2.prov)

    higgs_lead = add_fields(jet_higgs1[:, 0] + jet_higgs1[:, 1])
    higgs_sub = add_fields(jet_higgs2[:, 0] + jet_higgs2[:, 1])

    jets_ordered = ak.with_name(
        ak.concatenate([jet_higgs1[:, :2], jet_higgs2[:, :2]], axis=1),
        name="PtEtaPhiMCandidate",
    )

    return higgs_lead, higgs_sub, jets_ordered


def create_inputs(
    file, jets, jet_4vector, max_num_jets, global_fifth_jet, events, **kwargs
):
    print(events)
    if args.classification or args.signal:
        (
            difference,
            predictions_best,
            best_pairing_probabilities_sum,
            second_best_pairing_probabilities_sum,
        ) = get_pairing_information(jets, file)

        HiggsLeading, HiggsSubLeading, pairing_predictions_ordered = reconstruct_higgs(
            jet_4vector, predictions_best
        )
        if args.signal:
            r_HH = ak.to_numpy(
                np.sqrt(
                    (HiggsLeading.mass - 125) ** 2 + (HiggsSubLeading.mass - 120) ** 2
                )
            )
            mask_sr = ak.to_numpy(r_HH < 30)
            r_HH = r_HH[mask_sr]
            file.create_dataset(
                "INPUTS/Event/r_HH",
                np.shape(r_HH),
                dtype="float32",
                data=r_HH,
            )
            # mask_sr = np.full(len(jets), True)

        else:
            mask_sr = np.full(len(jets), True)
        print("r_hh", r_HH)
        print("len jets", len(jets))
        print("mask_sr", mask_sr.shape, np.sum(mask_sr))

        jet_4vector = jet_4vector[mask_sr]
        difference = difference[mask_sr]
        predictions_best = predictions_best[mask_sr]
        best_pairing_probabilities_sum = best_pairing_probabilities_sum[mask_sr]
        second_best_pairing_probabilities_sum = second_best_pairing_probabilities_sum[
            mask_sr
        ]
        HiggsLeading = HiggsLeading[mask_sr]
        HiggsSubLeading = HiggsSubLeading[mask_sr]
        pairing_predictions_ordered = pairing_predictions_ordered[mask_sr]

    else:
        mask_sr = np.full(len(jets), True)

    jets = jets[mask_sr]
    events = events[mask_sr]
    print("len jets", len(jets))

    if args.random_pt:
        print("")
        print("")
        print("random pt on")
        print("")
        print("")
        jet_4vector_full = jet_four_vector_fully_matched(jets)
        ptPnetRegNeutrino_array_orig = ak.to_numpy(
            ak.fill_none(
                ak.pad_none(kwargs["jet_pt_original"], max_num_jets, clip=True),
                PAD_VALUE,
            )
        )
        ptPnetRegNeutrino_array = ak.to_numpy(
            ak.fill_none(ak.pad_none(jets.pt, max_num_jets, clip=True), PAD_VALUE)
        )
        file.create_dataset(
            "INPUTS/Jet/ptPnetRegNeutrino_unweighted",
            np.shape(ptPnetRegNeutrino_array_orig),
            dtype="float32",
            data=ptPnetRegNeutrino_array_orig,
        )
        ptPnetRegNeutrino_ds = file.create_dataset(
            "INPUTS/Jet/ptPnetRegNeutrino",
            np.shape(ptPnetRegNeutrino_array),
            dtype="float32",
            data=ptPnetRegNeutrino_array,
        )
        file.create_dataset(
            "INPUTS/Event/random_pt_weight",
            np.shape(kwargs["random_weights"]),
            dtype="float32",
            data=kwargs["random_weights"],
        )
        higgs_leading, higgs_subleading, _ = reconstruct_higgs_from_provenance(
            jet_4vector_full
        )

        for higgs, name in zip(
            [higgs_leading, higgs_subleading], ["HiggsLeading", "HiggsSubLeading"]
        ):
            higgs_pt = higgs.pt
            file.create_dataset(
                f"INPUTS/{name}/pt",
                np.shape(higgs_pt),
                dtype="float32",
                data=higgs_pt,
            )

            higgs_eta = higgs.eta
            file.create_dataset(
                f"INPUTS/{name}/eta",
                np.shape(higgs_eta),
                dtype="float32",
                data=higgs_eta,
            )

            higgs_phi = higgs.phi
            file.create_dataset(
                f"INPUTS/{name}/phi",
                np.shape(higgs_phi),
                dtype="float32",
                data=higgs_phi,
            )

            higgs_mass = higgs.mass
            file.create_dataset(
                f"INPUTS/{name}/mass",
                np.shape(higgs_mass),
                dtype="float32",
                data=higgs_mass,
            )

    else:
        # first you create the variable and then the dataset
        ptPnetRegNeutrino_array = ak.to_numpy(
            ak.fill_none(ak.pad_none(jets.pt, max_num_jets, clip=True), PAD_VALUE)
        )
        file.create_dataset(
            "INPUTS/Jet/ptPnetRegNeutrino",
            np.shape(ptPnetRegNeutrino_array),
            dtype="float32",
            data=ptPnetRegNeutrino_array,
        )

    mask = ~(ptPnetRegNeutrino_array == PAD_VALUE)
    file.create_dataset("INPUTS/Jet/MASK", np.shape(mask), dtype="bool", data=mask)

    phi_array = ak.to_numpy(
        ak.fill_none(ak.pad_none(jets.phi, max_num_jets, clip=True), PAD_VALUE)
    )
    file.create_dataset(
        "INPUTS/Jet/phi", np.shape(phi_array), dtype="float32", data=phi_array
    )
    # compute the cos and sin of phi
    cos_phi = ak.to_numpy(
        ak.fill_none(ak.pad_none(np.cos(jets.phi), max_num_jets, clip=True), PAD_VALUE)
    )

    file.create_dataset(
        "INPUTS/Jet/cosPhi", np.shape(cos_phi), dtype="float32", data=cos_phi
    )

    sin_phi = ak.to_numpy(
        ak.fill_none(ak.pad_none(np.sin(jets.phi), max_num_jets, clip=True), PAD_VALUE)
    )
    file.create_dataset(
        "INPUTS/Jet/sinPhi", np.shape(sin_phi), dtype="float32", data=sin_phi
    )

    eta_array = ak.to_numpy(
        ak.fill_none(ak.pad_none(jets.eta, max_num_jets, clip=True), PAD_VALUE)
    )
    file.create_dataset(
        "INPUTS/Jet/eta", np.shape(eta_array), dtype="float32", data=eta_array
    )

    ## Adding variations of b-tags ##
    btag = ak.to_numpy(
        ak.fill_none(ak.pad_none(jets.btag, max_num_jets, clip=True), PAD_VALUE)
    )
    num_events = btag.shape[0]
    btag12 = np.stack(
        [btag[:, 0], btag[:, 1], *[np.full(num_events, 0)] * (max_num_jets - 2)], axis=1
    )
    bratio_sum_1 = btag[:, 0] / (btag[:, 0] + btag[:, 1])
    bratio_sum_2 = btag[:, 1] / (btag[:, 0] + btag[:, 1])
    bratio_sum_3 = btag[:, 2] / (btag[:, 2] + btag[:, 3])
    bratio_sum_4 = btag[:, 3] / (btag[:, 2] + btag[:, 3])
    # JetGoodHiggs will only have 4 jets...
    if max_num_jets > 4:
        bratio_sum_5 = btag[:, 4] / (btag[:, 2] + btag[:, 3])
        btag12_bratio = np.stack(
            [
                btag[:, 0],
                btag[:, 1],
                bratio_sum_3,
                bratio_sum_4,
                bratio_sum_5,
                *[np.full(num_events, PAD_VALUE)] * (max_num_jets - 5),
            ],
            axis=1,
        )
        bratio_all = np.stack(
            [
                bratio_sum_1,
                bratio_sum_2,
                bratio_sum_3,
                bratio_sum_4,
                bratio_sum_5,
                *[np.full(num_events, PAD_VALUE)] * (max_num_jets - 5),
            ],
            axis=1,
        )
    else:
        btag12_bratio = np.stack(
            [
                btag[:, 0],
                btag[:, 1],
                bratio_sum_3,
                bratio_sum_4,
                *[np.full(num_events, PAD_VALUE)] * (max_num_jets - 4),
            ],
            axis=1,
        )
        bratio_all = np.stack(
            [
                bratio_sum_1,
                bratio_sum_2,
                bratio_sum_3,
                bratio_sum_4,
                *[np.full(num_events, PAD_VALUE)] * (max_num_jets - 4),
            ],
            axis=1,
        )

    file.create_dataset("INPUTS/Jet/btag12_bratio", data=btag12_bratio, dtype="float32")
    file.create_dataset("INPUTS/Jet/btag12", data=btag12, dtype="float32")
    file.create_dataset("INPUTS/Jet/bratio_all", data=bratio_all, dtype="float32")
    file.create_dataset("INPUTS/Jet/btag", np.shape(btag), dtype="float32", data=btag)
    btag_wp = ak.to_numpy(
        ak.fill_none(ak.pad_none(jets.btag_wp, max_num_jets, clip=True), PAD_VALUE)
    )
    btag_wp_ds = file.create_dataset(
        "INPUTS/Jet/btag_wp",
        np.shape(btag_wp),
        dtype="int64",
        data=btag_wp,
    )
    btag_3wp = ak.to_numpy(
            ak.fill_none(ak.pad_none(ak.where(jets.btag_wp > 3.5, 3, jets.btag_wp), max_num_jets, clip=True), PAD_VALUE)
    )
    btag_3wp_ds = file.create_dataset(
        "INPUTS/Jet/btag_3wp",
        np.shape(btag_3wp),
        dtype="int64",
        data=btag_3wp,
    )

    btag_1wp = ak.to_numpy(
        ak.fill_none(ak.pad_none(ak.where(jets.btag_wp > 0, 0, jets.btag_wp), max_num_jets, clip=True), PAD_VALUE)
    )
    btag_1wp_ds = file.create_dataset(
        "INPUTS/Jet/btag_1wp",
        np.shape(btag_1wp),
        dtype="int64",
        data=btag_1wp,
    )
    bwp_diff_1 = btag_wp[:, 0] - btag_wp[:, 1]
    bwp_diff_2 = btag_wp[:, 1] - btag_wp[:, 0]
    bwp_diff_3 = btag_wp[:, 2] - btag_wp[:, 3]
    bwp_diff_4 = btag_wp[:, 3] - btag_wp[:, 2]
    # JetGoodHiggs will only have 4 jets...
    if max_num_jets > 4:
        bwp_diff_5 = np.where(btag_wp[:, 4] == 9999, btag_wp[:, 4], btag_wp[:, 4] - btag_wp[:, 3])
        btag_wp_diff = np.stack(
            [
                bwp_diff_1,
                bwp_diff_2,
                bwp_diff_3,
                bwp_diff_4,
                bwp_diff_5,
                *[np.full(num_events, PAD_VALUE)] * (max_num_jets - 5),
            ],
            axis=1,
        )
    else:
        btag_wp_diff = np.stack(
            [
                bwp_diff_1,
                bwp_diff_2,
                bwp_diff_3,
                bwp_diff_4,
                *[np.full(num_events, PAD_VALUE)] * (max_num_jets - 4),
            ],
            axis=1,
        )
    file.create_dataset("INPUTS/Jet/btag_wp_diff", data=btag_wp_diff, dtype="float32")

    mass_array = ak.to_numpy(
        ak.fill_none(ak.pad_none(jets.mass, max_num_jets, clip=True), PAD_VALUE)
    )
    file.create_dataset(
        "INPUTS/Jet/mass", np.shape(mass_array), dtype="float32", data=mass_array
    )

    kl_array = ak.to_numpy(events.kl)
    file.create_dataset(
        "INPUTS/Event/kl", np.shape(kl_array), dtype="float32", data=kl_array
    )
    is_preEE_array = ak.to_numpy(events.is_preEE)
    is_postEE_array = ak.to_numpy(events.is_postEE)
    file.create_dataset(
        "INPUTS/Event/is_preEE",
        np.shape(is_preEE_array),
        dtype="float32",
        data=is_preEE_array,
    )
    file.create_dataset(
        "INPUTS/Event/is_postEE",
        np.shape(is_postEE_array),
        dtype="float32",
        data=is_postEE_array,
    )
    if "random_pt_weights" in events:
        random_weights_array = ak.to_numpy(events.random_pt_weights)
        file.create_dataset(
            "INPUTS/Event/random_pt_weights",
            np.shape(random_weights_array),
            dtype="float32",
            data=random_weights_array,
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
        file.create_dataset(
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
        file.create_dataset(
            "INPUTS/Event/HT",
            np.shape(ht_array),
            dtype="int64",
            data=ht_array,
        )

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
        print("dr post flat", dR, flush=True)
        dR_min = ak.min(dR, axis=1)

        print("dR_min", dR_min, flush=True)

        dR_max = ak.max(dR, axis=1)

        print("dR_max", dR_max)

        file.create_dataset(
            "INPUTS/Event/dR_min",
            np.shape(dR_min),
            dtype="float32",
            data=dR_min,
        )

        file.create_dataset(
            "INPUTS/Event/dR_max",
            np.shape(dR_max),
            dtype="float32",
            data=dR_max,
        )

    if args.classification or args.signal:
        # print("higgs_1", higgs_1)
        # print("higgs_2", higgs_2)
        print("Higgs Leading", HiggsLeading)

        HiggsLeading_pt = HiggsLeading.pt
        file.create_dataset(
            "INPUTS/HiggsLeading/pt",
            np.shape(HiggsLeading_pt),
            dtype="float32",
            data=HiggsLeading_pt,
        )

        HiggsLeading_eta = HiggsLeading.eta
        file.create_dataset(
            "INPUTS/HiggsLeading/eta",
            np.shape(HiggsLeading_eta),
            dtype="float32",
            data=HiggsLeading_eta,
        )

        HiggsLeading_phi = HiggsLeading.phi
        file.create_dataset(
            "INPUTS/HiggsLeading/phi",
            np.shape(HiggsLeading_phi),
            dtype="float32",
            data=HiggsLeading_phi,
        )

        HiggsLeading_mass = HiggsLeading.mass
        file.create_dataset(
            "INPUTS/HiggsLeading/mass",
            np.shape(HiggsLeading_mass),
            dtype="float32",
            data=HiggsLeading_mass,
        )

        HiggsLeading_costheta = abs(np.cos(HiggsLeading.theta))
        file.create_dataset(
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
        file.create_dataset(
            "INPUTS/HiggsLeading/dR",
            np.shape(HiggsLeading_dR),
            dtype="float32",
            data=HiggsLeading_dR,
        )

        HiggsSubLeading_pt = HiggsSubLeading.pt
        file.create_dataset(
            "INPUTS/HiggsSubLeading/pt",
            np.shape(HiggsSubLeading_pt),
            dtype="float32",
            data=HiggsSubLeading_pt,
        )

        HiggsSubLeading_eta = HiggsSubLeading.eta
        file.create_dataset(
            "INPUTS/HiggsSubLeading/eta",
            np.shape(HiggsSubLeading_eta),
            dtype="float32",
            data=HiggsSubLeading_eta,
        )

        HiggsSubLeading_phi = HiggsSubLeading.phi
        file.create_dataset(
            "INPUTS/HiggsSubLeading/phi",
            np.shape(HiggsSubLeading_phi),
            dtype="float32",
            data=HiggsSubLeading_phi,
        )

        HiggsSubLeading_mass = HiggsSubLeading.mass
        file.create_dataset(
            "INPUTS/HiggsSubLeading/mass",
            np.shape(HiggsSubLeading_mass),
            dtype="float32",
            data=HiggsSubLeading_mass,
        )

        HiggsSubLeading_costheta = abs(np.cos(HiggsSubLeading.theta))
        file.create_dataset(
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
        file.create_dataset(
            "INPUTS/HiggsSubLeading/dR",
            np.shape(HiggsSubLeading_dR),
            dtype="float32",
            data=HiggsSubLeading_dR,
        )

        HH = HiggsLeading + HiggsSubLeading

        HH_pt = HH.pt
        file.create_dataset(
            "INPUTS/HH/pt",
            np.shape(HH_pt),
            dtype="float32",
            data=HH_pt,
        )

        HH_eta = HH.eta
        file.create_dataset(
            "INPUTS/HH/eta",
            np.shape(HH_eta),
            dtype="float32",
            data=HH_eta,
        )

        HH_phi = HH.phi
        file.create_dataset(
            "INPUTS/HH/phi",
            np.shape(HH_phi),
            dtype="float32",
            data=HH_phi,
        )

        HH_mass = HH.mass
        file.create_dataset(
            "INPUTS/HH/mass",
            np.shape(HH_mass),
            dtype="float32",
            data=HH_mass,
        )

        HH_dR = HiggsLeading.deltaR(HiggsSubLeading)
        file.create_dataset(
            "INPUTS/HH/dR",
            np.shape(HH_dR),
            dtype="float32",
            data=HH_dR,
        )

        HH_cos = abs(np.cos(HH.theta))
        file.create_dataset(
            "INPUTS/HH/cos_theta_star",
            np.shape(HH_cos),
            dtype="float32",
            data=HH_cos,
        )

        HH_dEta = abs(HiggsLeading.eta - HiggsSubLeading.eta)
        file.create_dataset(
            "INPUTS/HH/dEta",
            np.shape(HH_dEta),
            dtype="float32",
            data=HH_dEta,
        )

        HH_dPhi = HiggsLeading.deltaphi(HiggsSubLeading)
        file.create_dataset(
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

        file.create_dataset(
            "INPUTS/Event/Best_pairing_probabilities_sum",
            np.shape(best_pairing_probabilities_sum),
            dtype="float32",
            data=best_pairing_probabilities_sum,
        )

        file.create_dataset(
            "INPUTS/Event/Second_best_pairing_probabilities",
            np.shape(second_best_pairing_probabilities_sum),
            dtype="float32",
            data=second_best_pairing_probabilities_sum,
        )

        difference = ak.to_numpy(difference)

        file.create_dataset(
            "INPUTS/Event/Probability_difference",
            np.shape(difference),
            dtype="float32",
            data=difference,
        )

    else:
        weight_array = ak.to_numpy(events.weight)
        file.create_dataset(
            "WEIGHTS/weight",
            np.shape(weight_array),
            dtype="float32",
            data=weight_array,
        )
    # create new global variables for the fifth jet (if it exists) otherwise fill with PAD_VALUE
    if global_fifth_jet is not None:
        global_fifth_jet = global_fifth_jet[mask_sr]
        pt_array_5 = ak.to_numpy(
            ak.fill_none(ak.pad_none(global_fifth_jet.pt, 5, clip=True), PAD_VALUE)[
                :, 4
            ]
        )
        file.create_dataset(
            "INPUTS/FifthJet/pt", np.shape(pt_array_5), dtype="float32", data=pt_array_5
        )

        ptPnetRegNeutrino_array_5 = ak.to_numpy(
            ak.fill_none(
                ak.pad_none(global_fifth_jet.pt, 5, clip=True),
                PAD_VALUE,
            )[:, 4]
        )
        file.create_dataset(
            "INPUTS/FifthJet/ptPnetRegNeutrino",
            np.shape(ptPnetRegNeutrino_array_5),
            dtype="float32",
            data=ptPnetRegNeutrino_array_5,
        )

        phi_array_5 = ak.to_numpy(
            ak.fill_none(
                ak.pad_none(global_fifth_jet.phi, 5, clip=True),
                PAD_VALUE,
            )[:, 4]
        )
        file.create_dataset(
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
        file.create_dataset(
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
        file.create_dataset(
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
        file.create_dataset(
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
        file.create_dataset(
            "INPUTS/FifthJet/btag", np.shape(btag_5), dtype="float32", data=btag_5
        )
        btag_wp_5 = ak.to_numpy(
            ak.fill_none(ak.pad_none(global_fifth_jet.btag_wp, 5, clip=True), PAD_VALUE)[
                :, 4
            ]
        )
        btag_wp_ds_5 = file.create_dataset(
            "INPUTS/FifthJet/btag_wp", np.shape(btag_wp_5), dtype="int64", data=btag_wp_5
        )

        mass_array_5 = ak.to_numpy(
            ak.fill_none(ak.pad_none(global_fifth_jet.mass, 5, clip=True), PAD_VALUE)[
                :, 4
            ]
        )
        file.create_dataset(
            "INPUTS/FifthJet/mass",
            np.shape(mass_array_5),
            dtype="float32",
            data=mass_array_5,
        )
    return mask_sr


def add_info_to_file(input_to_file):
    # since the input an enumerate object, k corresponds to the index and jets to the actual list
    k, jets = input_to_file
    print(f"\n\nAdding info to file {main_dir}/{file_dict[k]}")
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

    if args.random_pt:
        random_weights = ak.Array(np.random.rand((len(jets))) + 0.5)
        jet_pt_original = jets.pt
        jets["pt"] = jets.pt * random_weights
        # jets["mass"] = jets.mass*random_weights

        jet_4vector = jet_four_vector(jets)
        mask_sr = create_inputs(
            file_out,
            jets,
            jet_4vector,
            max_num_jets_list[k],
            global_fifth_jet,
            events_list[k],
            random_weights=random_weights,
            jet_pt_original=jet_pt_original,
        )

    else:
        jet_4vector = jet_four_vector(jets)
        mask_sr = create_inputs(
            file_out,
            jets,
            jet_4vector,
            max_num_jets_list[k],
            global_fifth_jet,
            events_list[k],
        )

    # evaluate the model for the events
    # predictions_best,  best_pairing_probabilities_sum , second_best_pairing_probabilities_sum= get_pairing_information(jets,file_out)

    # print("best", predictions_best)
    # print("bes sum", best_pairing_probabilities_sum)
    # print("second best", second_best_pairing_probabilities_sum)

    create_targets(
        file_out, "h1", jets.prov[mask_sr], file_dict[k], max_num_jets_list[k]
    )
    create_targets(
        file_out, "h2", jets.prov[mask_sr], file_dict[k], max_num_jets_list[k]
    )
    print("Completed file ", file_dict[k])
    file_out.close()


if __name__ == "__main__":
    main_dir = args.output if args.output else os.path.dirname(args.input[0])
    os.makedirs(main_dir, exist_ok=True)

    leninput = len(list(args.input))

    dfs = []
    sum_gen_weights = []
    for filename in list(args.input):
        newdf = ak.from_parquet(filename)
        print(newdf.type)
        sum_gen_weights.append(sum(newdf.event.weight))
        if "spanet" in filename:
            newdf = ak.with_field(
                newdf,
                ak.with_field(
                    newdf.event, ak.ones_like(newdf.event.weight), "is_postEE"
                ),
                "event",
            )
            newdf = ak.with_field(
                newdf,
                ak.with_field(
                    newdf.event, ak.zeros_like(newdf.event.weight), "is_preEE"
                ),
                "event",
            )
        else:
            newdf = ak.with_field(
                newdf,
                ak.with_field(
                    newdf.event, ak.zeros_like(newdf.event.weight), "is_postEE"
                ),
                "event",
            )
            newdf = ak.with_field(
                newdf,
                ak.with_field(
                    newdf.event, ak.ones_like(newdf.event.weight), "is_preEE"
                ),
                "event",
            )
        dfs.append(newdf)
        print(dfs[-1].event.sb)

    if args.reweight:
        print("Found following sums for the event weights of the files")
        for filename, sumweight in zip(list(args.input), sum_gen_weights):
            print(f"{filename}: {sumweight}")

        update_dfs = []
        for df, weight in zip(dfs, sum_gen_weights):
            # dividing all the weights by the sum of the weights. Then multiplying with ratio between largest and current sum_gen_weight
            renorm_weight = max(sum_gen_weights) / weight
            print(f"Renormalized weight: {renorm_weight}")
            df = ak.with_field(
                df,
                ak.with_field(df.event, df.event.weight * renorm_weight, "weight"),
                "event",
            )
            print(df.event.weight)
            update_dfs.append(df)
        dfs = update_dfs

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

    jets_good = df.JetGood[: args.max_events] if args.max_events != -1 else df.JetGood
    jets_good_higgs = (
        df.JetGoodHiggs[: args.max_events] if args.max_events != -1 else df.JetGoodHiggs
    )

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
        events_all = df.event[: args.max_events] if args.max_events != -1 else df.event
        print("events_all", events_all)
        print("jets_all", jets_all)
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

    if args.type != -1:
        add_info_to_file((args.type, jets_list[args.type]))
    else:
        for number, jet in enumerate(jets_list):
            add_info_to_file((number, jet))
