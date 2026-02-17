#!/usr/bin/env python3

import re
import argparse
import awkward as ak
import coffea.util
import h5py
import numpy as np


# JET_COLLECTION = "JetTotalSPANetPtFlattenPadded"
JET_COLLECTION = "JetGood"
KEEP_TOGETHER_COLLECTIONS = ["add_jet1pt"]

# "all" means all non-jet variables are saved as global variables
# leave empty list to disable global variables
GLOBAL_VARIABLES = [""]

COFFEA_PADDING_VALUE = -999.0
H5_PADDING_VALUE = 9999.0
SEED = 9999
_permutations = {}

RESONANCES = {
    "h1": (1, ("b1", "b2")),
    "h2": (2, ("b3", "b4")),
    "vbf": (3, ("q1", "q2")),
}

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def is_awkward(x):
    return isinstance(x, (ak.Array, ak.Record))


def create_resonances_targets_from_provenance(jets_prov, max_jets):
    """
    Parameters
    ----------
    jets_prov : awkward.Array
        Shape (Nevents, Njets), provenance labels:
          1 → h1
          2 → h2
          3 → VBF
          pad value → invalid / padded
    max_jets : int

    Returns
    -------
    dict with keys:
      ("h1", "b1"), ("h1", "b2"), ("h2", "b3"), ("h2", "b4"), ("vbf", "q1"), ("vbf", "q2")
    Each value is a numpy array of shape (Nevents,) with jet indices or -1
    """

    # local jet indices per event
    indices = ak.local_index(jets_prov)

    targets = {}

    for resonance, (prov_id, labels) in RESONANCES.items():

        # mask jets belonging to this resonance
        mask = jets_prov == prov_id

        # pick jet indices, pad to exactly 2, fill missing with -1
        idx = ak.fill_none(ak.pad_none(indices[mask], 2, axis=1), -1)

        idx1 = idx[:, 0]
        idx2 = idx[:, 1]

        # enforce max_jets
        idx1 = ak.where(idx1 < max_jets, idx1, -1)
        idx2 = ak.where(idx2 < max_jets, idx2, -1)

        targets[(resonance, labels[0])] = ak.to_numpy(idx1)
        targets[(resonance, labels[1])] = ak.to_numpy(idx2)

    return targets


def create_dummy_targets(N, tr_targets, te_targets, train_mask, test_mask, shuffle):
    idx = 0
    for key in RESONANCES.keys():
        for label in RESONANCES[key][1]:
            arr = np.full(N, idx, dtype=np.int64)
            write_block_split(
                tr_targets,
                te_targets,
                [key, label],
                arr,
                train_mask,
                test_mask,
                shuffle,
            )
            idx += 1


def unwrap_accumulator(x):
    """Unwrap common coffea accumulator wrappers (duck-typed)."""
    if hasattr(x, "value"):
        try:
            return x.value
        except Exception:
            pass

    if hasattr(x, "_value"):
        try:
            return x._value
        except Exception:
            pass

    for meth in ("to_numpy", "numpy"):
        if hasattr(x, meth):
            try:
                return getattr(x, meth)()
            except Exception:
                pass

    return x


def unflatten_to_jagged(flat, counts):
    """flat: 1D array of length sum(counts)
    counts: 1D int array of length Nevents
    returns: awkward jagged array (Nevents, Nj)
    """
    flat = unwrap_accumulator(flat)
    counts = unwrap_accumulator(counts)

    flat = np.asarray(flat)
    counts = np.asarray(counts).astype(np.int64)

    if flat.ndim != 1:
        raise ValueError(f"Expected flat 1D array, got shape {flat.shape}")
    if counts.ndim != 1:
        raise ValueError(f"Expected 1D counts array, got shape {counts.shape}")
    if flat.shape[0] != int(counts.sum()):
        raise ValueError("Flat length != sum(counts)")

    return ak.unflatten(flat, counts)


def to_numpy_event_vector(x):
    """Convert event-level data to a strict 1D numeric numpy array."""
    x = unwrap_accumulator(x)

    if is_awkward(x):
        x = ak.to_numpy(x)

    arr = np.asarray(x)

    if arr.shape == () and arr.dtype == object:
        arr = np.asarray(unwrap_accumulator(arr.item()))

    if arr.ndim != 1:
        raise ValueError(f"Expected 1D event vector, got shape {arr.shape}")
    if arr.dtype == object:
        raise TypeError("Object dtype after unwrapping")
    return arr


def pad_clip_jets(jets, max_jets):
    jets = unwrap_accumulator(jets)
    
    # replace coffea padding to h5 padding
    jets=ak.where(jets==COFFEA_PADDING_VALUE, H5_PADDING_VALUE, jets)

    # Awkward jagged
    if is_awkward(jets):
        padded = ak.pad_none(jets, max_jets, axis=1, clip=True)
        dense = ak.to_numpy(ak.fill_none(padded, H5_PADDING_VALUE))
        return dense

    arr = np.asarray(jets)

    # Ragged numpy object → awkward
    if arr.dtype == object and arr.ndim == 1:
        jets_ak = ak.Array(arr)
        padded = ak.pad_none(jets_ak, max_jets, axis=1, clip=True)
        dense = ak.to_numpy(ak.fill_none(padded, H5_PADDING_VALUE))
        return dense

    # Dense numpy
    if arr.ndim < 2:
        raise ValueError(f"Invalid jet array shape {arr.shape}")

    N, Nj = arr.shape[:2]
    use = min(Nj, max_jets)

    out = np.full((N, max_jets) + arr.shape[2:], H5_PADDING_VALUE, dtype=arr.dtype)
    out[:, :use, ...] = arr[:, :use, ...]

    return out


def infer_collection_and_var(name):
    """Rules:
    - if name starts with "events_", treat as Event-level collection "Event"
    - else split into (collection, var) by first underscore, EXCEPT keep-together collections
      e.g. "HH_pt" -> ("HH", "pt")
           "add_jet1pt_pt" -> ("add_jet1pt", "pt")
    - if no underscore, put it under Event-level "Event"
    """
    if name.startswith("events_"):
        return "Event", name[len("events_") :]

    for c in KEEP_TOGETHER_COLLECTIONS:
        if name.startswith(c + "_"):
            return c, name[len(c) + 1 :]

    if "_" in name:
        return name.split("_", 1)

    return "Event", name


def dataset_to_class_index(dataset_key, class_labels):
    key = dataset_key.lower()
    for idx, lbl in enumerate(class_labels):
        if lbl.lower() in key:
            return idx
    raise ValueError(f"Dataset key '{dataset_key}' does not match any class label")


def get_permutation(N):
    global _permutations
    if N not in _permutations:
        rng = np.random.default_rng(SEED)
        _permutations[N] = rng.permutation(N)
    return _permutations[N]


def cast_floats32(x):
    x = np.asarray(x)
    if np.issubdtype(x.dtype, np.floating):
        return x.astype(np.float32, copy=False)
    return x


def cast_int64(x):
    x = np.asarray(x)
    if np.issubdtype(x.dtype, np.integer):
        return x.astype(np.int64, copy=False)
    return x


def extract_kl_value(s):
    match = re.search(r"kl-([mp0-9]+)", s)
    if not match:
        return None

    value_str = match.group(1)
    value_str = value_str.replace("m", "-").replace("p", ".")
    return float(value_str)

# -----------------------------------------------------------------------------
# HDF5 helpers
# -----------------------------------------------------------------------------


def ensure_resizable_dataset(group, path, data, shuffle, compression="gzip"):
    """Create or append to a resizable dataset located at group/<path_parts...>.
    data: numpy array, first dimension is N (events)
    """
    g = group
    for p in path[:-1]:
        g = g.require_group(p)

    name = path[-1]
    data = np.asarray(data)

    if data.dtype == object:
        raise TypeError(f"Object dtype at {'/'.join(path)}")

    if name not in g:
        dset = g.create_dataset(
            name,
            data=data,
            maxshape=(None,) + data.shape[1:],
            chunks=True,
            compression=compression,
            shuffle=False,
        )
    else:
        dset = g[name]
        old = dset.shape[0]
        dset.resize((old + data.shape[0],) + dset.shape[1:])
        dset[old:] = data

    if shuffle:
        full = dset[()]
        full = full[get_permutation(len(full))]
        dset[...] = full


def write_block_split(tr, te, path, data, train_mask, test_mask, shuffle):
    """Append split slices of `data` to train/test datasets."""
    ensure_resizable_dataset(tr, path, data[train_mask], shuffle)
    ensure_resizable_dataset(te, path, data[test_mask], shuffle)


# -----------------------------------------------------------------------------
# Main conversion
# -----------------------------------------------------------------------------


def coffea_to_h5(
    coffea_path,
    h5_path,
    regions,
    class_labels,
    max_jets,
    train_frac,
    do_data_shuffling,
    columns_key="columns",
    weight_name="weight",
):
    """Convert the columns from coffea to h5 format to use as SPANet inputs."""

    cols = coffea.util.load(coffea_path)[columns_key]

    rng = np.random.default_rng(SEED)
    h5_tr = h5_path.replace(".h5", "_train.h5")
    h5_te = h5_path.replace(".h5", "_test.h5")

    with h5py.File(h5_tr, "w") as ftr, h5py.File(h5_te, "w") as fte:

        def mk(f):
            return (
                f.create_group("INPUTS"),
                f.create_group("WEIGHTS"),
                f.create_group("CLASSIFICATIONS"),
                f.create_group("TARGETS"),
            )

        tr_in, tr_w, tr_c, tr_t = mk(ftr)
        te_in, te_w, te_c, te_t = mk(fte)

        sample_keys = list(cols.keys())
        for i, skey in enumerate(sample_keys):
            # shuffle only on the last dataset to avoid multiple shufflings
            shuffle = do_data_shuffling and (i == len(sample_keys) - 1)

            class_idx = dataset_to_class_index(skey, class_labels)
            region = regions[class_idx]

            for dataset in cols[skey]:
                if region not in cols[skey][dataset]:
                    raise ValueError(
                        f"Region '{region}' not found for dataset '{skey}' dataset '{dataset}'"
                    )

                payload = cols[skey][dataset][region]

                w = to_numpy_event_vector(payload[weight_name])
                N = len(w)

                train_mask = (
                    rng.random(N) < train_frac
                    if shuffle
                    else np.arange(N) < int(N * train_frac)
                )
                test_mask = ~train_mask

                write_block_split(
                    tr_w,
                    te_w,
                    ["weight"],
                    cast_floats32(w),
                    train_mask,
                    test_mask,
                    shuffle,
                )

                cls = np.full(N, class_idx, dtype=np.int64)
                write_block_split(
                    tr_c, te_c, ["Event", "class"], cls, train_mask, test_mask, shuffle
                )

                jet_counts = None
                jetN = f"{JET_COLLECTION}_N"
                if jetN in payload:
                    jet_counts = to_numpy_event_vector(payload[jetN]).astype(np.int64)

                jet_pt = unflatten_to_jagged(
                    unwrap_accumulator(payload[f"{JET_COLLECTION}_pt"]), jet_counts
                )
                mask_jet_pt = (
                    ak.to_numpy(
                        ak.fill_none(
                            ak.pad_none(
                                jet_pt,
                                max_jets,
                                clip=True,
                            ),
                            COFFEA_PADDING_VALUE,
                        )
                    )
                    != COFFEA_PADDING_VALUE
                )

                jet_mask_written = False

                # Create the Targets
                prov_key = f"{JET_COLLECTION}_provenance"
                if prov_key in payload:
                    jets_prov = unflatten_to_jagged(
                        unwrap_accumulator(payload[prov_key]), jet_counts
                    )

                    # Split train / test *before* creating targets
                    prov_tr = jets_prov[train_mask]
                    prov_te = jets_prov[test_mask]

                    targets_tr = create_resonances_targets_from_provenance(
                        prov_tr, max_jets
                    )
                    targets_te = create_resonances_targets_from_provenance(
                        prov_te, max_jets
                    )

                    for (r, q), arr in targets_tr.items():
                        ensure_resizable_dataset(tr_t, [r, q], cast_int64(arr), shuffle)

                    for (r, q), arr in targets_te.items():
                        ensure_resizable_dataset(te_t, [r, q], cast_int64(arr), shuffle)
                else:
                    create_dummy_targets(N, tr_t, te_t, train_mask, test_mask, shuffle)

                for name, arr in payload.items():
                    if name == weight_name:
                        continue

                    coll, var = infer_collection_and_var(name)
                    print(name)
                    arr_u = unwrap_accumulator(arr)
                    is_jet = coll == JET_COLLECTION and var != "N"
                    is_global = var in GLOBAL_VARIABLES or (
                        not is_jet and "all" in GLOBAL_VARIABLES and coll == "events"
                    )
                    print(
                        f"Processing {skey} {dataset} {region} variable {name} with shape {arr_u.shape} (jet: {is_jet}, global: {is_global})"
                    )

                    if is_jet:
                        arr_np = np.asarray(arr_u)
                        jets = (
                            unflatten_to_jagged(arr_u, jet_counts)
                            if arr_np.ndim == 1 and jet_counts is not None
                            else arr_u
                        )

                        jtr, jte = jets[train_mask], jets[test_mask]
                        mtr, mte = mask_jet_pt[train_mask], mask_jet_pt[test_mask]
                        dtr = pad_clip_jets(jtr, max_jets)
                        dte = pad_clip_jets(jte, max_jets)

                        ensure_resizable_dataset(
                            tr_in, ["Jet", var], cast_floats32(dtr), shuffle
                        )
                        ensure_resizable_dataset(
                            te_in, ["Jet", var], cast_floats32(dte), shuffle
                        )

                        if not jet_mask_written:
                            ensure_resizable_dataset(
                                tr_in, ["Jet", "MASK"], mtr, shuffle
                            )
                            ensure_resizable_dataset(
                                te_in, ["Jet", "MASK"], mte, shuffle
                            )
                            jet_mask_written = True
                    elif is_global:
                        arr_ev = to_numpy_event_vector(arr_u)
                        write_block_split(
                            tr_in,
                            te_in,
                            [coll, var],
                            cast_floats32(arr_ev),
                            train_mask,
                            test_mask,
                            shuffle,
                        )
                if "GluGlu" in dataset:
                    kl_val = extract_kl_value(dataset)
                    kl_val_array = kl_val * ak.ones_like(to_numpy_event_vector(payload[weight_name]))
                    write_block_split(
                        tr_in,
                        te_in,
                        ["Event", "kl"],
                        cast_floats32(kl_val_array),
                        train_mask,
                        test_mask,
                        shuffle,
                    )

    print(f"Wrote: {h5_tr}, {h5_te}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser("coffea → HDF5 converter")

    p.add_argument("-i", "--input", required=True, help="Input coffea file path")
    p.add_argument("-o", "--output", required=True, help="Output HDF5 file path")
    p.add_argument(
        "-r",
        "--regions",
        nargs="+",
        default=["2b_signal_region_postW", "4b_signal_region"],
        help="Regions to use for each class label",
    )

    p.add_argument(
        "-cl",
        "--class-labels",
        nargs="+",
        default=["DATA", "GluGlu"],
        help="Class labels to use for classification",
    )

    p.add_argument("-m", "--max-jets", type=int, default=5, help="Max jets to keep")
    p.add_argument(
        "-tf", "--train-frac", type=float, default=0.8, help="Train fraction"
    )
    p.add_argument(
        "-ns", "--no-shuffle", action="store_true", help="Disable data shuffling"
    )

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    coffea_to_h5(
        coffea_path=args.input,
        h5_path=args.output,
        regions=args.regions,
        class_labels=args.class_labels,
        max_jets=args.max_jets,
        train_frac=args.train_frac,
        do_data_shuffling=not args.no_shuffle,
    )
