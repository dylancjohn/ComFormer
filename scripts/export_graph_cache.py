"""Build train/val/test PygStructureDataset objects locally and export them
in a portable (numpy-only) format for transfer to another machine/arch --
avoids torch.save() on whole DataLoader/Dataset objects, which pickles
torch_geometric class instances and is fragile across different
torch/torch_geometric versions (see data_to_numpy_dict's docstring in
comformer/graphs.py for why).

Mirrors comformer.data.get_train_val_loaders' exact data loading, filtering,
and splitting logic up through building the three PygStructureDataset
objects, so the exported split is bit-for-bit what a real run would use
for the same dataset/target/n_train/n_val/n_test/split_seed.

Usage:
    python scripts/export_graph_cache.py --out_dir /path/to/export

Then transfer the resulting `train.pkl`, `val.pkl`, `test.pkl` to the
target machine and load them with scripts/import_graph_cache.py.
"""
import argparse
import math
import os
import pickle
import time

import pandas as pd
from jarvis.db.figshare import data as jdata

from comformer.data import get_id_train_val_test, get_pyg_dataset
from comformer.graphs import data_to_numpy_dict


def build_filtered_dat(dataset, target):
    """Exactly mirrors get_train_val_loaders' filtering of the raw dataset
    down to rows with a usable (non-null, non-nan) target value -- the
    split is computed over this filtered list, not the raw dataset, so
    reproducing it exactly matters for the split to match."""
    d = jdata(dataset)
    dat = []
    for i in d:
        if isinstance(i[target], list):
            dat.append(i)
        elif i[target] is not None and i[target] != "na" and not math.isnan(i[target]):
            dat.append(i)
    return dat


def export_split(dataset_split, id_tag, target, mean_train, std_train, **pyg_kwargs):
    data, mean_train, std_train = get_pyg_dataset(
        dataset=dataset_split,
        id_tag=id_tag,
        target=target,
        mean_train=mean_train,
        std_train=std_train,
        **pyg_kwargs,
    )
    payload = {
        "graphs": [data_to_numpy_dict(g) for g in data.graphs],
        "df": data.df[[id_tag, target]].reset_index(drop=True),
        "mean_train": mean_train,
        "std_train": std_train,
        "id_tag": id_tag,
        "target": target,
    }
    return payload, mean_train, std_train


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--dataset", default="megnet")
    p.add_argument("--target", default="gap pbe")
    p.add_argument("--id_tag", default="id")
    p.add_argument("--n_train", type=int, default=60000)
    p.add_argument("--n_val", type=int, default=5000)
    p.add_argument("--n_test", type=int, default=4239)
    p.add_argument("--split_seed", type=int, default=123)
    p.add_argument("--atom_features", default="cgcnn")
    p.add_argument("--neighbor_strategy", default="k-nearest")
    p.add_argument("--use_canonize", action="store_true", default=True)
    p.add_argument("--cutoff", type=float, default=4.0)
    p.add_argument("--max_neighbors", type=int, default=25)
    p.add_argument("--use_lattice", action="store_true", default=True)
    p.add_argument("--use_angle", action="store_true", default=False)
    p.add_argument("--line_graph", action="store_true", default=True)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading + filtering dataset...", flush=True)
    dat = build_filtered_dat(args.dataset, args.target)
    print(f"filtered dataset size: {len(dat)}", flush=True)

    id_train, id_val, id_test = get_id_train_val_test(
        total_size=len(dat),
        split_seed=args.split_seed,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        keep_data_order=False,
    )
    dataset_train = [dat[x] for x in id_train]
    dataset_val = [dat[x] for x in id_val]
    dataset_test = [dat[x] for x in id_test]
    print(
        f"split sizes: train={len(dataset_train)} val={len(dataset_val)} "
        f"test={len(dataset_test)}",
        flush=True,
    )

    pyg_kwargs = dict(
        atom_features=args.atom_features,
        neighbor_strategy=args.neighbor_strategy,
        use_canonize=args.use_canonize,
        name=args.dataset,
        line_graph=args.line_graph,
        cutoff=args.cutoff,
        max_neighbors=args.max_neighbors,
        classification=False,
        output_dir=args.out_dir,
        use_lattice=args.use_lattice,
        use_angle=args.use_angle,
        use_save=False,
    )

    for split_name, dataset_split, needs_mean in [
        ("train", dataset_train, True),
        ("val", dataset_val, False),
        ("test", dataset_test, False),
    ]:
        print(f"=== building {split_name} ({len(dataset_split)} structures) ===", flush=True)
        t0 = time.time()
        if split_name == "train":
            payload, mean_train, std_train = export_split(
                dataset_split, args.id_tag, args.target, None, None,
                tmp_name="train_data", **pyg_kwargs,
            )
        else:
            payload, _, _ = export_split(
                dataset_split, args.id_tag, args.target, mean_train, std_train,
                tmp_name=f"{split_name}_data", **pyg_kwargs,
            )
        out_path = os.path.join(args.out_dir, f"{split_name}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(payload, f, protocol=4)
        print(
            f"{split_name}: {len(payload['graphs'])} graphs, "
            f"took {time.time()-t0:.1f}s, saved to {out_path} "
            f"({os.path.getsize(out_path)/1e6:.1f} MB)",
            flush=True,
        )

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
