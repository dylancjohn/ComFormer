"""Load the portable graph cache produced by scripts/export_graph_cache.py
(built on a different machine/arch) and train directly from it, skipping
graph construction entirely.

Reconstructs the same PygStructureDataset + DataLoader objects that
comformer.data.get_train_val_loaders would have built, then calls
train_main(config, train_val_test_loaders=...) -- an existing hook in
train.py that skips get_train_val_loaders (and therefore the expensive
k=25 neighbor-search graph construction) when loaders are passed in
directly.

Usage:
    python scripts/import_graph_cache_and_train.py --cache_dir /path/to/cache --output_dir $SCRATCHDIR/comformer_mp_bandgap
"""
import argparse
import os
import pickle
from functools import partial

import pandas as pd
from torch.utils.data import DataLoader

from comformer.graphs import PygStructureDataset, numpy_dict_to_data
from comformer.train import train_main


def load_split(cache_dir, split_name, atom_features, line_graph, mean_train=None, std_train=None):
    path = os.path.join(cache_dir, f"{split_name}.pkl")
    with open(path, "rb") as f:
        payload = pickle.load(f)
    graphs = [numpy_dict_to_data(g) for g in payload["graphs"]]
    df = payload["df"]
    data = PygStructureDataset(
        df,
        graphs,
        target=payload["target"],
        atom_features=atom_features,
        line_graph=line_graph,
        id_tag=payload["id_tag"],
        classification=False,
        mean_train=mean_train if mean_train is not None else payload["mean_train"],
        std_train=std_train if std_train is not None else payload["std_train"],
        # These graphs already went through this exact remap once, when
        # export_graph_cache.py built them via get_pyg_dataset() -- g.x
        # is already the CGCNN feature vector, not a raw atomic number.
        already_featurized=True,
    )
    return data, payload["mean_train"], payload["std_train"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--atom_features", default="cgcnn")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--learning_rate", type=float, default=0.0005)
    p.add_argument("--wandb_project", default="comformer")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_run_name", default="icomformer-mp-bandgap")
    p.add_argument("--no_wandb", action="store_true")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    line_graph = True

    print("Loading train split from cache...", flush=True)
    train_data, mean_train, std_train = load_split(
        args.cache_dir, "train", args.atom_features, line_graph
    )
    print("Loading val split from cache...", flush=True)
    val_data, _, _ = load_split(
        args.cache_dir, "val", args.atom_features, line_graph, mean_train, std_train
    )
    print("Loading test split from cache...", flush=True)
    test_data, _, _ = load_split(
        args.cache_dir, "test", args.atom_features, line_graph, mean_train, std_train
    )
    print(
        f"n_train={len(train_data)} n_val={len(val_data)} n_test={len(test_data)}",
        flush=True,
    )

    # Exactly matches comformer.data.get_train_val_loaders' DataLoader
    # construction (collate_fn choice, drop_last, shuffle, etc.).
    collate_fn = train_data.collate
    if line_graph:
        collate_fn = train_data.collate_line_graph

    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, drop_last=True,
        num_workers=args.num_workers, pin_memory=False,
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, drop_last=False,
        num_workers=args.num_workers, pin_memory=False,
    )
    test_loader = DataLoader(
        test_data, batch_size=1, shuffle=False,
        collate_fn=collate_fn, drop_last=False,
        num_workers=args.num_workers, pin_memory=False,
    )
    prepare_batch = partial(train_data.prepare_batch)

    config = {
        "dataset": "megnet", "target": "gap pbe", "id_tag": "id",
        "epochs": args.epochs, "batch_size": args.batch_size,
        "weight_decay": 1e-05, "learning_rate": args.learning_rate,
        "criterion": "l1", "optimizer": "adamw", "scheduler": "polynomial",
        "pin_memory": False, "write_predictions": True,
        "num_workers": args.num_workers, "atom_features": args.atom_features,
        "cutoff": 4.0, "max_neighbors": 25, "pyg_input": True,
        "use_lattice": True, "use_angle": False,
        "output_dir": args.output_dir,
        "model": {"name": "iComformer"},
        "log_wandb": not args.no_wandb,
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
        "wandb_run_name": args.wandb_run_name,
    }

    train_main(
        config,
        train_val_test_loaders=(
            train_loader, val_loader, test_loader, prepare_batch,
            mean_train, std_train,
        ),
    )


if __name__ == "__main__":
    main()
