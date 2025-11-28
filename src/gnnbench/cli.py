import argparse, os, csv, torch, copy, itertools, hashlib, json
from typing import Dict, Any, List, Iterable
from gnnbench import (
    seed_everything, get_device, make_gcn,
    load_dataset, load_from_files, make_masks, basic_eda,
    train_gcn
)
from gnnbench.utils import timestamp, ensure_dir, save_json

# Helpers
def _short_cfg_id(d: Dict[str, Any], length: int=8) -> str:
    """
    Stable short id from a dict (order-insensitive)
    """
    s = json.dumps(d, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(s.encode()).hexdigest()[:length]

def _grid_from_json_str(grid_str: str) -> Dict[str, List[Any]]:
    f"""
    Parse a JSON dict that maps param names -> list of values
    Example: '{"layers":[2,4], "hidden":[16,64], "dropout": [0.5]}'
    """
    try:
        g = json.loads(grid_str)
        if not isinstance(g, dict):
            raise ValueError("Grid must be a JSON object mapping param -> list")
        for k, v in g.items():
            if not isinstance(v, list):
                raise ValueError(f"grid['{k}'] must be a list")
            return g
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON for --gird: {e}")

def _product_dict(grid: Dict[str, List[Any]]) -> Iterable[Dict[str, Any]]:
    """
    ittertools.product over dict of lists -> list of dict combos
    """
    if not grid:
        yield {}
        return
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    for combo in itertools.product(*vals):
        yield {k: v for k, v in zip(keys,combo)}

def _apply_overrides(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str,Any]:
    out = copy.deepcopy(base)
    out.update(overrides)
    return out

# Argparser
def build_parser():
    p = argparse.ArgumentParser("gnnbench-train", description="Baseline GCN training on built-in or external graphs.")
    # data: built-in
    p.add_argument("--dataset", type=str, default="Cora",
                   help="PyG dataset name (e.g., Cora, DeezerEurope, LastFMAsia, Elliptic).")
    p.add_argument("--datasets", nargs="+", default=None,
                    help="Run the same configuration across multiple built-in datasets (space-separated).")
    p.add_argument("--root", type=str, default="data", help="Root folder for built-in datasets.")

    # data: external files
    p.add_argument("--edges", type=str, default=None, help="Edge list CSV/TSV (cols: src,dst or similar).")
    p.add_argument("--features", type=str, default=None, help="Features (.npy/.npz/.csv).")
    p.add_argument("--labels", type=str, default=None, help="Labels (.npy or .csv with a 'label' column).")
    p.add_argument("--node_id_col", type=str, default=None, help="ID column to align features/labels CSV to nodes.")
    p.add_argument("--directed", action="store_true", help="Treat edges as directed (no symmetrization).")

    # splits
    p.add_argument("--split", type=str, default="auto", choices=["auto","random","ratio"],
                   help="How to create masks if dataset doesn't include them.")
    p.add_argument("--train_per_class", type=int, default=20, help="If split='random'.")
    p.add_argument("--val_per_class", type=int, default=30, help="If split='random'.")
    p.add_argument("--train_ratio", type=float, default=0.6, help="If split='ratio'.")
    p.add_argument("--val_ratio", type=float, default=0.2, help="If split='ratio'.")

    # model
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--layers", type=int, default=2, help="Total GCN layers (1..10).")
    p.add_argument("--norm", type=str, default="none", choices=["batch","layer","none"])
    p.add_argument("--no-residual", dest="residual", action="store_false", help="Disable residual connections.")
    p.set_defaults(residual=True)

    # training
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="Run multiple random seeds (space-separated).")
    p.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda"])

    # Sweep controls
    p.add_argument("--grid", type=str, default=None,
                   help="JSON dict of hyperparameter lists to grid-search, "
                        "e.g. '{\"layers\":[2,4,6],\"hidden\":[16,64],\"dropout\":[0.5]}'")
    
    # (Optional) narrow which params are allowed in grid (to catch typos)
    p.add_argument("--allow_grid_keys", nargs="*", default=None,
                   help="If provided, restrict grid keys to this set (helps catch typos).")
    
    # output
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--save_model", action="store_true", help="Save model state_dict to results folder.")
    return p

# core runner
def run_single(args_ns, dataset_name: str, overrides: Dict[str, Any], run_suffix: str = "") -> Dict[str, Any]:
    """
    Run one experiment on one dataset (built-in or external).
    `overrides` can include any CLI arg you want to change for this run.
    """
    # materialize args for this run
    args = copy.deepcopy(vars(args_ns))
    args.update(overrides)
    device = get_device(args["device"])

    # choose data source
    if args["edges"]:
        data = load_from_files(
            edges_path=args["edges"],
            features_path=args["features"],
            labels_path=args["labels"],
            directed=args["directed"],
            node_id_col=args["node_id_col"],
        )
        if not hasattr(data, "y"):
            raise ValueError("Labels are required for supervised training. Provide --labels.")
        dataset_label = dataset_name or "external"
    else:
        data = load_dataset(dataset_name or args["dataset"], args["root"])
        dataset_label = dataset_name or args["dataset"]

    # masks & EDA
    data = make_masks(
        data, split=args["split"],
        train_per_class=args["train_per_class"], val_per_class=args["val_per_class"],
        train_ratio=args["train_ratio"], val_ratio=args["val_ratio"]
    )
    eda = basic_eda(data)
    print(f"\n=== DATASET: {dataset_label} | EDA: {eda}")

    num_classes = int(data.y.max().item()) + 1
    model = make_gcn(
        num_features=data.num_features,
        num_classes=num_classes,
        hidden=args["hidden"],
        dropout=args["dropout"],
        num_layers=args["layers"],
        residual=args["residual"],
        norm=args["norm"]
    ).to(device)

    # organize output
    base_name = args["run_name"] or f"{dataset_label}_{timestamp()}"
    if run_suffix:
        base_name = f"{base_name}_{run_suffix}"
    out_dir = os.path.join(args["results_dir"], base_name)
    ensure_dir(out_dir)

    result = train_gcn(
        model, data.to(device),
        epochs=args["epochs"], patience=args["patience"],
        lr=args["lr"], weight_decay=args["weight_decay"],
        results_dir=out_dir, save_best=True, verbose=True
    )

    # save summary + history
    summary = {"dataset": dataset_label, "args": args, "eda": eda, **result}
    save_json(summary, os.path.join(out_dir, "summary.json"))

    hist = result.get("history", [])
    if hist:
        with open(os.path.join(out_dir, "history.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=hist[0].keys())
            writer.writeheader()
            writer.writerows(hist)

    if args["save_model"]:
        torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))

    print("→ Best val macro-F1:", round(result["val_best_macro_f1"], 4))
    print("→ Test metrics:", {k: round(v, 4) for k, v in result["test"].items()})
    print("→ Saved:", out_dir)
    return summary

# entrypoint
def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

        # detect sweep mode
    multi_datasets = args.datasets is not None and len(args.datasets) > 0
    multi_seeds = args.seeds is not None and len(args.seeds) > 1
    has_grid = args.grid is not None

    # validate grid keys if provided
    if has_grid:
        grid = _grid_from_json_str(args.grid)
        if args.allow_grid_keys is not None:
            illegal = [k for k in grid.keys() if k not in set(args.allow_grid_keys)]
            if illegal:
                raise ValueError(f"Grid contains keys not in allowlist: {illegal}")
    else:
        grid = {}

    # If nothing multi, run a single job (backwards compatible)
    if not (multi_datasets or multi_seeds or has_grid):
        seed_everything(args.seed)
        return run_single(args, dataset_name=args.dataset, overrides={}, run_suffix="")

    # otherwise: SWEEP
    # list of datasets to iterate
    datasets = args.datasets if multi_datasets else [args.dataset]

    # list of seeds to iterate
    seeds = args.seeds if multi_seeds else [args.seed]

    # combinations of hyperparams from grid (empty grid -> [{}])
    combos = list(_product_dict(grid))
    if not combos:
        combos = [{}]

    # sweep index log
    sweep_root = os.path.join(args.results_dir, f"sweep_{timestamp()}")
    ensure_dir(sweep_root)
    index_rows = []

    run_idx = 0
    for ds in datasets:
        for combo in combos:
            for sd in seeds:
                run_idx += 1
                # overrides for this run
                overrides = dict(combo)
                overrides["seed"] = sd
                # nice suffix for folder name
                cfg_id = _short_cfg_id({"dataset": ds, "combo": combo, "seed": sd})
                suffix = f"{ds}_cfg{cfg_id}_seed{sd}"
                # print a header
                print(f"\n##### RUN {run_idx}: dataset={ds}, seed={sd}, combo={combo} #####")
                seed_everything(sd)
                out = run_single(args, dataset_name=ds, overrides=overrides, run_suffix=suffix)

                # record in sweep index
                idx_row = {
                    "run_idx": run_idx,
                    "dataset": ds,
                    "seed": sd,
                    "cfg_id": cfg_id,
                    "out_dir": out["args"]["results_dir"],  # base results dir
                    "layers": out["args"]["layers"],
                    "hidden": out["args"]["hidden"],
                    "dropout": out["args"]["dropout"],
                    "norm": out["args"]["norm"],
                    "residual": out["args"]["residual"],
                    "lr": out["args"]["lr"],
                    "weight_decay": out["args"]["weight_decay"],
                    "epochs": out["args"]["epochs"],
                    "patience": out["args"]["patience"],
                    "val_best_macro_f1": out["val_best_macro_f1"],
                    "test_acc": out["test"]["acc"],
                    "test_macro_f1": out["test"]["macro_f1"],
                }
                # add grid fields (if any)
                for k, v in combo.items():
                    idx_row[f"grid_{k}"] = v
                index_rows.append(idx_row)

    # write an index CSV for the whole sweep
    index_csv = os.path.join(sweep_root, "sweep_index.csv")
    if index_rows:
        fieldnames = list(index_rows[0].keys())
        with open(index_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(index_rows)
        print(f"\n== Sweep complete. Index: {index_csv} ==")
    else:
        print("\n== Sweep complete. (No runs?) ==")


if __name__ == "__main__":
    main()
