import argparse, os, csv, torch
from gnnbench import (
    seed_everything, get_device, make_gcn,
    load_dataset, load_from_files, make_masks, basic_eda,
    train_gcn
)
from gnnbench.utils import timestamp, ensure_dir, save_json

def build_parser():
    p = argparse.ArgumentParser("gnnbench-train", description="Baseline GCN training on built-in or external graphs.")
    # data: built-in
    p.add_argument("--dataset", type=str, default="DeezerEurope",
                   help="PyG dataset name (e.g., Cora, DeezerEurope, LastFMAsia, Elliptic).")
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
    p.add_argument("--norm", type=str, default="batch", choices=["batch","layer","none"])
    p.add_argument("--no-residual", dest="residual", action="store_false", help="Disable residual connections.")
    p.set_defaults(residual=True)

    # training
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda"])

    # output
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--save_model", action="store_true", help="Save model state_dict to results folder.")
    return p

def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    seed_everything(args.seed)
    device = get_device(args.device)

    # choose loader: external files or built-in dataset
    if args.edges:
        data = load_from_files(
            edges_path=args.edges,
            features_path=args.features,
            labels_path=args.labels,
            directed=args.directed,
            node_id_col=args.node_id_col,
        )
        if not hasattr(data, "y"):
            raise ValueError("Labels are required for supervised training. Provide --labels.")
    else:
        data = load_dataset(args.dataset, args.root)

    data = make_masks(
        data, split=args.split,
        train_per_class=args.train_per_class, val_per_class=args.val_per_class,
        train_ratio=args.train_ratio, val_ratio=args.val_ratio
    )
    eda = basic_eda(data)
    print("EDA:", eda)

    num_classes = int(data.y.max().item()) + 1
    model = make_gcn(
        num_features=data.num_features,
        num_classes=num_classes,
        hidden=args.hidden,
        dropout=args.dropout,
        num_layers=args.layers,
        residual=args.residual,
        norm=args.norm
    ).to(device)

    run_name = args.run_name or (f"{args.dataset}_{timestamp()}" if not args.edges else f"external_{timestamp()}")
    out_dir = os.path.join(args.results_dir, run_name)
    ensure_dir(out_dir)

    result = train_gcn(
        model, data.to(device),
        epochs=args.epochs, patience=args.patience,
        lr=args.lr, weight_decay=args.weight_decay,
        results_dir=out_dir, save_best=True, verbose=True
    )

    # save summary + history
    save_json({"args": vars(args), "eda": eda, **result}, os.path.join(out_dir, "summary.json"))
    hist = result.get("history", [])
    if hist:
        with open(os.path.join(out_dir, "history.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=hist[0].keys())
            writer.writeheader()
            writer.writerows(hist)

    if args.save_model:
        torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))

    print("\nBest val macro-F1:", round(result["val_best_macro_f1"], 4))
    print("Test metrics:", {k: round(v, 4) for k, v in result["test"].items()})
    print("Saved to:", out_dir)

if __name__ == "__main__":
    main()
