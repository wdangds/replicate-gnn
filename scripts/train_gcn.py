#!/usr/bin/env python
import argparse, os, torch
from gnnbench import (
    ExperimentConfig, seed_everything, get_device,
    load_dataset, make_masks, basic_eda, make_gcn, train_gcn,
    timestamp, ensure_dir, save_json
)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="DeezerEurope")
    p.add_argument("--root", type=str, default="data")
    p.add_argument("--split", type=str, default="auto", choices=["auto","random","ratio"])
    p.add_argument("--train_per_class", type=int, default=20)
    p.add_argument("--val_per_class", type=int, default=30)
    p.add_argument("--train_ratio", type=float, default=0.6)
    p.add_argument("--val_ratio", type=float, default=0.2)

    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.5)

    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda"])

    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--run_name", type=str, default=None)
    args = p.parse_args()

    seed_everything(args.seed)
    device = get_device(args.device)

    data = load_dataset(args.dataset, args.root)
    data = make_masks(data, split=args.split,
                      train_per_class=args.train_per_class,
                      val_per_class=args.val_per_class,
                      train_ratio=args.train_ratio,
                      val_ratio=args.val_ratio)
    eda = basic_eda(data)
    print("EDA:", eda)

    model = make_gcn(data.num_features, int(data.y.max().item()) + 1,
                     hidden=args.hidden, dropout=args.dropout).to(device)

    # organize results subfolder per run
    run_name = args.run_name or f"{args.dataset}_{timestamp()}"
    out_dir = os.path.join(args.results_dir, run_name)
    ensure_dir(out_dir)

    result = train_gcn(
        model, data.to(device),
        epochs=args.epochs, patience=args.patience,
        lr=args.lr, weight_decay=args.weight_decay,
        results_dir=out_dir, save_best=True, verbose=True
    )

    save_json({"args": vars(args), "eda": eda, **result}, os.path.join(out_dir, "summary.json"))
    print("\nBest val macro-F1:", round(result["val_best_macro_f1"], 4))
    print("Test metrics:", {k: round(v, 4) for k, v in result["test"].items()})
    print("Saved to:", out_dir)

if __name__ == "__main__":
    main()
