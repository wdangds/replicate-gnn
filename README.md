# gnnbench

Tiny, reproducible **baseline GCN** benchmarking toolkit (PyTorch Geometric) for **node classification** with:

* configurable depth (**1–10 layers**), residuals, and normalization
* plug-and-play **built-in datasets** (Cora, WebKB, DeezerEurope, LastFMAsia, Elliptic, …)
* a **generic file loader** for **your own graphs** (CSV/NPY)
* one-command CLI + simple Python API
* auto-saved **EDA + metrics + training history**

---

## TL;DR

```bash
# install in editable mode
pip install -e .

# run a baseline on Cora (uses canonical Planetoid masks)
gnnbench-train --dataset Cora --root data --layers 2

# run on your own files
gnnbench-train \
  --edges data/my_edges.csv \
  --features data/my_features.npy \
  --labels data/my_labels.csv \
  --node_id_col node_id \
  --split ratio --train_ratio 0.6 --val_ratio 0.2 \
  --layers 4 --norm batch --dropout 0.6
```

Artifacts (EDA, metrics, history) land in `results/<run_name>/`.

---

## Why this exists

You replicated a baseline GCN. This package turns that setup into a **small, reusable tool** so you can:

* swap datasets quickly,
* **sweep depth** (1–10) and study over-smoothing,
* keep results tidy and reproducible.

---

## Repo layout

```
.
├─ src/gnnbench/
│  ├─ __init__.py
│  ├─ cli.py                 # CLI entry (python -m gnnbench.cli)
│  ├─ config.py              # (optional) experiment dataclass
│  ├─ data.py                # built-ins + generic loader
│  ├─ metrics.py
│  ├─ models.py              # DeepGCN (1–10 layers, residuals, norm)
│  ├─ train.py               # training loop + early stopping + eval
│  └─ utils.py
├─ scripts/
│  └─ train_gcn.py           # optional 3-line wrapper that calls gnnbench.cli
├─ data/                     # downloaded datasets / your csv/npy files
├─ results/                  # run outputs (created automatically)
├─ pyproject.toml
└─ README.md                 # this file
```

---

## Installation

```bash
# from repo root (inside your venv)
pip install -e .
```

> PyTorch Geometric requires a compatible PyTorch; if you already use PyG, you’re good. Otherwise install PyTorch first, then this.

---

## Command line usage

After `pip install -e .`, you get a console command:

```bash
gnnbench-train --help
```

### Common examples

**Cora (classic 2-layer GCN):**

```bash
gnnbench-train --dataset Cora --root data --layers 2
```

**Random per-class split (20/30):**

```bash
gnnbench-train --dataset Cora --split random --train_per_class 20 --val_per_class 30 --layers 2
```

**Ratio split (60/20/20) and deeper model:**

```bash
gnnbench-train --dataset Cora --split ratio --train_ratio 0.6 --val_ratio 0.2 \
  --layers 6 --norm batch --dropout 0.6 --patience 100
```

**Save trained weights:**

```bash
gnnbench-train --dataset Cora --save_model
```

### External files (your own graph)

Minimum:

```bash
gnnbench-train \
  --edges data/my_edges.csv \
  --features data/my_features.npy \
  --labels data/my_labels.csv \
  --node_id_col node_id \
  --layers 4
```

* `--edges` **CSV/TSV** with at least two columns (source/target). Column names are flexible:

  * detected from {`src`,`source`,`u`,`from`,`node1`} × {`dst`,`target`,`v`,`to`,`node2`}, or the **first two columns**.
  * Node IDs may be **ints or strings**; the loader maps them internally to `[0..N-1]`.
* `--features` `.npy` / `.npz` (shape `N×F`) or `.csv` (numeric columns). If your CSV has an ID column, pass `--node_id_col`.
* `--labels` `.npy` (shape `N,`) or `.csv` with a `label` column (strings auto-mapped to ints). If CSV has an ID column, pass `--node_id_col`.
* If you omit `--features`, the loader falls back to a **1D normalized degree feature** so you can still run.

### Key flags at a glance

* **Data choice**: `--dataset NAME` (built-in) **or** `--edges/--features/--labels` (external)
* **Splits**: `--split auto|random|ratio` (+ per-class or ratio args)
* **Model**: `--layers 1..10`, `--hidden 64`, `--dropout 0.5`, `--norm batch|layer|none`, `--no-residual` (to disable)
* **Training**: `--epochs 500`, `--patience 50`, `--lr 0.01`, `--weight_decay 5e-4`, `--seed 42`, `--device auto|cpu|cuda`
* **Outputs**: `--results_dir results`, `--run_name NAME`, `--save_model`

---

## Python API

```python
from gnnbench import (
    load_dataset, load_from_files, make_masks, basic_eda,
    make_gcn, train_gcn, seed_everything, get_device
)

seed_everything(42)
data = load_dataset("Cora", root="data")           # or: load_from_files(edges, features, labels, node_id_col="id")
data = make_masks(data, split="auto")              # uses Planetoid masks if present

print(basic_eda(data))                             # dict: nodes, edges, classes, homophily, ...

device = get_device("auto")
model = make_gcn(
    num_features=data.num_features,
    num_classes=int(data.y.max()) + 1,
    hidden=64, dropout=0.5, num_layers=2, norm="batch", residual=True
).to(device)

result = train_gcn(
    model, data.to(device),
    epochs=300, patience=40, lr=0.01, weight_decay=5e-4,
    results_dir="results/Cora_manual"
)

print("Best val macro-F1:", result["val_best_macro_f1"])
print("Test:", result["test"])
```

---

## What gets saved

Each run creates `results/<run_name>/` with:

* `summary.json` — args, EDA, best val metric, final test metrics
* `history.csv` — per-epoch validation metrics (loss, acc, macro_f1)
* `model.pt` — (if `--save_model`) `state_dict()` of the best model

You can set the folder explicitly with `--run_name`.

---

## Built-in datasets (via PyG)

* Planetoid: `Cora`, `CiteSeer`, `PubMed`
* WebKB: `Cornell`, `Texas`, `Wisconsin`
* WikipediaNetwork: `Chameleon`, `Squirrel`
* Others: `Actor`, `GitHub`, `DeezerEurope`, `LastFMAsia`, `Twitch`, `EllipticBitcoin`

> Add more by extending `get_dataset` in `src/gnnbench/data.py`.

---

## Model details

**DeepGCN** (in `models.py`) with:

* **GCNConv** layers, depth `num_layers ∈ [1,10]`
* residual connections between hidden layers (toggle with `--no-residual`)
* per-layer normalization: `--norm batch|layer|none`
* dropout on hidden layers

Tips for depth > 4:

* keep residuals on, use BatchNorm/LayerNorm
* raise dropout (0.6–0.7) and sometimes lower LR (e.g., 0.005)
* tiny graphs (e.g., WebKB) can degrade when too deep — that’s the point of the experiment 

---

## EDA metrics (printed to console & saved)

* `num_nodes`, `num_edges`, `avg_degree`, `num_features`
* `num_classes`, `class_counts`
* **edge homophily** (fraction of edges connecting same label)
* `has_masks` (whether dataset provided masks)

---

## Reproducibility

* Global seeding via `--seed` (or `seed_everything()`).
* Keep the **split type** fixed across runs when comparing models.
* Results depend on masks; for fair comparisons across methods, reuse the same masks.

---

## Troubleshooting

* **“permission denied: scripts/train_gcn.py”**
  Run with `python scripts/train_gcn.py …` or `chmod +x scripts/train_gcn.py`.

* **No GPU / CUDA**
  Use `--device cpu` (default auto already picks CPU if GPU isn’t available).

---

## Roadmap / ideas

* Optional **PairNorm** / **Jumping Knowledge** toggles for deeper GCNs
* Temporal splits (e.g., Elliptic) helper
* Extra baselines (MLP / LINK) for quick ablations
* TensorBoard logging

---

## License

MIT.

---

## Citation

If you use this for a class report or paper, please cite PyTorch, PyTorch Geometric, and the original dataset sources.
