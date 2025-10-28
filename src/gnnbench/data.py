from cgi import test
from typing import Tuple, Dict, Optional
import os
import numpy as np
from numpy._typing import _64Bit
import torch
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.transforms import ToUndirected
from torch_geometric.utils import is_undirected
from torch_geometric.datasets import (
    Planetoid, WebKB, WikipediaNetwork, Actor, GitHub, DeezerEurope, LastFMAsia,
    Twitch, EllipticBitcoinDataset
)

# BUILT IN DATASET (PyG)
def _as_single_graph(dataset):
    data = dataset[0]
    if not hasattr(data, "train_mask"):
        # masks may be absent, will be created by make_masks
        pass
    return data

def get_dataset(name: str, root: str):
    name_l = name.lower()
    if name_l in {"cora", "citeseer", "pubmed"}:
        return Planetoid(root, name_l.capitalize(), transform=ToUndirected())
    if name_l in {"cornell", "texas", "wisconsin"}:
        return WebKB(root, name_l.capitalize(), transform=ToUndirected())
    if name_l in {"chameleon", "squirrel"}:
        return WikipediaNetwork(root, name_l.capitalize(), transform=ToUndirected())
    if name_l == "actor":
        return Actor(root, transform=ToUndirected())
    if name_l == "github":
        return GitHub(root, transform=ToUndirected())
    if name_l == "deezereurope":
        return DeezerEurope(root, transform=ToUndirected())
    if name_l == "lastfmasia":
        return LastFMAsia(root, transform=ToUndirected())
    if name_l.startswith("twitch"):
        return Twitch(root, transform=ToUndirected())
    if name_l in {"elliptic", "ellipticbitcoin"}:
        return EllipticBitcoinDataset(root)
    raise ValueError(f"Unknwon dataset: {name}")

def load_dataset(name: str, root: str) -> Data:
    ds = get_dataset(name, root)
    return _as_single_graph(ds)

# Splits and basic EDA
def _ids_to_mask(ids, N):
    m = torch.zeros(N, dtype=torch.bool)
    m[torch.as_tensor(ids, dtype=torch.long)] = True
    return m

def _stratified_per_class(y, per_class, seen=None):
    rng = np.random.default_rng()
    y_np = y.cpu().numpy()
    classes = np.unique(y_np)
    mask = np.zeros_like(y_np, dtype= bool)
    for c in classes:
        idx = np.where(y_np == c)[0]
        if seen is not None:
            idx = np.setdiff1d(idx, np.where(seen)[0])
        if len(idx) == 0:
            continue
        take = min(per_class, len(idx))
        pick = rng.choice(idx, size=take, replace=False)
        mask[pick] = True
    return torch.from_numpy(mask)

def _ratio_split(n, train_ratio, val_ratio):
    rng = np.random.default_rng()
    idx = np.arrange(n)
    rng.shuffle(idx)
    n_train = int(n*train_ratio)
    n_val = int(n*val_ratio)
    train_idx = idx[:n_train]
    val_idx = idx[n_train: n_train+n_val]
    test_idx = idx[n_train+n_val:]
    return _ids_to_mask(train_idx, n), _ids_to_mask(val_idx, n), _ids_to_mask(test_idx, n)

def make_masks(
    data: Data,
    split: str="auto",
    train_per_class: int = 20,
    val_per_class: int = 30,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2
 ) -> Data:
    need = (not hasattr(data, "train_mask")) or split!= "auto"
    if not need: 
        return data
    
    N = data.num_nodes
    y = data.y

    if split == "random":
        train_mask = _stratified_per_class(y, train_per_class)
        val_mask = _stratified_per_class(y, val_per_class, seen = train_mask)
        test_mask = ~(train_mask | val_mask)
    elif split == "ratio":
        train_mask, val_mask, test_mask = _ratio_split(N, train_ratio, val_ratio)
    else: # auto but mask missing -> use random 
        train_mask = _stratified_per_class(y, train_per_class)
        val_mask = _stratified_per_class(y, val_per_class, seen = train_mask)
        test_mask = ~(train_mask | val_mask)
    
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data

def homophily(edge_index, y):
    u, v = edge_index
    return float((y[u]==y[v]).float().mean().item())

def basic_eda(data: Data) -> Dict:
    N = data.num_nodes
    undirected = is_undirected(data.edge_index)
    E = data.num_edges // (2 if undirected else 1)
    deg = torch.bincount(data.edge_index[0], minlength=N)
    cls, counts = torch.unique(data.y, return_counts = True)
    return {
        "num_nodes": int(N),
        "num_edges": int(E),
        "avg_degree": float(deg.float().mean().item()),
        "num_features": int(data.x.size(1)),
        "num_classes": int(len(cls)),
        "class_counts": {int(c): int(n) for c, n in zip(cls, counts)},
        "homophily": round(homophily(data.edge_index, data.y), 4),
        "has_masks": bool(hasattr(data, "train_mask")),
    }

# Generic load from files for external datasets
def _detect_edge_cols(df: pd.DataFrame):
    cand_src = ["src", "source", "u", "from", "node1"]
    cand_dst = ["dst", "target", "v", "to", "node2"]
    cols_lower = [c.lower() for c in df.columns]
    s = next((c for c in cand_src if c in cols_lower), None)
    t = next((c for c in cand_dst if c in cols_lower), None)
    if s is None or t is None:
        if len(df.columns) >=2:
            return df.columns[0], df.columns[1]
        raise ValueError("Cannot detect edge columsn. Expected headers like src, dst")
    s_real = next(c for c in df.columns if c.lower() ==s)
    t_real = next(c for c in df.columns if c.lower() == t)
    return s_real, t_real

def _to_index(series: pd.Series, id2idx: dict) -> np.ndarray:
    return series.map(id2idx).to_numpy()

def _load_features(path: str, N: int, id_map: Optional[dict], node_id_col: Optional[str]):
    if path is None:
        return None
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        X = np.load(path)
        assert X.shape[0] == N, f"features rows ({X.shape[0]}) != N ({N}))"
        return torch.from_numpy(X).float()
    elif ext == ".npz":
        X = np.load(path)["arr_0"]
        assert X.shape[0] == N, f"features rows ({X.shape[0]}) != N ({N}))"
        return torch.from_numpy(X).float()
    else:
        df = pd.read_csv(path)
        if node_id_col and node_id_col in df.columns:
            assert id_map is not None, "id_map is required to align by node_id_col"
            df = df.set_index(node_id_col)
            inv = {v: k for k, v in id_map.items()}
            ordered_ids = [inv[i] for i in range(N)]
            df = df.loc[ordered_ids]
        X = df.select_dtypes(include = [np.number]).to_numpy()
        assert X.shape[0]==N, f"features rows ({X.shape[0]}) != N ({N})"
        return torch.from_numpy(X).float()

def _load_labels(path: str, N: int, id_map: Optional[dict], node_id_col: Optional[str]):
    if path is None:
        return None
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        y = np.load(path)
        assert y.shape[0] == N, f"labels rows ({y.shape[0]}) != N ({N})"
        if not np.issubdtype(y.dtype, np.integer):
            uniq = {v: i for i, v in enumerate(pd.unique(y))}
            y = np.vectorize(uniq.get)(y)
        return torch.from_numpy(y).long()
    else:
        df = pd.read_csv(path)
        label_col = next((c for c in df.columns if c.lower() in {"label", "y", "class"}), None)
        if label_col is None:
            raise ValueError("Labels CSV must contain a 'label' column.")
        if node_id_col and node_id_col in df.columns:
            assert id_map is not None, "id_map is required to align labels by node_id_col"
            df = df.set_index(node_id_col)
            inv = {v: k for k, v in id_map.items()}
            ordered_ids = [inv[i] for i in range(N)]
            y = df.loc[ordered_ids][label_col].to_numpy()
        else:
            y = df[label_col].to_numpy()
            assert y.shape[0] == N, f"labels rows ({y.shape[0]}) != N ({N})"
        if not np.issubdtype(y.dtype, np.integer):
            uniq = {v: i for i, v in enumerate(pd.unique(y))}
            y = np.vectorize(uniq.get)(y)
        return torch.from_numpy(y).long()

def load_from_files(
    edges_path: str,
    features_path: Optional[str] = None,
    labels_path: Optional[str] = None,
    *,
    directed: bool = False,
    make_undirected: bool = True,
    add_self_loops: bool = False,
    node_id_col: Optional[str] = None,
    structural_degree_feature_if_missing: bool = True,
) -> Data:
    """
    Build a PyG Data object from user-provided files.

    - edges_path: CSV/TSV with two cols (e.g., src,dst). Node IDs may be strings or ints.
    - features_path: .npy/.npz/.csv (rows=N). If CSV with an ID column, pass node_id_col.
    - labels_path: .npy/.csv with 'label' column (rows=N). If CSV with IDs, pass node_id_col.
    """
    # edges
    df = pd.read_csv(edges_path, sep=None, engine="python")
    s_col, t_col = _detect_edge_cols(df)
    src_raw, dst_raw = df[s_col], df[t_col]

    if not (np.issubdtype(src_raw.dtype, np.integer) and np.issubdtype(dst_raw.dtype, np.integer)):
        nodes = pd.Index(pd.unique(pd.concat([src_raw, dst_raw], ignore_index=True)))
        id2idx = {nid: i for i, nid in enumerate(nodes)}
        row = _to_index(src_raw, id2idx)
        col = _to_index(dst_raw, id2idx)
        N = len(nodes)
    else:
        row = src_raw.to_numpy()
        col = dst_raw.to_numpy()
        N = int(max(row.max(), col.max())) + 1
        id2idx = None

    # symmetrize if needed
    if not directed and make_undirected:
        new_row = np.concatenate([row, col])
        new_col = np.concatenate([col, row])
        row, col = new_row, new_col

    edge_index = torch.as_tensor(np.vstack([row, col]), dtype=torch.long)

    # features & labels
    x = _load_features(features_path, N, id2idx, node_id_col) if features_path else None
    y = _load_labels(labels_path, N, id2idx, node_id_col) if labels_path else None

    if x is None and structural_degree_feature_if_missing:
        deg = np.bincount(edge_index[0].cpu().numpy(), minlength=N).astype(np.float32)
        deg = (deg - deg.mean()) / (deg.std() + 1e-8)
        x = torch.from_numpy(deg[:, None])

    data = Data(x=x, edge_index=edge_index)
    if y is not None:
        data.y = y

    if add_self_loops:
        loops = torch.arange(N, dtype=torch.long).unsqueeze(0).repeat(2, 1)
        data.edge_index = torch.cat([data.edge_index, loops], dim=1)

    return data