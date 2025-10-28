from cgi import test
from typing import Tuple, Dict
import os
import numpy as np
from numpy._typing import _64Bit
import torch
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