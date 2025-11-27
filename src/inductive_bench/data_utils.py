# inductive_bench/data_utils.py

from typing import Tuple
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import torch
from torch_geometric.data import Data

def stratified_split_nodes(
    node_df: pd.DataFrame,
    train_frac: float,
    label_col: str = "label_id",
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified split of node ids into train/val/test
    Val and test split the remanining nodes equally.
    """
    # train vs temp (val+test)
    all_nodes = node_df.index.to_numpy()
    y = node_df[label_col].values

    # train vs temp
    train_nodes, temp_nodes, y_train, y_temp = train_test_split(
        all_nodes,
        y, 
        test_size = 1.0 - train_frac,
        random_state = random_state,
        stratify = y
    )

    val_nodes, test_nodes, y_val, y_test = train_test_split(
        temp_nodes,
        y_temp,
        test_size = 0.7, 
        random_state = random_state,
        stratify = y_temp,
    )
    return train_nodes, val_nodes, test_nodes

def make_subgraph(
    node_ids, 
    node_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    feature_cols,
) -> Data:
    """
    Build an induced subgraph Data(x, edge_index, y) for the given node_ids.
    """
    node_ids = pd.Index(node_ids)
    node_set = set(node_ids)

    sub_edges = edge_df[
        edge_df['source'].isin(node_set) & edge_df['target'].isin(node_set)
    ].copy()

    # map old node ids -> [0..n-1]
    id_map = {old_id: i for i, old_id in enumerate(node_ids)}
    sub_edges["source_new"] = sub_edges["source"].map(id_map)
    sub_edges["target_new"] = sub_edges["target"].map(id_map)

    edge_index = torch.tensor(
        sub_edges[["source_new", "target_new"]].values.T,
        dtype=torch.long,
    )
    x = torch.tensor(
        node_df.loc[node_ids, feature_cols].values,
        dtype=torch.float,
    )
    y = torch.tensor(
        node_df.loc[node_ids, "label_id"].values,
        dtype=torch.long,
    )
    return Data(x=x, edge_index=edge_index, y=y)