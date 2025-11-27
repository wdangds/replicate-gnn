# inductive_bench/graph_features.py

from typing import List, Tuple
import numpy as np
import pandas as pd
import networkx as nx


def compute_graph_features(
    node_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    undirected: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Compute graph-based features for each node and return:
      - updated node_df with new columns
      - list of new feature column names

    Features (per node):
      1. deg              : degree
      2. log_deg          : log(1 + degree)
      3. clustering       : local clustering coefficient
      4. pagerank         : PageRank score (alpha=0.85)
      5. core_number      : k-core number
      6. triangles        : number of triangles node participates in
      7. eig_centrality   : eigenvector centrality
    """

    # build graph
    if undirected:
        G = nx.from_pandas_edgelist(edge_df, source="source", target="target")
    else:
        G = nx.from_pandas_edgelist(edge_df, source="source", target="target",
                                    create_using=nx.DiGraph())

    # make sure all nodes exist in the graph
    G.add_nodes_from(node_df.index.tolist())

    # 1. degree
    deg_dict = dict(G.degree())
    node_df["deg"] = node_df.index.to_series().map(deg_dict).astype(float)

    # 2. clustering coefficient
    clustering_dict = nx.clustering(G)
    node_df["clustering"] = node_df.index.to_series().map(clustering_dict).astype(float)

    # 3. PageRank
    pagerank_dict = nx.pagerank(G, alpha=0.85)
    node_df["pagerank"] = node_df.index.to_series().map(pagerank_dict).astype(float)

    # 4. core number
    core_dict = nx.core_number(G)
    node_df["core_number"] = node_df.index.to_series().map(core_dict).astype(float)

    # 5. triangles
    tri_dict = nx.triangles(G)
    node_df["triangles"] = node_df.index.to_series().map(tri_dict).astype(float)

    graph_cols = [
        "deg",
        "clustering",
        "pagerank",
        "core_number",
        "triangles",
    ]
    return node_df, graph_cols
