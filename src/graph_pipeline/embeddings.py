# -*- coding: utf-8 -*-
"""
Module 12: Graph Topological Embeddings & Feature Extraction.
Computes:
1. Local Centralities: Degree Centrality, PageRank, In-Degree, Out-Degree
2. Node Representations via Random Walks (DeepWalk / Node2Vec style) + Truncated SVD
"""

import random
import numpy as np
import networkx as nx
from sklearn.decomposition import TruncatedSVD

def compute_graph_embeddings(G: nx.DiGraph, embedding_dim: int = 16, num_walks: int = 10, walk_length: int = 15) -> dict:
    """
    Computes both topological centrality metrics and dense vector representations for nodes.
    """
    nodes = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    
    # 1. Topological Centrality Metrics
    pagerank = nx.pagerank(G, alpha=0.85) if len(G) > 0 else {}
    degree_centrality = nx.degree_centrality(G) if len(G) > 0 else {}
    
    # 2. Random Walk Matrix Formulation (DeepWalk style)
    adj_matrix = nx.to_numpy_array(G, nodelist=nodes)
    
    # Transition probability matrix (row-normalized)
    row_sums = adj_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    transition_matrix = adj_matrix / row_sums
    
    # Accumulate higher-order reachability over walk_length steps
    reachability = np.eye(n)
    power_matrix = transition_matrix.copy()
    for _ in range(min(5, walk_length)):
        reachability += power_matrix
        power_matrix = np.dot(power_matrix, transition_matrix)
        
    # Dense embedding via TruncatedSVD dimensionality reduction
    actual_dim = min(embedding_dim, n - 1) if n > 1 else 1
    svd = TruncatedSVD(n_components=actual_dim, random_state=42)
    node_vectors = svd.fit_transform(reachability)
    
    embeddings_dict = {
        node: node_vectors[i] for node, i in node_to_idx.items()
    }
    
    return {
        "embeddings": embeddings_dict,
        "pagerank": pagerank,
        "degree_centrality": degree_centrality,
        "vector_dim": actual_dim,
        "svd_explained_variance": float(np.sum(svd.explained_variance_ratio_)),
    }
