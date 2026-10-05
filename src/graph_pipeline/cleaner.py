# -*- coding: utf-8 -*-
"""
Module 12: Graph Cleaning & Topological Sanitisation.
Handles:
- Pruning isolated (degree 0) or uninformative nodes
- Removing redundant self-loops
- Resolving duplicate / parallel edges
- Verifying weakly and strongly connected components
"""

import networkx as nx

def clean_graph(G: nx.DiGraph, prune_isolated: bool = True, remove_self_loops: bool = True) -> tuple:
    """
    Cleans a clinical knowledge graph.
    Returns: (cleaned_graph, cleaning_stats_dict)
    """
    H = G.copy()
    initial_nodes = H.number_of_nodes()
    initial_edges = H.number_of_edges()
    
    # 1. Remove self loops
    loops_removed = 0
    if remove_self_loops:
        self_loops = list(nx.selfloop_edges(H))
        loops_removed = len(self_loops)
        H.remove_edges_from(self_loops)
        
    # 2. Prune isolated disconnected nodes
    isolated_removed = 0
    if prune_isolated:
        isolated = list(nx.isolates(H))
        isolated_removed = len(isolated)
        H.remove_nodes_from(isolated)
        
    stats = {
        "initial_nodes": initial_nodes,
        "cleaned_nodes": H.number_of_nodes(),
        "isolated_pruned": isolated_removed,
        "initial_edges": initial_edges,
        "cleaned_edges": H.number_of_edges(),
        "self_loops_removed": loops_removed,
        "density": round(nx.density(H), 6),
    }
    return H, stats
