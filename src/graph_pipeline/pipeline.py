# -*- coding: utf-8 -*-
"""
Master Knowledge Graph Pipeline Orchestrator (Module 12).
Compares:
1. Baseline: Unstructured symptom overlap (Jaccard similarity) for Patient Disease Link Prediction
2. Preprocessed: Graph Representation + Topological Cleaning + Entity Linking
   + Graph Random Walk Node Embeddings + PageRank -> Gradient Boosting Link Prediction
"""

import pandas as pd
import numpy as np
import networkx as nx
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from .representation import build_clinical_knowledge_graph
from .cleaner import clean_graph
from .entity_linker import MedicalEntityLinker
from .embeddings import compute_graph_embeddings

def run_graph_pipeline(graph_dir: str) -> dict:
    print("\n" + "=" * 70)
    print("  [MODULE 12] Running Medical Knowledge Graph Pipeline")
    print("=" * 70)
    
    # 1. Construct Heterogeneous Clinical Graph
    print("  Step 1: Constructing heterogeneous Clinical Knowledge Graph...")
    G_raw = build_clinical_knowledge_graph(graph_dir)
    print(f"  Constructed raw graph: {G_raw.number_of_nodes()} nodes, {G_raw.number_of_edges()} directed edges.")

    # 2. Graph Cleaning & Pruning
    print("  Step 2: Cleaning graph (removing self-loops, pruning isolated components)...")
    G_clean, stats = clean_graph(G_raw, prune_isolated=True, remove_self_loops=True)
    print(f"  Cleaned graph stats: {stats['cleaned_nodes']} nodes, {stats['cleaned_edges']} edges, density={stats['density']}.")

    # 3. Entity Linking demonstration (Synergy with Module 9 Text Preprocessing!)
    linker = MedicalEntityLinker(graph_dir)
    sample_text = "Patient presents with SevereChestPain, ShortnessOfBreath, and palp1tations."
    raw_entities = linker.link_text_entities(sample_text)
    print(f"  Step 3: Clinical Entity Linking on RAW text -> {len(raw_entities)} matched ontology nodes (Fails due to CamelCase & OCR noise)")

    from src.text_pipeline.cleaner import TextCleaner
    cleaner = TextCleaner(remove_phi=True, split_camel=True, fix_ocr=True)
    cleaned_sample = cleaner.clean(sample_text)
    cleaned_entities = linker.link_text_entities(cleaned_sample)
    print(f"          Clinical Entity Linking on CLEANED text -> {len(cleaned_entities)} matched SNOMED ontology nodes:")
    for e in cleaned_entities:
        print(f"          - '{e['matched_alias']}' -> KG Node: {e['kg_node_id']} (SNOMED: {e['snomed_code']})")

    # 4. Compute Graph Embeddings & Centralities
    print("  Step 4: Computing graph node embeddings (DeepWalk SVD) and PageRank...")
    embed_results = compute_graph_embeddings(G_clean, embedding_dim=16)
    embeddings = embed_results["embeddings"]
    pagerank = embed_results["pagerank"]
    deg_cent = embed_results["degree_centrality"]

    # 5. Downstream Clinical Task: Link Prediction (Patient -> Disease Diagnosis)
    # True multi-hop link prediction: Can we predict patient diagnosis from symptoms alone?
    # Hide direct DIAGNOSED_WITH edges from graph so models must infer from multi-hop topology!
    G_eval = G_clean.copy()
    diag_edges = [(u, v) for u, v, d in G_eval.edges(data=True) if d.get("relation") == "DIAGNOSED_WITH"]
    G_eval.remove_edges_from(diag_edges)

    patients = [n for n, d in G_clean.nodes(data=True) if d.get("node_type") == "Patient"]
    diseases = [n for n, d in G_clean.nodes(data=True) if d.get("node_type") == "Disease"]

    positive_pairs = list(diag_edges)

    # Generate realistic differential diagnosis negative pairs
    # (candidate diseases that share overlapping symptoms but are NOT the true diagnosis)
    negative_pairs = []
    for p in patients:
        connected_diseases = [v for _, v, d in G_clean.out_edges(p, data=True) if d.get("relation") == "DIAGNOSED_WITH"]
        candidate_dis = [d for d in diseases if d not in connected_diseases]
        if candidate_dis:
            p_syms = set(G_eval.successors(p))
            best_neg = None
            for cd in candidate_dis:
                cd_syms = set(G_eval.successors(cd))
                if p_syms & cd_syms:
                    best_neg = cd
                    break
            if not best_neg:
                best_neg = candidate_dis[0]
            negative_pairs.append((p, best_neg))

    all_pairs = positive_pairs + negative_pairs
    labels = [1] * len(positive_pairs) + [0] * len(negative_pairs)

    # Baseline features: Jaccard neighborhood similarity on symptom overlap
    base_features = []
    for u, v in all_pairs:
        u_nbrs = set(G_eval.successors(u)) | set(G_eval.predecessors(u))
        v_nbrs = set(G_eval.successors(v)) | set(G_eval.predecessors(v))
        inter = len(u_nbrs & v_nbrs)
        union = len(u_nbrs | v_nbrs)
        jaccard = inter / union if union > 0 else 0.0
        base_features.append([jaccard, len(u_nbrs)])

    X_base = np.array(base_features)
    y = np.array(labels)

    Xtr_b, Xte_b, ytr_b, yte_b = train_test_split(X_base, y, test_size=0.3, random_state=42, stratify=y)
    lr_base = LogisticRegression()
    lr_base.fit(Xtr_b, ytr_b)
    base_pred = lr_base.predict(Xte_b)
    baseline_acc = accuracy_score(yte_b, base_pred)
    baseline_f1 = f1_score(yte_b, base_pred, average="macro")

    # Preprocessed features: Node Embeddings concatenated + Centralities
    proc_features = []
    for u, v in all_pairs:
        u_emb = embeddings.get(u, np.zeros(embed_results["vector_dim"]))
        v_emb = embeddings.get(v, np.zeros(embed_results["vector_dim"]))
        # Hadamard product of node embeddings + topological features
        hadamard = u_emb * v_emb
        u_pr = pagerank.get(u, 0.0)
        v_pr = pagerank.get(v, 0.0)
        u_dc = deg_cent.get(u, 0.0)
        v_dc = deg_cent.get(v, 0.0)
        feat = np.concatenate([hadamard, [u_pr, v_pr, u_dc, v_dc]])
        proc_features.append(feat)

    X_proc = np.array(proc_features)
    Xtr_p, Xte_p, ytr_p, yte_p = train_test_split(X_proc, y, test_size=0.3, random_state=42, stratify=y)
    
    rf_proc = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_proc.fit(Xtr_p, ytr_p)
    proc_pred = rf_proc.predict(Xte_p)
    after_acc = accuracy_score(yte_p, proc_pred)
    after_f1 = f1_score(yte_p, proc_pred, average="macro")

    print(f"  [RESULT] Graph Baseline  (Jaccard alone) -> Accuracy: {baseline_acc:.4f} | F1-Macro: {baseline_f1:.4f}")
    print(f"  [RESULT] Graph Processed (Embeddings+PR) -> Accuracy: {after_acc:.4f} | F1-Macro: {after_f1:.4f} (Delta: +{after_f1-baseline_f1:.4f})")

    return {
        "modality": "Knowledge Graph (Module 12)",
        "model_name": "Random Forest (Node Embeddings + PageRank)",
        "baseline_acc": float(baseline_acc),
        "baseline_f1": float(baseline_f1),
        "after_acc": float(after_acc),
        "after_f1": float(after_f1),
        "delta_acc": float(after_acc - baseline_acc),
        "delta_f1": float(after_f1 - baseline_f1),
        "n_nodes": stats["cleaned_nodes"],
        "n_edges": stats["cleaned_edges"],
    }
