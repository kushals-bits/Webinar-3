# -*- coding: utf-8 -*-
"""
Module 12: Graph Data Representation.
Constructs a heterogeneous clinical knowledge network connecting:
Patients <-> Symptoms <-> Diseases <-> Treatments / Drugs.
"""

import json
import pandas as pd
import networkx as nx
from pathlib import Path

def build_clinical_knowledge_graph(graph_dir: str) -> nx.DiGraph:
    """
    Loads edge lists and node ontologies, returning an annotated NetworkX DiGraph.
    """
    g_dir = Path(graph_dir)
    edges_csv = g_dir / "full_clinical_graph_edges.csv"
    if not edges_csv.exists():
        edges_csv = g_dir / "medical_kg_edges.csv"
        
    df_edges = pd.read_csv(edges_csv)
    
    # Load ontology nodes
    diseases = json.load(open(g_dir / "kg_nodes_diseases.json")) if (g_dir / "kg_nodes_diseases.json").exists() else []
    symptoms = json.load(open(g_dir / "kg_nodes_symptoms.json")) if (g_dir / "kg_nodes_symptoms.json").exists() else []
    drugs = json.load(open(g_dir / "kg_nodes_drugs.json")) if (g_dir / "kg_nodes_drugs.json").exists() else []
    
    G = nx.DiGraph()
    
    # Add annotated ontology nodes
    for d in diseases:
        G.add_node(d["id"], label=d["name"], node_type="Disease", category=d.get("category"), icd10=d.get("icd10"))
    for s in symptoms:
        G.add_node(s["id"], label=s["name"], node_type="Symptom", snomed=s.get("snomed"))
    for dr in drugs:
        G.add_node(dr["id"], label=dr["name"], node_type="Drug", drug_class=dr.get("class"))
        
    # Add edges
    for _, r in df_edges.iterrows():
        u = str(r["source"])
        v = str(r["target"])
        rel = str(r["relation"])
        w = float(r.get("weight", 1.0))
        
        # If node not yet present (e.g. Patient ID), infer type
        if u not in G:
            G.add_node(u, label=u, node_type="Patient" if u.startswith("PAT") else "Entity")
        if v not in G:
            G.add_node(v, label=v, node_type="Patient" if v.startswith("PAT") else "Entity")
            
        G.add_edge(u, v, relation=rel, weight=w)
        
    return G
