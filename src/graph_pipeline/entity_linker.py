# -*- coding: utf-8 -*-
"""
Module 12: Entity Linking from NLP Clinical Text to Knowledge Graph.
Resolves symptom mentions from Module 9 patient reports into formal SNOMED/ICD nodes.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

class MedicalEntityLinker:
    """Maps extracted symptom keywords to formal Knowledge Graph ontology IDs."""
    def __init__(self, graph_dir: str):
        g_dir = Path(graph_dir)
        symptoms_file = g_dir / "kg_nodes_symptoms.json"
        
        self.symptom_alias_map = {}
        self.snomed_map = {}
        
        if symptoms_file.exists():
            symptoms = json.load(open(symptoms_file))
            for s in symptoms:
                sid = s["id"]
                name = s["name"].lower()
                self.symptom_alias_map[name] = sid
                self.snomed_map[sid] = s.get("snomed")
                
        # Domain aliases / clinical synonyms
        domain_aliases = {
            "itching": "SYM_01",
            "itch": "SYM_01",
            "mole": "SYM_02",
            "asymmetric mole": "SYM_02",
            "chest pain": "SYM_03",
            "severe chest pain": "SYM_03",
            "pus": "SYM_04",
            "pus discharge": "SYM_04",
            "shortness of breath": "SYM_05",
            "breathlessness": "SYM_05",
            "redness": "SYM_06",
            "erythema": "SYM_06",
            "palpitations": "SYM_07",
            "arrhythmia": "SYM_07",
            "ulcer": "SYM_08",
            "non-healing ulcer": "SYM_08",
        }
        self.symptom_alias_map.update(domain_aliases)

    def link_text_entities(self, text: str) -> List[Dict]:
        """Scans clinical text and returns matched Knowledge Graph nodes."""
        text_lower = text.lower()
        matched = []
        for alias, node_id in self.symptom_alias_map.items():
            if alias in text_lower:
                matched.append({
                    "matched_alias": alias,
                    "kg_node_id": node_id,
                    "snomed_code": self.snomed_map.get(node_id, "Unknown"),
                })
        return matched
