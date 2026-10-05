"""
spaCy Text Pipeline – Modern Industrial-Strength NLP.
High-throughput nlp.pipe, POS tagging, NER, noun-chunk extraction.
"""
import spacy
from typing import List, Dict, Any

class SpacyPipeline:
    def __init__(self, model_name="en_core_web_sm", batch_size=1000):
        self.nlp = spacy.load(model_name)
        self.batch_size = batch_size

    def _doc_to_dict(self, doc) -> Dict[str, Any]:
        return {
            "text": doc.text,
            "tokens": [t.text for t in doc],
            "lemmas": [t.lemma_ for t in doc],
            "pos": [t.pos_ for t in doc],
            "ents": [{"text": e.text, "label": e.label_} for e in doc.ents],
            "noun_chunks": [c.text for c in doc.noun_chunks],
        }

    def process(self, texts: List[str]) -> List[Dict[str, Any]]:
        return [self._doc_to_dict(doc) for doc in self.nlp.pipe(texts, batch_size=self.batch_size)]
