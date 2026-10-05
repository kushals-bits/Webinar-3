"""
NLTK Text Pipeline – Classical NLP Processing.
Sentence/word tokenisation, stopword removal (negation-aware),
Stemming (Porter/Snowball), POS-informed WordNet Lemmatisation.
"""
import string
from typing import List, Dict, Any
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer

NEGATIONS = {"not", "no", "never", "none", "cannot", "nor", "neither"}

class NLTKPipeline:
    def __init__(self, remove_stopwords=True, preserve_negations=True, use_lemmatizer=True):
        self.remove_stopwords = remove_stopwords
        self.preserve_negations = preserve_negations
        self.stop_words = set(stopwords.words("english"))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer() if use_lemmatizer else None

    @staticmethod
    def _wn_pos(tag):
        if tag.startswith("J"): return wordnet.ADJ
        if tag.startswith("V"): return wordnet.VERB
        if tag.startswith("R"): return wordnet.ADV
        return wordnet.NOUN

    def _filter(self, tokens):
        if not self.remove_stopwords:
            return tokens
        return [t for t in tokens
                if t.lower() not in self.stop_words
                or (self.preserve_negations and t.lower() in NEGATIONS)]

    def process(self, texts: List[str]) -> List[Dict[str, Any]]:
        results = []
        for txt in texts:
            sents = nltk.sent_tokenize(txt)
            tokens = []
            for s in sents:
                tokens.extend(nltk.word_tokenize(s))
            tokens = [t for t in tokens if t not in string.punctuation]
            filtered = self._filter(tokens)
            stems = [self.stemmer.stem(t) for t in filtered]
            if self.lemmatizer:
                pos_tags = nltk.pos_tag(filtered)
                lemmas = [self.lemmatizer.lemmatize(w, self._wn_pos(p)) for w, p in pos_tags]
            else:
                lemmas = filtered
            results.append({"tokens": filtered, "stems": stems, "lemmas": lemmas})
        return results

    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        res = self.process([text])
        return res[0]["lemmas"] if res else []

