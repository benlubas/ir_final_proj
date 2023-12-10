import os
from typing import Dict
import json

from document_parser import Document, DocumentParser
from collections import Counter

DATA_PATH = "data"

CLASSES = ["left", "center", "right"]

class NaiveBayes:
    """Add 1 smoothed Naive Bayes classifier"""

    def __init__(self):
        pass

    def _doc_counts(self, docs: Dict[str, Document]):
        """Counts the number of occurrences of each word in each doc in the corpus."""
        counts = {}
        for id, doc in docs.items():
            counts[id] = Counter()
            for word in doc.content.split():
                counts[id][word] += 1
        return counts

    def _count_sentiment(self, docs: Dict[str, Document]) -> Dict[str, Counter]:
        """Counts the number of positive, negative, and neutral occurrences of each word in the
        corpus"""
        data = {s: Counter() for s in CLASSES}
        counts = self._doc_counts(docs)

        for id, doc in docs.items():
            doc_counts = counts[id]
            for word, count in doc_counts.items():
                data[doc.bias_text][word] += count

        return data

    def _create_or_load_sentiment_stats(
        self, docs: Dict[str, Document]
    ) -> Dict[str, Counter]:
        """Loads the sentiment stats from a file if it exists, otherwise creates it and saves it"""
        stats_path = os.path.join(DATA_PATH, "sentiment_stats.json")
        if os.path.exists(stats_path):
            with open(stats_path, "r") as f:
                data = json.load(f)
        else:
            data = self._count_sentiment(docs)
            with open(stats_path, "w") as f:
                json.dump(data, f)
        return data

    def load(self, path: str) -> Dict[str, Counter]:
        """Loads the sentiment stats from a file if it exists, otherwise creates it and saves it
        args:
            path: path to a directory containing JSON data
        returns:
            a dict of sentiment to counters for each word
        """
        stats_path = os.path.join(DATA_PATH, "sentiment_stats.json")
        if os.path.exists(stats_path):
            with open(stats_path, "r") as f:
                data = json.load(f)
        else:
            docs = DocumentParser().read_all(path)
            data = self._count_sentiment(docs)
            with open(stats_path, "w") as f:
                json.dump(data, f)
        return data

    # TODO: Test this. Not sure that it works. If it does, how it works.
    def train(self, term_stats):
        params = {s: {} for s in term_stats.keys()}
        for sentiment, counts in term_stats.items():
            params[sentiment]["denom"] = sum(counts.values()) + len(counts.values())
            params[sentiment]["counts"] = dict(counts.items())
            for word in counts:
                params[sentiment]["counts"][word] += 1

        # a function that will return P(word | sentiment) based on 'term_stats'
        def p(word, sentiment):
            if word in params[sentiment]["counts"]:
                return params[sentiment]["counts"][word] / params[sentiment]["denom"]
            return 1 / params[sentiment]["denom"]

        return p

    # TODO: connect this up, it's currently not very usable
    def predict(self, doc_counts, pr):
        CONST = 6e3
        prediction_data = {}
        for id, counts in doc_counts.items():
            p_pos = 1
            p_neg = 1
            for word, count in counts.items():
                # calculate the product of the probability of each word for both positive and negative
                p_pos *= pr(word, "pos") * count * CONST
                p_neg *= pr(word, "neg") * count * CONST

            prediction_data[id] = {
                "pos": p_pos,
                "neg": p_neg,
            }
