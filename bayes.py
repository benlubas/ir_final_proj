import os
from types import FunctionType
from typing import Callable, Dict
import json

from document_parser import Document, DocumentParser
from collections import Counter

DATA_PATH = "data"

CLASSES = ["left", "center", "right"]

PR_CONST = 6e4

class NaiveBayes:
    """Add 1 smoothed Naive Bayes classifier"""

    pr: Callable[[str, str], float] | None
    doc_parser: DocumentParser

    def __init__(self, doc_parser: DocumentParser):
        self.pr = None
        self.sentiment_stats = None
        self.doc_parser = doc_parser

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

    def create_sentiment_stats(self, docs: Dict[str, Document]) -> Dict[str, Counter]:
        """Create the sentiment stats for the given docs, saves the stats to a file
        args:
            docs: Dictionary of documents to create the stats from
        returns:
            a dict of sentiment to counters for each word
        """
        stats_path = os.path.join(DATA_PATH, "sentiment_stats.json")
        data = self._count_sentiment(docs)
        with open(stats_path, "w") as f:
            json.dump(data, f)
        self.sentiment_stats = data
        return data

    def create_or_load_sentiment_stats(
        self, docs: Dict[str, Document]
    ) -> Dict[str, Counter]:
        """Loads the sentiment stats from a file if it exists, otherwise creates it and saves it
        args:
            docs: Dictionary of documents to create the stats from
        returns:
            a dict of sentiment to counters for each word
        """
        stats_path = os.path.join(DATA_PATH, "sentiment_stats.json")
        if os.path.exists(stats_path):
            with open(stats_path, "r") as f:
                data = json.load(f)
        else:
            data = self.create_sentiment_stats(docs)
        self.sentiment_stats = data
        return data

    def train(self):
        """Trains the classifier on the given sentiment stats"""
        if self.sentiment_stats is None:
            raise Exception("Must create/load sentiment stats before training")

        params = {s: {} for s in self.sentiment_stats.keys()}
        for sentiment, counts in self.sentiment_stats.items():
            params[sentiment]["denom"] = sum(counts.values()) + len(counts.values())
            params[sentiment]["counts"] = dict(counts.items())
            for word in counts:
                params[sentiment]["counts"][word] += 1

        # a function that will return P(word | sentiment) based on 'term_stats'
        def pr(word, sentiment):
            if word in params[sentiment]["counts"]:
                return params[sentiment]["counts"][word] / params[sentiment]["denom"]
            return 1 / params[sentiment]["denom"]

        self.pr = pr

    def predict(self, docs: Dict[str, Document]) -> Dict[str, Dict[str, float]]:
        """Predicts sentiments of documents given a map from doc ID to word counts in the
        document"""
        if self.pr is None:
            raise Exception("Must train classifier before predicting")

        doc_counts = self._doc_counts(docs)
        prediction_data = {}
        for id, counts in doc_counts.items():
            probabilities = {s: 1.0 for s in CLASSES}
            for word, count in counts.items():
                # calculate the product of the probability of each word for both positive and negative
                for sentiment in CLASSES:
                    probabilities[sentiment] *= self.pr(word, sentiment) * count * PR_CONST
            prediction_data[id] = normalize(probabilities)
        return prediction_data

    def predict_doc(self, doc: Document) -> Dict[str, float]:
        """Predicts the sentiment of a single document"""
        if self.pr is None:
            raise Exception("Must train classifier before predicting")

        probabilities = {s: 1.0 for s in CLASSES}
        for word in doc.content.split():
            for sentiment in CLASSES:
                probabilities[sentiment] *= self.pr(word, sentiment) * PR_CONST
        return probabilities

def normalize(d: Dict[str, float]) -> Dict[str, float]:
    """Normalizes the values in a dictionary to sum to 1"""
    total = sum(d.values())
    return {k: v / total for k, v in d.items()}
