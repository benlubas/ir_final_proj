from math import log10
import os
from typing import Callable, Dict, Tuple
import json

from document_parser import Document, DocumentParser
from collections import Counter

CLASSES = ["left", "center", "right"]


class NaiveBayes:
    """Add 1 smoothed Naive Bayes classifier"""

    pr: Callable[[str, str], float] | None
    doc_parser: DocumentParser

    def __init__(self, file_path: str, doc_parser: DocumentParser):
        self.pr = None
        self.sentiment_stats = None
        self.doc_parser = doc_parser
        self.file_path = file_path
        os.makedirs(file_path, exist_ok=True)

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
        stats_path = os.path.join(self.file_path, f"sentiment_stats.json")
        data = self._count_sentiment(docs)
        with open(stats_path, "w") as f:
            json.dump(data, f)
        self.sentiment_stats = data
        return data

    def load_or_create_sentiment_stats(
        self, docs: Dict[str, Document]
    ) -> Dict[str, Counter]:
        """Loads the sentiment stats from a file if it exists, otherwise creates it and saves it
        args:
            docs: Dictionary of documents to create the stats from
        returns:
            a dict of sentiment to counters for each word
        """
        stats_path = os.path.join(self.file_path, "sentiment_stats.json")
        if os.path.exists(stats_path):
            print(f"loading stats from {stats_path}")
            with open(stats_path, "r") as f:
                data = json.load(f)
        else:
            data = self.create_sentiment_stats(docs)
        self.sentiment_stats = data
        return data

    def train(self):
        """Trains the classifier"""
        if self.sentiment_stats is None:
            raise Exception("Must create sentiment stats before training")

        print("Training classifier")
        params = {s: {} for s in self.sentiment_stats.keys()}
        for sentiment, counts in self.sentiment_stats.items():
            params[sentiment]["denom"] = sum(counts.values()) + len(counts.values())
            params[sentiment]["counts"] = dict(counts.items())
            for word in counts:
                params[sentiment]["counts"][word] += 1

        return params

    def load_params_or_train(self):
        """Loads the parameters from a file if it exists, otherwise trains the classifier and saves
        the params"""
        if self.sentiment_stats is None:
            raise Exception("Must create/load sentiment stats before training")

        path = os.path.join(self.file_path, "params.json")
        if os.path.exists(path):
            print(f"Loading params from {self.file_path}")
            with open(path, "r") as f:
                params = json.load(f)
        else:
            params = self.train()
            with open(os.path.join(self.file_path, "params.json"), "w") as f:
                json.dump(params, f)

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
            probabilities = {s: log10(1/3) for s in CLASSES}
            for word, count in counts.items():
                # calculate the product of the probability of each word for left, right, and center
                for sentiment in CLASSES:
                    probabilities[sentiment] += (
                        log10(self.pr(word, sentiment)) * count
                    )
            prediction_data[id] = probabilities
        return prediction_data

    def predict_doc(self, doc: Document) -> Dict[str, float]:
        """Predicts the sentiment of a single document"""
        if self.pr is None:
            raise Exception("Must train classifier before predicting")

        probabilities = {s: log10(1/3) for s in CLASSES}
        for word in doc.content.split():
            for sentiment in CLASSES:
                probabilities[sentiment] += log10(self.pr(word, sentiment))
        return probabilities

    def predict_scale_doc(self, doc: Document) -> list:
        """Predicts a sentiment scale for a doc, outputs a value on the scale of -1 <= val <= 1, -1 being left, 0 being center, 1 being right"""

        doc_sentiments = self.predict_doc(doc)
        doc_len = len(doc.content.split(" "))

        sentiment_predicted = max(doc_sentiments.items(), key=lambda x: x[1])

        sentiment_scale_val =  sentiment_predicted[1] / doc_len

        if sentiment_predicted[0] == 'left':
            centeredness = sentiment_scale_val / (doc_sentiments['right'] / doc_len)
            term_weight = 10**sentiment_scale_val
        elif sentiment_predicted[0] == 'right':
            centeredness = sentiment_scale_val / (doc_sentiments['left'] / doc_len)
            term_weight = 10**sentiment_scale_val
        else:
            centeredness = 0
            term_weight = 0
            # centeredness = sentiment_scale_val / ((doc_sentiments.get('left') / doc_len) + (doc_sentiments.get('right') / doc_len))
            # if doc_sentiments.get("left") > doc_sentiments.get('right'):
            #     centeredness *= -1
        return [sentiment_predicted[0], centeredness, term_weight]
