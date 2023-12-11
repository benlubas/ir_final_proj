from collections import Counter
import json
import os
from dataclasses import dataclass
from typing import Dict
from nltk import PorterStemmer
from nltk import word_tokenize


@dataclass
class Document:
    """Information for a single document"""

    topic: str
    source: str
    bias_text: str
    url: str
    title: str
    date: str
    authors: str
    content: str
    ID: str


class DocumentParser:
    """Parses documents from json files into Document objects"""

    path: str
    """Path to the /data directory in the Article-Bias-Prediction repo"""
    stem: bool

    def __init__(self, path: str, stem: bool = False):
        self.path = path
        self.stemmer = PorterStemmer()
        self.stem = stem

    def read_all(self) -> Dict[str, Document]:
        """Reads all files in the given directory and returns a dict of Document objects"""
        documents = {}
        all_docs_path = os.path.join(self.path, "jsons/")
        for file in os.listdir(all_docs_path):
            if file.endswith(".json"):
                d = self.read_file(os.path.join(all_docs_path, file))
                documents[d.ID] = d
        return documents

    def stem_doc(self, doc: Document) -> Document:
        """Returns a new doc with stemmed content"""
        content = [self.stemmer.stem(w) for w in word_tokenize(doc.content)]
        return Document(**{**doc.__dict__, "content": " ".join(content)})

    def read_split(self, split: str) -> Dict[str, Document]:
        """Reads all documents that belong to a split and returns a dict of Document objects
        args:
            split: the name of the split to read, one of "train", "test", or "valid"
        """
        documents = {}
        split_path = os.path.join(self.path, "splits/random", f"{split}.tsv")
        print(f"Reading {split} split from {split_path}")
        bias = Counter()
        with open(split_path, "r") as f:
            for line in f.readlines()[1:]:
                id, b = line.strip().split("\t")
                bias[b] += 1
                documents[id] = self.read_file(
                    os.path.join(self.path, "jsons/", f"{id}.json")
                )
        print(f"splits:\nLeft:   {bias['0']}\nCenter: {bias['1']}\nRight:  {bias['2']}")
        return documents

    def read_file(self, file_path: str) -> Document:
        """Reads the file at the given path and returns the contents"""
        with open(file_path, "r") as f:
            data = json.load(f)
        del data["content_original"]
        del data["source_url"]
        del data["bias"]
        return Document(**data) if not self.stem else self.stem_doc(Document(**data))
