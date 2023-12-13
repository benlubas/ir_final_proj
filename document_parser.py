from collections import Counter
import json
import os
from dataclasses import dataclass
from typing import Dict
from nltk import PorterStemmer
from nltk import word_tokenize
from nltk.corpus import stopwords
import tantivy


@dataclass
class Document:
    """Information for a single document"""

    content: str
    topic: str = ""
    source: str = ""
    bias_text: str = ""
    url: str = ""
    title: str = ""
    date: str = ""
    authors: str = ""
    ID: str = ""


class DocumentParser:
    """Parses documents from json files into Document objects"""

    path: str
    """Path to the /data directory in the Article-Bias-Prediction repo"""
    stem: bool
    stop_remove: bool
    documents: Dict[str, Document] | None

    def __init__(self, path: str, stem: bool = False, stop_remove: bool = False):
        self.path = path
        self.stemmer = PorterStemmer()
        self.stem = stem
        self.stop_remove = stop_remove
        self.stats = {}
        self.documents = None

    def read_all(self) -> Dict[str, Document]:
        """Reads all files in the given directory and returns a dict of Document objects"""
        if self.documents:
            return self.documents
        documents = {}
        all_docs_path = os.path.join(self.path, "jsons/")
        for file in os.listdir(all_docs_path):
            if file.endswith(".json"):
                d = self.read_file(os.path.join(all_docs_path, file))
                documents[d.ID] = d
        self.documents = documents
        return documents

    def stem_doc(self, doc: Document) -> Document:
        """Returns a new doc with stemmed content"""
        content = [self.stemmer.stem(w) for w in word_tokenize(doc.content)]
        return Document(**{**doc.__dict__, "content": " ".join(content)})

    def stop_remove_doc(self, doc: Document) -> Document:
        """ returns a new doc with stop words removed """
        stop_words = set(stopwords.words('english'))
        content = " ".join([w for w in word_tokenize(doc.content) if not w.lower() in stop_words])
        return Document(**{**doc.__dict__, "content": content})

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
        self.stats["bias_counts"] = {
            "left": bias["0"],
            "center": bias["1"],
            "right": bias["2"],
        }
        return documents

    def read_file(self, file_path: str) -> Document:
        """Reads the file at the given path and returns the contents"""
        with open(file_path, "r") as f:
            data = json.load(f)
        del data["content_original"]
        del data["source_url"]
        del data["bias"]
        doc = Document(**data)
        if self.stop_remove:
            doc = self.stop_remove_doc(doc)
        if self.stem:
            doc = self.stem_doc(doc)
        return doc

    def add_tanivity_documents(self, index_writer):
        """Adds all documents to the given tantivy index writer"""
        documents = self.read_all()
        for doc in documents.values():
            index_writer.add_document(tantivy.Document(
                ID=doc.ID,
                title=doc.title,
                content=doc.content,
                bias_text=doc.bias_text,
                authors=doc.authors,
                date=doc.date,
                source=doc.source,
                topic=doc.topic,
                url=doc.url,
            ))
        index_writer.commit()
