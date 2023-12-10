import json
import os
from dataclasses import dataclass
from typing import Dict

@dataclass
class Document():
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

class DocumentParser():
    """Parses documents from json files into Document objects"""
    def read_all(self, file_path: str) -> Dict[str, Document]:
        """Reads all files in the given directory and returns a dict of Document objects"""
        documents = {}
        for file in os.listdir(file_path):
            if file.endswith(".json"):
                d = self.read_file(os.path.join(file_path, file))
                documents[d.ID] = d
        return documents

    def read_file(self, file_path: str) -> Document:
        """Reads the file at the given path and returns the contents"""
        with open(file_path, "r") as f:
            data = json.load(f)
        del data["content_original"]
        del data["source_url"]
        del data["bias"]
        return Document(**data)
