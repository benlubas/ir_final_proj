""" Stem all the documents and save them, this takes like 10 minutes to run. Make sure to copy the
contents of Article-Bias-Prediction/data/jsons into ../data/stemmed/jsons """


from nltk import PorterStemmer, word_tokenize
import os

from nltk.jsontags import json

stemmer = PorterStemmer()

for file in os.listdir("data/stemmed/jsons/"):
    with open(os.path.join("data/stemmed/jsons/", file), "r") as f:
        doc = json.load(f)
        doc["content"] = " ".join(
            [stemmer.stem(w) for w in word_tokenize(doc["content"])]
        )
    with open(os.path.join("data/stemmed/jsons/", file), "w") as f:
        json.dump(doc, f)
