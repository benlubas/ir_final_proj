""" Stem and stop word remove all the documents and save them, this takes like 10 minutes to run. Make sure to copy the
contents of Article-Bias-Prediction/data/jsons into ../data/stop_stem_jsons """


from nltk import PorterStemmer, word_tokenize
import os
from nltk.corpus import stopwords
from nltk.jsontags import json

stop_words = set(stopwords.words('english'))

stemmer = PorterStemmer()

for file in os.listdir("data/stop_stem_jsons/"):
    with open(os.path.join("data/stop_stem_jsons/", file), "r") as f:
        doc = json.load(f)
        doc["content"] = " ".join([stemmer.stem(w) for w in word_tokenize(doc["content"]) if w.lower() not in stop_words])
    with open(os.path.join("data/stop_stem_jsons/", file), "w") as f:
        json.dump(doc, f)
