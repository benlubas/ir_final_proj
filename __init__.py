import os
from typing import Dict
from argparser import build_parser, handle_docs_command, handle_query_command
from naive_bayes import NaiveBayes
from document_parser import Document, DocumentParser
from tantivy_search import TantivySearch

DATA_PATH = "./data"


def test_accuracy(model: NaiveBayes, test_set: Dict[str, Document]) -> float:
    predictions = model.predict(test)

    correct = 0
    total = len(test_set)
    for id, p in predictions.items():
        prediction = sorted(p.items(), key=lambda x: x[1])[-1][0]
        if prediction == test_set[id].bias_text:
            correct += 1
    return correct / total


def compare_other_bayes(vanila_bayes, vanila_test):
    stem_doc_parser = DocumentParser("./data/stemmed/")
    stem_train = stem_doc_parser.read_split("train")
    stem_test = stem_doc_parser.read_split("test")

    stop_remove_doc_parser = DocumentParser("./data/stop_removed/")
    stop_remove_train = stop_remove_doc_parser.read_split("train")
    stop_remove_test = stop_remove_doc_parser.read_split("test")

    stop_stem_doc_parser = DocumentParser("./data/stop_stem/")
    stop_stem_train = stop_stem_doc_parser.read_split("train")
    stop_stem_test = stop_stem_doc_parser.read_split("test")

    stem_bayes = NaiveBayes(os.path.join(DATA_PATH, "stem_bayes"), stem_doc_parser)
    stem_bayes.load_or_create_sentiment_stats(stem_train)
    stem_bayes.load_params_or_train()

    stop_remove_bayes = NaiveBayes(
        os.path.join(DATA_PATH, "stop_remove_bayes"), stop_remove_doc_parser
    )
    stop_remove_bayes.load_or_create_sentiment_stats(stop_remove_train)
    stop_remove_bayes.load_params_or_train()

    stop_stem_bayes = NaiveBayes(
        os.path.join(DATA_PATH, "stop_stem_bayes"), stop_stem_doc_parser
    )
    stop_stem_bayes.load_or_create_sentiment_stats(stop_stem_train)
    stop_stem_bayes.load_params_or_train()

    print(
        f"Vanila NaiveBayes accuracy: {test_accuracy(vanila_bayes, vanila_test)}"
    )  # ~ 62%
    print(
        f"Stemmed NaiveBayes accuracy: {test_accuracy(stem_bayes, stem_test)}"
    )  # ~ 31% (wow)
    print(
        f"Stopword Removal NaiveBayes accuracy: {test_accuracy(stop_remove_bayes, stop_remove_test)}"
    )  # ~ 44%
    print(
        f"Stopword Removal and Stemming NaiveBayes accuracy: {test_accuracy(stop_remove_bayes, stop_stem_test)}"
    )  # 44% (identical)
    # print(list(docs["left"].items())[:100])


def test_sentiment(vanila_doc_parser, bayes):
    test_doc = vanila_doc_parser.read_file(
        "../Article-Bias-Prediction/data/jsons/0a2hVwQs5IIjm7ur.json"
    )
    sentiment_stats = bayes.predict_scale_doc(test_doc)
    print("Showing bias scale:")
    print("Document: " + test_doc.topic)
    print("Predicted Sentiment: " + sentiment_stats[0])
    print(
        "Sentiment ratio (how much more right or left the document is): "
        + str(sentiment_stats[1])
    )
    print("Political sentiment strenght: " + str(sentiment_stats[2]))


if __name__ == "__main__":
    # NOTE: Only have to do this once
    # import nltk
    # nltk.download('punkt')

    vanila_doc_parser = DocumentParser("../Article-Bias-Prediction/data/")
    train = vanila_doc_parser.read_split("train")
    test = vanila_doc_parser.read_split("test")

    bayes = NaiveBayes(os.path.join(DATA_PATH, "vanila_bayes"), vanila_doc_parser)
    bayes.load_or_create_sentiment_stats(train)
    bayes.load_params_or_train()

    # compare_other_bayes(bayes, test)
    # test_sentiment(vanila_doc_parser, bayes)

    ts = TantivySearch("vanila")
    if not ts.index_exists:
        ts.add_documents(vanila_doc_parser)

    args = build_parser().parse_args()
    print()  # print blank line
    match args.command:
        case "docs":
            handle_docs_command(args, vanila_doc_parser.read_all())
        case "query":
            handle_query_command(args, bayes, ts)
