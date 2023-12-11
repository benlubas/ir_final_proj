from typing import Dict
from naive_bayes import NaiveBayes
from document_parser import Document, DocumentParser

# TODO: arg parser
# parser = argparse.ArgumentParser(description='Political Sentiment in Conjunction with Search')
# parser.add_argument('action', help="action to perform")

def test_accuracy(model: NaiveBayes, test_set: Dict[str, Document]) -> float:
    predictions = model.predict(test)

    correct = 0
    total = len(test_set)
    for id, p in predictions.items():
        prediction = sorted(p.items(), key=lambda x: x[1])[-1][0]
        if prediction == test_set[id].bias_text:
            correct += 1
    return correct / total

if __name__ == "__main__":
    # NOTE: Only have to do this once
    # import nltk
    # nltk.download('punkt')

    vanila_doc_parser = DocumentParser("../Article-Bias-Prediction/data/")
    train = vanila_doc_parser.read_split("train")
    test = vanila_doc_parser.read_split("test")

    stem_doc_parser = DocumentParser("./data/stemmed/")
    stem_train = stem_doc_parser.read_split("train")
    stem_test = stem_doc_parser.read_split("test")

    bayes = NaiveBayes(vanila_doc_parser)
    bayes.create_or_load_sentiment_stats(train, "vanila_sentiment_stats")
    bayes.train()

    stem_bayes = NaiveBayes(stem_doc_parser)
    stem_bayes.create_or_load_sentiment_stats(stem_train, "stem_sentiment_stats")
    stem_bayes.train()

    print(f"Vanila NaiveBayes accuracy: {test_accuracy(bayes, test)}") # ~ 46%
    print(f"Stemmed NaiveBayes accuracy: {test_accuracy(stem_bayes, stem_test)}") # ~ 25% (wow)
    # print(list(docs["left"].items())[:100])
