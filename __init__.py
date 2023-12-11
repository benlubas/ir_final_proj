import argparse

from bayes import NaiveBayes
from document_parser import DocumentParser

parser = argparse.ArgumentParser(description='Political Sentiment in Conjunction with Search')
parser.add_argument('action', help="action to perform")

if __name__ == "__main__":
    doc_parser = DocumentParser("../Article-Bias-Prediction/data/")
    bayes = NaiveBayes(doc_parser)
    train = doc_parser.read_split("train")
    test = doc_parser.read_split("test")
    bayes.create_or_load_sentiment_stats(train)
    bayes.train()
    predictions = bayes.predict(test)

    correct = 0
    total = len(test)
    for id, p in predictions.items():
        prediction = sorted(p.items(), key=lambda x: x[1])[-1][0]
        if prediction == test[id].bias_text:
            correct += 1
        # print(f"preditction: {prediction}")
        # print(f"answer: {test[id].bias_text}")
        # break
    print(f"accuracy: {correct / total}") # ~ 49%
    # print(list(docs["left"].items())[:100])
