import argparse

from bayes import NaiveBayes

parser = argparse.ArgumentParser(description='Political Sentiment in Conjunction with Search')
parser.add_argument('action', help="action to perform")

if __name__ == "__main__":
    bayes = NaiveBayes()
    docs = bayes.load("/home/benlubas/github/Article-Bias-Prediction/data/jsons")
    # print(docs.keys())
    print(list(docs["left"].items())[:100])
