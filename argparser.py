"""
A command line application with the following behavior/capabilities:
- `docs stats` - print out stats about the collection, or a specific document
- `docs show` - print out the contents of a specific document
"""
import argparse
import json
from typing import Dict

from document_parser import Document
from naive_bayes import NaiveBayes
from tantivy_search import TantivySearch


# Query output width in terminal columns
WIDTH = 80


def build_parser():
    parser = argparse.ArgumentParser(description="Final project")

    subparsers = parser.add_subparsers(dest="command")

    docs = subparsers.add_parser(
        "docs", help="interact with documents in the collection"
    )
    docs.add_argument("docs_command", choices=["stats", "show"])
    docs.add_argument("--id", help="document id")

    query = subparsers.add_parser(
        "query", help="Query models for either sentiment (bayes) or documents (tantivy)"
    )
    query.add_argument(
        "model",
        help="which model would you like to query",
        choices=["bayes", "tantivy"],
    )
    query.add_argument("-f", "--file", help="file with query text")
    query.add_argument("-q", "--query", help="query as a string")
    query.add_argument(
        "-j",
        "--json",
        help="path to document represented as json (useful for sentiment analysis of a document)",
    )
    query.add_argument("-o", "--only", help="only show results with this bias")
    query.add_argument("-r", "--remove", help="remote results with this bias")
    query.add_argument(
        "-l", "--limit", help="limit the number of results", type=int, default=10
    )
    query.add_argument(
        "-d",
        "--debug",
        help="print extra info about rankings",
        action="store_true",
    )
    query.add_argument(
        "-b",
        "--bias",
        help="prefer which bias",
        type=str,
        choices=["left", "right", "center", "none"],
        default="none",
    )
    return parser


def handle_docs_command(args, docs: Dict[str, Document]):
    match args.docs_command:
        case "stats":
            # TODO: more interesting stats.
            print(f"There are {len(docs)} documents in the collection")
        case "show":
            if not args.id:
                print("Must provide an id with `doc show`")
                return
            if args.id not in docs:
                print(f"Document with id {args.id} not found")
                return
            print(docs[args.id])


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def handle_query_command(args, bayes: NaiveBayes, tantivy: TantivySearch):
    if args.file:
        with open(args.file, "r") as f:
            text = f.read()
    elif args.json:
        with open(args.json, "r") as f:
            text = json.load(f)["content"]
    elif args.query:
        text = args.query
    else:
        text = input()

    line = "=" * (len(text) + 10)
    print(f'{line}\nQuery: "{text}":\n{line}')
    match args.model:
        case "bayes":
            # want to take some text, and predict the bias, either from a file, or from a string
            doc = Document(content=text)
            prediction = bayes.predict_doc(doc)
            # TODO: need to use the sliding scale stuff for this, can no longer normalize it and get
            # an accurate result
            print(prediction)
            # total = sum(prediction.values())
            # for bias, score in sorted(
            #     prediction.items(), key=lambda x: x[1], reverse=True
            # ):
                # print(f"{bias.rjust(8)}: {round(score / total * 100, 2)}%")
        case "tantivy":
            results = tantivy.query(text)
            results = adjust_rankings(results, args.bias, bayes)
            if args.remove:
                results = [res for res in results if res[1] != args.remove]
            if args.only:
                results = [res for res in results if res[1] == args.only]
            for doc, prediction, score in results[: args.limit]:
                title = (
                    f"{doc['title'][0][:WIDTH - 24]}..."
                    if len(doc["title"][0]) > WIDTH - 24
                    else doc["title"][0]
                )
                print(
                    format_result(
                        title,
                        score,
                        doc["ID"][0],
                        doc["content"][0],
                        prediction,
                        args.debug,
                    )
                )


# 10% boost, This gives mostly the requested articles, highly dependent on the query
BOOST_MULTIPLIER = 1.1


def adjust_rankings(results, bias: str, cassifier: NaiveBayes):
    for i in range(len(results)):
        doc, score = results[i]
        classes = cassifier.predict_doc(Document(content=doc["content"][0]))
        prediction = most_likely(classes)
        if prediction == bias:
            results[i] = (doc, prediction, score * BOOST_MULTIPLIER)
        else:
            results[i] = (doc, prediction, score)
    return sorted(results, key=lambda x: x[-1], reverse=True)


def format_result(title, score, id, content, predicted_sentiment, debug):
    colors = {
        "left": bcolors.OKBLUE,
        "right": bcolors.FAIL,  # not throwing shade, repubs are just normally red
        "center": bcolors.HEADER,
        "none": bcolors.OKGREEN,
    }
    title_color = colors[predicted_sentiment]
    heading = f"{title_color}{title}{bcolors.ENDC}"
    heading += f"{bcolors.OKCYAN}{id.rjust(WIDTH - len(title))}{bcolors.ENDC}"
    preview = f"{content[:WIDTH]}\n{content[WIDTH:WIDTH * 2 - 3]}..."
    fmt_score = f"\nScore: {score}"
    return f"{heading}{fmt_score if debug else ''}\n{preview}\n"


def most_likely(d: Dict[str, float]) -> str:
    """Returns the key with the lowest value in the given dictionary (and is thus the most likely
    prediction)"""
    total = sum(d.values())
    return sorted([(k, v / total) for k, v in d.items()], key=lambda x: x[1])[0][0]
