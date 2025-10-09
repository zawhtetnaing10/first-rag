#!/usr/bin/env python3

import argparse
import math
import inverted_index
import search_utils
import keyword_search


def handle_search(inv_index, query):
    # Load the inverted index from disk. If there are any errors, just exit
    try:
        inv_index.load()
    except Exception:
        exit

    # print the search query here
    print(f"Searching for: {query}")

    # do the keyword search
    keyword_search.keyword_search(query, inv_index)


def handle_build(inv_index):
    # Build the index
    inv_index.build()
    # Save to disk
    inv_index.save()


def handle_tf(inv_index, document_id, term):
    # Load the inverted index from disk. If there are any errors, just exit
    try:
        inv_index.load()
    except Exception:
        exit

    # Get the term frequency for the given term.
    try:
        tf = inv_index.get_tf(doc_id=document_id, term=term)
        print(f"Term frequency for {term} is: {tf}")
    except KeyError:
        print(0)


def handle_idf(inv_index, term):
    # Load the inverted index from disk. If there are any errors, just exit
    try:
        inv_index.load()
    except Exception:
        exit

    # Calculate idf
    idf = calculate_idf(inv_index, term)

    print(f"Inverse document frequency of '{term}': {idf:.2f}")


def calculate_idf(inv_index, term):
    # Process the term (Tokenize)
    processed_terms = keyword_search.process_text(term)

    # Get index and docmap from inv_index
    index = inv_index.index
    docmap = inv_index.docmap

    # Doc count
    doc_count = len(docmap)

    # Get docs for each processed term and add them to a set
    term_docs = set()
    for processed_term in processed_terms:
        docs = index.get(processed_term)
        term_docs.update(docs)

    # Term doc count
    term_doc_count = 0
    if term_docs is not None:
        term_doc_count = len(term_docs)

    return math.log((doc_count + 1) / (term_doc_count + 1))


def handle_tfidf(inv_index, document_id, term):
    # Load the inverted index from disk. If there are any errors, just exit
    try:
        inv_index.load()
    except Exception:
        exit

     # Get the term frequency for the given term.
    try:
        tf = inv_index.get_tf(document_id, term)
        print(f"Term frequency for {term} is: {tf}")

        idf = calculate_idf(inv_index, term)

        tf_idf = tf * idf

        print(
            f"TF-IDF score of '{term}' in document '{document_id}': {tf_idf:.2f}")

    except KeyError:
        print(0)


def handle_bm25idf(inv_index, term):
    # Load the inverted index from disk. If there are any errors, just exit
    try:
        inv_index.load()
    except Exception:
        exit

    bm25idf = inv_index.get_bm25_idf(term)
    print(f"BM25 IDF score of '{term}': {bm25idf:.2f}")


def handle_bm25tf(inv_index, doc_id, term, k1, b):
    # Load the inverted index from disk. If there are any errors, just exit
    try:
        inv_index.load()
    except Exception:
        exit

    bm25tf = inv_index.get_bm25_tf(doc_id, term, k1, b)
    print(f"BM25 TF score of '{term}' in document '{doc_id}': {bm25tf:.2f}")


def handle_bm25search(inv_index, query):
    # Load the inverted index from disk. If there are any errors, just exit
    try:
        inv_index.load()
    except Exception:
        exit

    # Fetch and print the results together with scores.
    results = inv_index.bm25_search(query)
    for index, result in enumerate(results):
        print(
            f"{index + 1}. ({result["id"]}) {result["title"]} - Score: {result["score"]:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands")

    # Search
    search_parser = subparsers.add_parser(
        "search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser(
        "build", help="Build the inverted index for movies")

    # TF
    tf_parser = subparsers.add_parser(
        "tf", help="Get term frequency for a term")
    tf_parser.add_argument("document_id", type=str, help="ID of the document")
    tf_parser.add_argument(
        "term", type=str, help="Term used to get the frequency")

    # IDF
    idf_parser = subparsers.add_parser(
        "idf", help="Get the inverse document frequency for a term")
    idf_parser.add_argument("term", type=str, help="Term used to get idf")

    tfidf_parser = subparsers.add_parser(
        "tfidf", help="Get the tfidf for a term in a document")
    tfidf_parser.add_argument("document_id", type=str,
                              help="Id of the document to look calculate tfidf")
    tfidf_parser.add_argument("term", type=str, help="Term used to get tfidf")

    # BM25 IDF
    bm25_idf_parser = subparsers.add_parser(
        'bm25idf', help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument(
        "term", type=str, help="Term to get BM25 IDF score for")

    # BM25 TF
    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument(
        "term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs='?', default=search_utils.BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument(
        "b", type=float, nargs='?', default=search_utils.BM25_B, help="Tunable BM25 b parameter")

    # BM25 Search
    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring"
    )
    bm25search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    # Create inverted index
    inv_index = inverted_index.InvertedIndex()

    # Handle Commands
    match args.command:
        case "search":
            handle_search(inv_index, args.query)
        case "build":
            handle_build(inv_index)
        case "tf":
            handle_tf(inv_index, args.document_id, args.term)
        case "idf":
            handle_idf(inv_index, args.term)
        case "tfidf":
            handle_tfidf(inv_index, args.document_id, args.term)
        case "bm25idf":
            handle_bm25idf(inv_index, args.term)
        case "bm25tf":
            handle_bm25tf(inv_index, args.doc_id, args.term, args.k1, args.b)
        case "bm25search":
            handle_bm25search(inv_index, args.query)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
