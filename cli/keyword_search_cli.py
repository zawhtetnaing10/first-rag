#!/usr/bin/env python3

import argparse
import inverted_index

import keyword_search


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands")

    search_parser = subparsers.add_parser(
        "search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser(
        "build", help="Build the inverted index for movies")

    tf_parser = subparsers.add_parser(
        "tf", help="Get term frequency for a term")
    tf_parser.add_argument("document_id", type=str, help="ID of the document")
    tf_parser.add_argument(
        "term", type=str, help="Term used to get the frequency")

    args = parser.parse_args()

    inv_index = inverted_index.InvertedIndex()
    match args.command:
        case "search":
            # Load the inverted index from disk. If there are any errors, just exit
            try:
                inv_index.load()
            except Exception:
                exit

            # print the search query here
            query = args.query
            print(f"Searching for: {query}")

            # do the keyword search
            keyword_search.keyword_search(query, inv_index)

        case "build":
            # Build the index
            inv_index.build()
            # Save to disk
            inv_index.save()
        case "tf":
            # Load the inverted index from disk. If there are any errors, just exit
            try:
                inv_index.load()
            except Exception:
                exit

            # Get the term frequency for the given term.
            try:
                doc_id = args.document_id
                term = args.term

                tf = inv_index.get_tf(doc_id=doc_id, term=term)
                print(f"Term frequency for {term} is: {tf}")
            except KeyError:
                print(0)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
