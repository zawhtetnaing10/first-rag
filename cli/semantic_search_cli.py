import argparse
import json

from lib import semantic_search


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands")

    # Verify
    subparsers.add_parser(
        "verify", help="Verify the model")

    # Embed text
    embed_parser = subparsers.add_parser(
        "embed_text", help="Embed a single input.")
    embed_parser.add_argument(
        "text", type=str, help="Text to embed.")

    # Verify embeddings
    subparsers.add_parser(
        "verify_embeddings", help="Embed all movie data and print out the embeddings.")

    # Embed query
    embed_query_parser = subparsers.add_parser(
        "embedquery", help="Embed the given query.")
    embed_query_parser.add_argument(
        "query", type=str, help="Query to embed.")

    # Search
    search_parser = subparsers.add_parser(
        "search", help="Semantic search for a given query.")
    search_parser.add_argument(
        "query", type=str, help="Query to search.")
    search_parser.add_argument(
        "--limit", type=int, default=5, help="Limits number of elements returned.")

    args = parser.parse_args()

    match args.command:
        case "verify":
            semantic_search.verify_model()
        case "embed_text":
            semantic_search.embed_text(args.text)
        case "verify_embeddings":
            semantic_search.verify_embeddings()
        case "embedquery":
            semantic_search.embed_query_text(args.query)
        case "search":
            # Semantic search object
            search_obj = semantic_search.SemanticSearch()
            with open("data/movies.json", 'r') as f:
                # Load the movies
                movies_json = json.load(f)
                documents = movies_json["movies"]

                # Load or create embeddings
                search_obj.load_or_create_embeddings(documents)

                # Run search
                query = args.query
                limit = args.limit

                # Run the search and print out the results
                results = search_obj.search(query, limit)
                for index, result in enumerate(results):
                    print(
                        f"{index+1}. {result["title"]} (score: {result["score"]:.4f})\n{result["description"]}")
                    print()

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
