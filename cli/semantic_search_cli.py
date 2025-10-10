import argparse

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
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
