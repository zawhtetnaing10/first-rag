import argparse
import json
import re

from lib import semantic_search


def handle_semantic_search(query, limit):
    # Semantic search object
    search_obj = semantic_search.SemanticSearch()
    with open("data/movies.json", 'r') as f:
        # Load the movies
        movies_json = json.load(f)
        documents = movies_json["movies"]

        # Load or create embeddings
        search_obj.load_or_create_embeddings(documents)

        # Run search
        query = query
        limit = limit

        # Run the search and print out the results
        results = search_obj.search(query, limit)
        for index, result in enumerate(results):
            print(
                f"{index+1}. {result["title"]} (score: {result["score"]:.4f})\n{result["description"]}")
            print()


def handle_chunk(text: str, chunk_size: int, overlap: int):
    words = text.split()

    resulting_list = []
    temp_list = []
    for word in words:
        if len(temp_list) < chunk_size:
            # If still less than chunk size, append the word
            temp_list.append(word)
        else:
            # If not append the temp_list
            previous_list = temp_list.copy()
            resulting_list.append(previous_list)

            # Clear the temp list
            temp_list.clear()

            # Add overlapping words. From the previous list
            if overlap > 0:
                temp_list.extend(previous_list[-overlap:])

            # Add the current word.
            temp_list.append(word)

    # Add the leftover words
    if len(temp_list) > 0:
        resulting_list.append(temp_list)

    # Join the inner lists to string
    result = [
        ' '.join(string_list) for string_list in resulting_list
    ]

    char_count = len(text)

    # Print out the result
    print(f"Chunking {char_count} characters")
    for index, chunk in enumerate(result):
        print(f"{index + 1}. {chunk}")


def handle_semantic_chunk(text: str, chunk_size: int, overlap: int):
    result = semantic_chunk(text, chunk_size, overlap)

    # Print out the result
    print(f"Semantically chunking {len(text)} characters")
    for index, chunk in enumerate(result):
        print(f"{index + 1}. {chunk}")


def semantic_chunk(text: str, chunk_size: int, overlap: int):
    sentences = re.split(r"(?<=[.!?])\s+", text)

    resulting_list = []
    temp_list = []
    for sentence in sentences:
        if len(temp_list) < chunk_size:
            # If still less than chunk size, append the sentence
            temp_list.append(sentence)
        else:
            # If not append the temp_list
            previous_list = temp_list.copy()
            resulting_list.append(previous_list)

            # Clear the temp list
            temp_list.clear()

            # Add overlapping sentences. From the previous list
            if overlap > 0:
                temp_list.extend(previous_list[-overlap:])

            # Add the current sentence.
            temp_list.append(sentence)

    # Add the leftover sentences
    if len(temp_list) > 0:
        resulting_list.append(temp_list)

    # Join the inner lists to string
    result = [
        ' '.join(string_list) for string_list in resulting_list
    ]

    return result


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

    # chunk
    chunk_parser = subparsers.add_parser(
        "chunk", help="Chunk the given text")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int,
                              default=200, help="Chunk size")
    chunk_parser.add_argument("--overlap", type=int,
                              default=0, help="Overlap size")

    # semantic_chunk
    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Chunk the given text using Semantic Chunk")
    # semantic_chunk_parser.add_argument(
    #     "--text", required=True, type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument(
        "text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int,
                                       default=4, help="Chunk size")
    # semantic_chunk_parser.add_argument("--overlap", type=int,
    #                                    default=1, help="Overlap size")
    semantic_chunk_parser.add_argument("--overlap", type=int,
                                       default=0, help="Overlap size")

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
            handle_semantic_search(args.query, args.limit)
        case "chunk":
            handle_chunk(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            handle_semantic_chunk(args.text, args.max_chunk_size, args.overlap)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
