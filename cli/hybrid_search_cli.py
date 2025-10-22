import argparse
import json
import os
import time

from dotenv import load_dotenv
from google import genai

import lib.hybrid_search.hybrid_search as hybrid_search
import lib.utils.prompt_utils as prompt_utils
import lib.utils.genai_utils as genai_utils

from sentence_transformers import CrossEncoder


def handle_normalize(scores):
    """
        Normalize the scores into 0 and 1 range
    """

    # Call normalize scores utility function.
    result = hybrid_search.normalize_scores(scores)

    # Print
    for result in result:
        print(f"* {result:.3f}")


def handle_weighted_search(query, alpha, limit):
    """
        Do a weighted search using hybrid_search
    """

    # Do a weighted search using HybridSearch
    with open("data/movies.json", 'r') as f:
        # Load the movies
        movies_json = json.load(f)
        documents = movies_json["movies"]

        search_obj = hybrid_search.HybridSearch(documents)
        results = search_obj.weighted_search(query, alpha, limit)

        for index, result in enumerate(results):
            print(f"{index + 1}. {result["title"]}")
            print(f"\t Hybrid Score: {result["hybrid_score"]:.3f}")
            print(
                f"\t BM25: {result["normalized_bm25"]:.3f}, Semantic: {result["normalized_semantic"]:.3f}")
            print(f"\t {result["description"]}")


def handle_rrf_search(query, k, limit, enhance, rerank, evaluate):
    """
        Do a rrf search using hybrid_search
    """

    # Load gemini LLM Client

    # Do a rrf search using HybridSearch
    with open("data/movies.json", 'r') as f:
        # Load the movies
        movies_json = json.load(f)
        documents = movies_json["movies"]

        search_obj = hybrid_search.HybridSearch(documents)

        results, enhanced_query = search_obj.rrf_search_with_enhance_and_query(
            query=query,
            k=k,
            limit=limit,
            enhance=enhance,
            rerank=rerank
        )

        # If enhance method is not empty. Print it out for the user to see
        if enhance is not None and len(enhance) > 0:
            print(
                f"Enhanced query ({enhance}): '{query}' -> '{enhanced_query}'\n")

        # If there's a rerank, print this out
        if rerank is not None and len(rerank) > 0:
            print(f"Reranking top {limit} results using {rerank} method...")

        # Print out the results
        print(f"Reciprocal Rank Fusion Results for '{query}' (k={k}):")
        for index, result in enumerate(results):
            print(f"{index + 1}. {result["title"]}")

            # If reranking is enabled, print out the reranked score as well
            if rerank is not None and len(rerank) > 0:
                if rerank == "individual":
                    print(f"\t Rerank Score: {result["rerank_score"]}/10")
                elif rerank == "batch":
                    print(f"\t Rerank Rank: {result["rerank_rank"]}")
                else:
                    print(
                        f"\t Cross Encoder Score: {result["cross_encoder_score"]}")

            print(f"\t RRF Score: {result["rrf_score"]:.3f}")
            print(
                f"\t BM25 Rank: {result["bm25_rank"]}, Semantic Rank: {result["semantic_rank"]}")
            print(f"\t {result["description"]}")

        # Evaluation starts here
        if evaluate:
            client = genai_utils.get_gemini_client()

            # Create a prompt and generate the scores
            prompt = prompt_utils.get_evaluate_prompt(query, results)
            response_text = genai_utils.generate_contents(
                prompt=prompt, client=client)

            # Parse the response
            eval_scores = json.loads(
                genai_utils.remove_hardcoded_json_symbols(response_text))

            # Print out the scores
            for i in range(len(eval_scores)):
                score = eval_scores[i]
                doc = results[i]

                print(f"{i + 1}. {doc["title"]}: {score}/3")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands")

    # Normalize
    normalize_parser = subparsers.add_parser(
        "normalize", help="Normalize a list of scores into 0 and 1 range"
    )
    normalize_parser.add_argument(
        "scores", type=float, nargs='+', help="scores list")

    # Weighted Search
    weighted_search_parser = subparsers.add_parser(
        "weighted-search", help="Use a hybrid weighted search for a query."
    )
    weighted_search_parser.add_argument(
        "query", type=str, help="Query to search")
    weighted_search_parser.add_argument(
        "--alpha", type=float, default=0.5, help="Alpha or weight between keyword search and semantic search")
    weighted_search_parser.add_argument(
        "--limit", type=int, default=5, help="The limit of the result")

    # RRF Search
    rrf_search_parser = subparsers.add_parser(
        "rrf-search", help="Use a hybrid rrf search for a query."
    )
    rrf_search_parser.add_argument(
        "query", type=str, help="Query to search")
    rrf_search_parser.add_argument(
        "--k", type=int, default=60, help="Value of k to calculate rrf score")
    rrf_search_parser.add_argument(
        "--limit", type=int, default=5, help="The limit of the result")
    rrf_search_parser.add_argument(
        "--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Query enhancement method.")
    rrf_search_parser.add_argument(
        "--rerank-method", type=str, choices=["individual", "batch", "cross_encoder"], help="Rerank method."
    )
    rrf_search_parser.add_argument(
        "--evaluate", action="store_true", default=False, help="Let LLM evaluate the result."
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            handle_normalize(args.scores)
        case "weighted-search":
            handle_weighted_search(args.query, args.alpha, args.limit)
        case "rrf-search":
            handle_rrf_search(args.query, args.k, args.limit,
                              args.enhance, args.rerank_method, args.evaluate)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
