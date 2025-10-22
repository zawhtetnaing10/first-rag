import argparse
import json
import os
import time

from dotenv import load_dotenv
from google import genai

import lib.hybrid_search.hybrid_search as hybrid_search
import lib.utils.prompt_utils as prompt_utils

from sentence_transformers import CrossEncoder


def handle_normalize(scores):
    """
        Normalize the scores into 0 and 1 range
    """

    # Call normalize scores utility function.
    result = hybrid_search.normalize_scores(scores)

    # Print
    for result in result:
        print(f"* {result:.4f}")


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
            print(f"\t Hybrid Score: {result["hybrid_score"]:.4f}")
            print(
                f"\t BM25: {result["normalized_bm25"]:.4f}, Semantic: {result["normalized_semantic"]:.4f}")
            print(f"\t {result["description"]}")


def handle_rrf_search(query, k, limit, enhance, rerank):
    """
        Do a rrf search using hybrid_search
    """

    # Load gemini LLM Client

    # Do a rrf search using HybridSearch
    with open("data/movies.json", 'r') as f:
        # Load the movies
        movies_json = json.load(f)
        documents = movies_json["movies"]

        # Initialize the LLM
        # Load api key from env
        load_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY")
        print(f"Using key {api_key[:6]}...")

        # Prepare genai client.
        client = genai.Client(api_key=api_key)

        # Enhance the query if needed
        enhanced_query = enhance_query(query, enhance, client)

        search_obj = hybrid_search.HybridSearch(documents)

        # Prepare the result
        results = []
        if rerank is not None and len(rerank) > 0:
            # If there's a rerank method, get 500 times the result of the limit.
            results = search_obj.rrf_search(enhanced_query, k, limit * 5)

            # Actually rerank here
            results = rerank_results(
                query, results.copy(), rerank, client, limit)
        else:
            results = search_obj.rrf_search(enhanced_query, k, limit)

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

            print(f"\t RRF Score: {result["rrf_score"]:.4f}")
            print(
                f"\t BM25 Rank: {result["bm25_rank"]}, Semantic Rank: {result["semantic_rank"]}")
            print(f"\t {result["description"]}")


def enhance_query(query, enhance, client: genai.Client) -> str:
    """
        Enhance query according to enhance parameter.
        enhance -> "spell" - Let LLM fix the spelling
    """
    if enhance == "spell":
        # Prompt to fix query
        prompt = prompt_utils.get_enhance_spell_prompt(query)
        # Let gemini fix the typo
        response = client.models.generate_content(
            model="gemini-2.0-flash-001", contents=prompt)

        # Return the fixed query
        return response.text
    elif enhance == "rewrite":
        # Prompt to fix query
        prompt = prompt_utils.get_enhance_rewrite_prompt(query)
        # Let gemini rewrite the query
        response = client.models.generate_content(
            model="gemini-2.0-flash-001", contents=prompt)

        # Return the fixed query
        return response.text
    elif enhance == "expand":
        # Prompt to fix query
        prompt = prompt_utils.get_enhance_expand_prompt(query)
        # Let gemini expand the query
        response = client.models.generate_content(
            model="gemini-2.0-flash-001", contents=prompt)

        # Return the fixed query
        return response.text
    else:
        # If nothing is provided, just return the original query.
        return query


def rerank_results(query, results, rerank, client: genai.Client, limit):
    """
        Rerank the results based on LLM
    """
    if rerank == "individual":
        # Generate rerank score using LLM for each doc
        for doc in results:
            prompt = prompt_utils.get_individual_rerank_prompt(query, doc)
            response = client.models.generate_content(
                model="gemini-2.0-flash-001", contents=prompt)
            rerank_score = float(response.text)
            doc["rerank_score"] = rerank_score
            # Sleep before each gen ai call.
            time.sleep(5)

        # Sort by rerank score
        results.sort(
            key=lambda result: result["rerank_score"], reverse=True)

        # Return the limited results
        return results[:limit]

    elif rerank == "batch":
        # Do a batch re-ranking
        prompt = prompt_utils.get_batch_rerank_prompt(query, results)

        response = client.models.generate_content(
            model="gemini-2.0-flash-001", contents=prompt)
        try:
            print(f"LLM Batch Rerank response {response.text}")
            # Bootdev run and submits are failing without replacing the strings
            reranked_ids = json.loads(response.text)

            # convert results into a dict
            result_dict = {}
            for result in results:
                result_dict[result["id"]] = result

            # Update the rank each doc
            for index, id in enumerate(reranked_ids):
                result_doc = result_dict[id]
                result_doc["rerank_rank"] = index + 1

            # Get the values Sort with rerank_rank
            ranked_results = list(result_dict.values())
            ranked_results.sort(key=lambda result: result["rerank_rank"])

            # Return the limited results
            return ranked_results[:limit]

        except json.JSONDecodeError as e:
            print("Failed to parse the reranked ids")
            exit(1)
    elif rerank == "cross_encoder":
        # Do a cross encoder reranking

        # Create pairs of query and doc-title + doc-description
        pairs = []
        for doc in results:
            pairs.append(
                [query, f"{doc.get('title', '')} - {doc.get('description', '')}"])

        # Create cross encoder instance
        cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")

        # Cross encoded scores
        scores = cross_encoder.predict(pairs).tolist()

        # Add cross encoder score to each doc
        for index, score in enumerate(scores):
            doc = results[index]
            doc["cross_encoder_score"] = score

        # Sort the docs by cross encoder score
        results.sort(
            key=lambda result: result["cross_encoder_score"], reverse=True)

        return results[:limit]

    else:
        return results


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

    args = parser.parse_args()

    match args.command:
        case "normalize":
            handle_normalize(args.scores)
        case "weighted-search":
            handle_weighted_search(args.query, args.alpha, args.limit)
        case "rrf-search":
            handle_rrf_search(args.query, args.k, args.limit,
                              args.enhance, args.rerank_method)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
