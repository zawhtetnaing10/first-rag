import argparse
import json
from lib.hybrid_search.hybrid_search import HybridSearch


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    # run evaluation logic here
    with open('data/golden_dataset.json', 'r') as f, open('data/movies.json', 'r') as mf:
        # Load test data
        golden_data_set_content = json.load(f)
        golden_data_set = golden_data_set_content["test_cases"]

        # Load actual movies
        movies_content = json.load(mf)
        movies = movies_content["movies"]

        # Final result
        eval_results = []

        for test_data in golden_data_set:
            # Get the query
            query = test_data["query"]

            # Run the search for each query and get the results
            search_obj = HybridSearch(movies)
            search_results = search_obj.rrf_search(
                query=query, k=60.0, limit=limit)
            search_result_titles = [
                result["title"] for result in search_results
            ]

            # Get the relevant docs
            relevant_docs = test_data["relevant_docs"]

            # Relevant searches
            relevant_searches = []

            for title in search_result_titles:
                if title in relevant_docs:
                    # If the title is in the relevant docs, add to relevant searches
                    relevant_searches.append(title)

            # Calculate precision
            precision = len(relevant_searches) / len(search_result_titles)

            # Calculate recall
            recall = len(relevant_searches) / len(relevant_docs)

            # Calculate F1
            f1 = 2 * (precision * recall) / (precision + recall)

            # Add the results to the final_result list
            eval_results.append(
                {
                    "query": query,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "retrieved": ",".join(search_result_titles),
                    "relevant": ",".join(relevant_searches)
                }
            )

        # Print out the results
        print(f"k={limit}")
        for eval_result in eval_results:
            print(f"- Query: {eval_result["query"]}")
            print(f"\t - Precision@{limit}: {eval_result["precision"]:.4f}")
            print(f"\t - Recall@{limit}: {eval_result["recall"]:.4f}")
            print(f"\t - F1 Score: {eval_result["f1"]:.4f}")
            print(f"\t - Retrieved: {eval_result["retrieved"]}")
            print(f"\t - Relevant: {eval_result["relevant"]}")


if __name__ == "__main__":
    main()
