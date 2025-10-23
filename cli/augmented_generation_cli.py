import argparse
import json

from lib.hybrid_search.hybrid_search import HybridSearch
from lib.keyword_search.inverted_index import InvertedIndex
import lib.utils.genai_utils as genai_utils
import lib.utils.prompt_utils as prompt_utils


def print_results(search_result_titles, llm_response, final_label):
    """
        Print out the results and LLM Response
    """
    # Print the LLM's results
    print(f"Search Results:")
    for title in search_result_titles:
        print(f"  - {title}")
        print(f"\n")

    print(final_label)
    print(llm_response)


def load_movies_and_do_rrf_search(query, limit, full_description=False):
    """
        Load movies and do a rrf search
    """
    with open('data/movies.json', 'r') as mf:
        # Load actual movies
        movies_content = json.load(mf)
        movies = movies_content["movies"]

        # Run the search for each query and get the results
        search_obj = HybridSearch(movies)
        search_results, _ = search_obj.rrf_search_with_enhance_and_query(
            query=query, limit=limit, k=60.0, enhance=None, rerank=None)
        search_result_titles = [
            result["title"] for result in search_results
        ]

        # If full description is needed, it has to be fetched from index.docmap and updated.
        if full_description:
            for result in search_results:
                full_doc = search_obj.idx.docmap[result["id"]]
                result["description"] = full_doc["description"]

        return search_results, search_result_titles


def load_genai_and_generate(prompt):
    """
        Load LLM client and generate contents according to prompt.
    """
    client = genai_utils.get_gemini_client()
    return genai_utils.generate_contents(prompt, client)


def handle_rag(query):
    """
        Do an augmented generation
    """
    search_results, search_result_titles = load_movies_and_do_rrf_search(
        query)

    # Get LLM
    prompt = prompt_utils.get_rag_prompt(query, search_results)
    response_text = load_genai_and_generate(prompt)

    # Print the LLM's results
    print_results(search_result_titles, response_text, "RAG Response:")


def handle_summarize(query, limit):
    """
        Do an augmented generation + summarization
    """
    search_results, search_result_titles = load_movies_and_do_rrf_search(
        query, limit)

    # Get LLM
    prompt = prompt_utils.get_summarize_prompt(query, search_results)
    response_text = load_genai_and_generate(prompt)

    # Print the LLM's results
    print_results(search_result_titles, response_text, "LLM Summary:")


def handle_citations(query, limit):
    """
        Do an augmented generation + summarization
    """
    search_results, search_result_titles = load_movies_and_do_rrf_search(
        query, limit)

    # Get LLM
    prompt = prompt_utils.get_citations_prompt(query, search_results)
    response_text = load_genai_and_generate(prompt)

    # Print the LLM's results
    print_results(search_result_titles, response_text, "LLM Answer:")


def handle_question(query, limit):
    """
        Do an augmented generation + summarization
    """
    search_results, search_result_titles = load_movies_and_do_rrf_search(
        query, limit, full_description=True)

    # Get LLM
    prompt = prompt_utils.get_question_prompt(query, search_results)
    response_text = load_genai_and_generate(prompt)

    # Print the LLM's results
    print_results(search_result_titles, response_text, "Answer:")


def main():
    parser = argparse.ArgumentParser(
        description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands")

    # Rag
    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    # Summarize
    rag_parser = subparsers.add_parser(
        "summarize", help="Perform Summarization (search + summarize the results)"
    )
    rag_parser.add_argument("query", type=str, help="Search query")
    rag_parser.add_argument("--limit", type=int,
                            default=5, help="Result limit")

    # Citation
    rag_parser = subparsers.add_parser(
        "citations", help="Perform RAG with citations"
    )
    rag_parser.add_argument("query", type=str, help="Search query")
    rag_parser.add_argument("--limit", type=int,
                            default=5, help="Result limit")

    # Question
    rag_parser = subparsers.add_parser(
        "question", help="Perform RAG with question"
    )
    rag_parser.add_argument("query", type=str, help="Search query")
    rag_parser.add_argument("--limit", type=int,
                            default=5, help="Result limit")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            # do RAG stuff here
            handle_rag(query)
        case "summarize":
            handle_summarize(args.query, args.limit)
        case "citations":
            handle_citations(args.query, args.limit)
        case "question":
            handle_question(args.query, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
