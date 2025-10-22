import os

from lib.keyword_search.inverted_index import InvertedIndex
from lib.semantic_search.chunked_semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents):
        # Build semantic chunk embeddings
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        # Build index for keyword search
        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def bm_25_search(self, query, limit):
        """
            Keyword search
        """
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        """
            Weighted search
        """
        # Keyword search
        bm25_results = self.bm_25_search(query, limit * 500)

        # Semantic search
        semantic_search_results = self.semantic_search.search_chunks(
            query, limit * 500)

        # Normalize keyword search results and add the normalized scores back
        bm25_scores = [
            result["score"] for result in bm25_results
        ]
        # Create the updated bm25 results with normalized scores
        normalized_bm25_scores = normalize_scores(bm25_scores)
        bm25_results_normalized = []
        for index, bm25_result in enumerate(bm25_results):
            normalized_score = normalized_bm25_scores[index]
            bm25_result["normalized_score"] = normalized_score
            bm25_results_normalized.append(bm25_result)

        # Normalize the semantic search results and add their normalized scores back
        semantic_search_scores = [
            result["score"] for result in semantic_search_results
        ]
        # Create the updated semantic_search results with normalized scores
        normalized_semantic_scores = normalize_scores(semantic_search_scores)
        semantic_results_normalized = []
        for index, semantic_result in enumerate(semantic_search_results):
            normalized_score = normalized_semantic_scores[index]
            semantic_result["normalized_score"] = normalized_score
            semantic_results_normalized.append(semantic_result)

        # Create a dictionary
        hybrid_score_dicts = {}
        # Add the keyword search first
        for bm_25 in bm25_results_normalized:
            result_dict = {
                "id": bm_25["id"],
                "title": bm_25["title"],
                "description": bm_25["description"],
                "bm25": bm_25["score"],
                "normalized_bm25": bm_25["normalized_score"],
                "semantic": 0.0,
                "normalized_semantic": 0.0
            }
            hybrid_score_dicts[result_dict["id"]] = result_dict

        # Add the semantic search
        for sem_search in semantic_results_normalized:
            movie_id = sem_search["id"]
            if movie_id in hybrid_score_dicts:
                dict = hybrid_score_dicts[movie_id]
                dict["semantic"] = sem_search["score"]
                dict["normalized_semantic"] = sem_search["normalized_score"]
                dict["description"] = sem_search["description"]
            else:
                dict = {
                    "id": sem_search["id"],
                    "title": sem_search["title"],
                    "description": sem_search["description"],
                    "bm25": 0.0,
                    "normalized_bm25": 0.0,
                    "semantic": sem_search["score"],
                    "normalized_semantic": sem_search["normalized_score"]
                }
                hybrid_score_dicts[dict["id"]] = dict

        # Calculate hybrid score for each dict
        for _, value in hybrid_score_dicts.items():
            bm25_score = value["normalized_bm25"]
            semantic_score = value["normalized_semantic"]

            hybrid = hybrid_score(bm25_score, semantic_score, alpha)

            value["hybrid_score"] = hybrid

        # Create the list of results from the dict
        result = list(hybrid_score_dicts.values())

        # Sort by hybrid score
        result.sort(key=lambda item: item["hybrid_score"], reverse=True)

        # Return the limited results
        return result[:limit]

    def rrf_search(self, query, k, limit=10):
        """
            RRf search
        """
        # Keyword search
        bm25_results = self.bm_25_search(query, limit * 500)
        # Sort them using bm_25 score. The top one will have the highest rank and score.
        bm25_results.sort(key=lambda result: result["score"], reverse=True)

        # Semantic search
        semantic_search_results = self.semantic_search.search_chunks(
            query, limit * 500)
        # Sort them using semantic_search score. The top one will have the highest rank and score.
        semantic_search_results.sort(
            key=lambda result: result["score"], reverse=True)

        # RRF scores dict
        rrf_scores_dict = {}

        # Add keyword search first
        for index, bm25 in enumerate(bm25_results):
            dict = {
                "id": bm25["id"],
                "title": bm25["title"],
                "description": bm25["description"],
                "bm25_rank": index + 1,
                "semantic_rank": 0
            }
            rrf_scores_dict[dict["id"]] = dict

        # Add semantic search results
        for sem_idx, sem_search in enumerate(semantic_search_results):
            if sem_search["id"] in rrf_scores_dict:
                sem_dict = rrf_scores_dict[sem_search["id"]]
                sem_dict["semantic_rank"] = sem_idx + 1
                sem_dict["description"] = sem_search["description"]
            else:
                sem_dict = {
                    "id": sem_search["id"],
                    "title": sem_search["title"],
                    "description": sem_search["description"],
                    "bm25_rank": 0,
                    "semantic_rank": sem_idx + 1
                }
                rrf_scores_dict[sem_dict["id"]] = sem_dict

        # Get the values from dict
        results = list(rrf_scores_dict.values())

        # Calculate and update the total rrf score
        for result in results:
            bm25_rank = result["bm25_rank"]
            semantic_rank = result["semantic_rank"]

            rrf_bm25 = 0.0
            if bm25_rank != 0:
                rrf_bm25 = rrf_score(bm25_rank, k)

            rrf_semantic = 0.0
            if semantic_rank != 0:
                rrf_semantic = rrf_score(semantic_rank, k)

            result["rrf_score"] = rrf_bm25 + rrf_semantic

        # Sort by rrf score
        results.sort(key=lambda result: result["rrf_score"], reverse=True)

        return results[:limit]


def rrf_score(rank, k=60.0):
    """
        Calculate the reciprocral rank fusion score based on rank and k
    """
    return 1.0 / (k + rank)


def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    """
        Calculate hybrid score from keyword search and semantic_search
    """
    return alpha * bm25_score + (1 - alpha) * semantic_score


def normalize_scores(scores):
    """
        Normalize the scores into 0 to 1 range
    """

    # Result
    result = []

    # No scores are given.
    if len(scores) == 0:
        return

    min_score = min(scores)
    max_score = max(scores)

    # If min == max, just print 1.0s
    if min_score == max_score:
        result = [
            1.0 for score in scores
        ]
        return result

    # Result
    result = [
        (score - min_score) / (max_score - min_score) for score in scores
    ]
    return result
