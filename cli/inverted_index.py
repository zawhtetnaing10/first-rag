from collections import Counter
import json
import math
import os
import pickle
import keyword_search
import search_utils


class InvertedIndex:
    def __init__(self) -> None:
        self.index = {}
        self.docmap = {}
        self.term_frequencies = {}
        self.doc_lengths = {}

    # Save document ids for a given text separated into tokens.
    def __add_document(self, doc_id: int, text: str):
        # Tokenize the text
        tokens = keyword_search.process_text(text)

        # Save doc length
        self.doc_lengths[doc_id] = len(tokens)

        # Add each token to the index with a set of document ids
        for token in tokens:
            doc_id_set = self.index.setdefault(token, set())
            doc_id_set.add(doc_id)

        # Add term frequency
        counter_for_token = self.term_frequencies.setdefault(
            doc_id, Counter())
        counter_for_token.update(tokens)

    # Get the average doc length
    def __get_avg_doc_length(self) -> float:
        # Sum of all doc length
        total_doc_length = sum(self.doc_lengths.values())
        # Num of docs
        number_of_docs = len(self.doc_lengths)

        # Edgecase for no documents
        if number_of_docs == 0:
            return 0.0

        return total_doc_length / number_of_docs

    # Get documents for a given token sorted in ascending order
    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term, set())

        doc_ids_list = list(doc_ids)
        doc_ids_list.sort()

        return doc_ids_list

    # Get term frequencies
    def get_tf(self, doc_id: str, term: str) -> int:
        counter = self.term_frequencies[int(doc_id)]
        return counter[term]

    # Get BM 25 idf
    def get_bm25_idf(self, term: str) -> float:

        # df
        tokens = keyword_search.process_text(term)
        if len(tokens) > 1:
            raise Exception("There must be only one token")
        term_docs = self.index.get(tokens[0])
        term_doc_count = len(term_docs)  # This is df

        # N
        doc_count = len(self.docmap)

        # log((N - df + 0.5) / (df + 0.5) + 1)
        return math.log((doc_count - term_doc_count + 0.5) / (term_doc_count + 0.5) + 1)

    # Get BM 25 tf
    def get_bm25_tf(self, doc_id: int, term: str, k1: float = search_utils.BM25_K1, b: float = search_utils.BM25_B) -> float:
        # Tokenize the term and limit the count to 1
        tokens = keyword_search.process_text(term)
        if len(tokens) > 1:
            raise Exception("There must be only one token")

        # Get Raw tf
        raw_tf = self.get_tf(str(doc_id), tokens[0])

        # Length normalization
        doc_length = self.doc_lengths.get(doc_id)
        if doc_length is None:
            raise Exception("Document not found.")
        length_norm = 1 - b + b * (doc_length / self.__get_avg_doc_length())

        # (tf × (k1 + 1)) / (tf + k1)
        return (raw_tf * (k1 + 1)) / (raw_tf + k1 * length_norm)

    # Get bm25 score
    def bm25(self, doc_id: int, term: str) -> float:
        bm25tf = self.get_bm25_tf(doc_id, term)
        bm25idf = self.get_bm25_idf(term)

        return bm25tf * bm25idf

    # BM 25 search
    def bm25_search(self, query: str, limit: int = 5) -> list[dict]:
        # Initialize score dict
        scores_dict = {}

        # Calculate score and populate scores_dict
        tokens = keyword_search.process_text(query)
        for token in tokens:
            doc_ids = self.get_documents(token)

            for doc_id in doc_ids:
                bm25_score = self.bm25(doc_id, token)

                scores_dict[doc_id] = scores_dict.get(doc_id, 0) + bm25_score

        # Map scores dict to a list of tuples
        doc_score_tuples = []
        for doc_id, score in scores_dict.items():
            doc_score_tuples.append((doc_id, score))

        # Sort the tuple
        # Sort by score since tuple[1] is bm25 score
        doc_score_tuples.sort(key=lambda tuple: tuple[1], reverse=True)

        # Prepare the result. Will be a movie doc but with a score attached to it.
        result = []
        for doc_score_tuple in doc_score_tuples:
            movie = self.docmap[doc_score_tuple[0]]
            score = doc_score_tuple[1]
            # Add the score attribute to each movie.
            movie["score"] = score
            result.append(movie)

        return result[:limit]

    # Build the index. Get all the movies and add them to index and docmap

    def build(self):
        movie_file_path = "data/movies.json"

        try:
            with open(movie_file_path, 'r') as f:
                data = json.load(f)

                movie_list = data["movies"]

                for movie in movie_list:
                    # Add document to index
                    self.__add_document(
                        movie["id"], f"{movie["title"]} {movie["description"]}")
                    # Add document to docmap
                    self.docmap[movie["id"]] = movie

        except FileNotFoundError:
            print(f"File not found. {movie_file_path}")
        except json.JSONDecodeError:
            print("Cannot decode json.")

    # Save index and docmap to disk
    def save(self):
        # Create cache directory if not exists
        os.makedirs('cache/', exist_ok=True)

        # File paths for index and docmaps
        index_file_path = "cache/index.pkl"
        docmap_file_path = "cache/docmap.pkl"
        term_frequencies_path = "cache/term_frequencies.pkl"
        doc_length_path = "cache/doc_lengths.pkl"

        # dump all the data
        with open(index_file_path, 'wb') as i, open(docmap_file_path, 'wb') as d, open(term_frequencies_path, 'wb') as tf, open(doc_length_path, 'wb') as dl:
            pickle.dump(self.index, i)
            pickle.dump(self.docmap, d)
            pickle.dump(self.term_frequencies, tf)
            pickle.dump(self.doc_lengths, dl)

        # This is for debugging. Delete after thorough testing
        if os.path.exists(index_file_path):
            print("Index successfully saved to disk")

        if os.path.exists(docmap_file_path):
            print("Docmap successfully saved")

        if os.path.exists(term_frequencies_path):
            print("Term frequencies successfully saved")

        if os.path.exists(doc_length_path):
            print("Doc lengths successfully saved")

    # Load the indices
    def load(self):
        # File paths for index and docmaps
        index_file_path = "cache/index.pkl"
        docmap_file_path = "cache/docmap.pkl"
        term_frequencies_path = "cache/term_frequencies.pkl"
        doc_length_path = "cache/doc_lengths.pkl"

        try:
            # open the files and load the data to memory.
            with open(index_file_path, 'rb') as i, open(docmap_file_path, 'rb') as d, open(term_frequencies_path, 'rb') as tf, open(doc_length_path, 'rb') as dl:
                self.index = pickle.load(i)
                self.docmap = pickle.load(d)
                self.term_frequencies = pickle.load(tf)
                self.doc_lengths = pickle.load(dl)

        except FileNotFoundError:
            raise Exception(
                "The index files not found. Please use build command to build the index.")
