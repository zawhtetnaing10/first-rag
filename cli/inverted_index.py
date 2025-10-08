from collections import Counter
import json
import os
import pickle
import keyword_search

# Inverted index for movies


class InvertedIndex:
    def __init__(self) -> None:
        self.index = {}
        self.docmap = {}
        self.term_frequencies = {}

    # Save document ids for a given text separated into tokens.
    def __add_document(self, doc_id: int, text: str):
        stop_words_path = "data/stopwords.txt"
        with open(stop_words_path, 'r') as s:

            # Load stop words
            stop_words_content = s.read()
            stop_words = stop_words_content.splitlines()

            # Tokenize the text
            tokens = keyword_search.process_text(text, stop_words)

            # Add each token to the index with a set of document ids
            for token in tokens:
                doc_id_set = self.index.setdefault(token, set())
                doc_id_set.add(doc_id)

            # Add term frequency
            counter_for_token = self.term_frequencies.setdefault(
                doc_id, Counter())
            counter_for_token.update(tokens)

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

        # dump all the data
        with open(index_file_path, 'wb') as i, open(docmap_file_path, 'wb') as d, open(term_frequencies_path, 'wb') as tf:
            pickle.dump(self.index, i)
            pickle.dump(self.docmap, d)
            pickle.dump(self.term_frequencies, tf)

        # This is for debugging. Delete after thorough testing
        if os.path.exists(index_file_path):
            print("Index successfully saved to disk")

        if os.path.exists(docmap_file_path):
            print("Docmap successfully saved")

        if os.path.exists(term_frequencies_path):
            print("Term frequencies successfully saved")

    # Load the indices
    def load(self):
        # File paths for index and docmaps
        index_file_path = "cache/index.pkl"
        docmap_file_path = "cache/docmap.pkl"
        term_frequencies_path = "cache/term_frequencies.pkl"

        try:
            # open the files and load the data to memory.
            with open(index_file_path, 'rb') as i, open(docmap_file_path, 'rb') as d, open(term_frequencies_path, 'rb') as tf:
                self.index = pickle.load(i)
                self.docmap = pickle.load(d)
                self.term_frequencies = pickle.load(tf)

        except FileNotFoundError:
            raise Exception(
                "The index files not found. Please use build command to build the index.")
