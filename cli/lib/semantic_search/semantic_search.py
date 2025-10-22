import json
import os
from sentence_transformers import SentenceTransformer
import numpy as np


MOVIE_EMBEDDINGS_PATH = "cache/movie_embeddings.npy"


class SemanticSearch:
    def __init__(self) -> None:
        # Load the model (downloads automatically the first time)
        self.model: SentenceTransformer = SentenceTransformer(
            'all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text: str):
        if len(text.strip()) == 0:
            raise ValueError("The text must not be empty.")

        # We'll only embed the first input for now.
        embeddings = self.model.encode([text])
        result = embeddings[0]

        return result

    def build_embeddings(self, documents):
        # Set documents
        self.documents = documents

        # Set document map and save string representations
        string_reps = []
        for doc in documents:
            self.document_map[doc["id"]] = doc

            # Save string representations
            string_rep = f"{doc['title']}: {doc['description']}"
            string_reps.append(string_rep)

        # Encode the string representations
        self.embeddings = self.model.encode(
            string_reps, show_progress_bar=True)

        # Save the embeddings
        try:
            with open(MOVIE_EMBEDDINGS_PATH, 'wb') as f:
                np.save(f, self.embeddings)
        except FileNotFoundError:
            print(f"File not found for {MOVIE_EMBEDDINGS_PATH}")

        # Return the embeddings
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        # Populate documents and documents map
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc

        # Check if file exists
        if os.path.exists(MOVIE_EMBEDDINGS_PATH):
            # If it is, load the file and save to embeddings
            with open(MOVIE_EMBEDDINGS_PATH, 'rb') as f:
                self.embeddings = np.load(f)

                # TODO: - Do something else if the lengths are not equal
                if len(self.embeddings) == len(documents):
                    return self.embeddings
        else:
            # If it isn't, rebuild the embeddings and return the result
            return self.build_embeddings(documents)

    # Semantic search
    def search(self, query, limit):
        if len(self.embeddings) == 0:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first.")

        # Embed the query
        q_embedding = self.generate_embedding(query)

        # Similarity tuples. (similarity_score, document)
        similarity_tuples = []
        for index, doc_embedding in enumerate(self.embeddings):
            similarity_score = cosine_similarity(q_embedding, doc_embedding)
            document = self.documents[index]

            similarity_tuples.append((similarity_score, document))

        # Sort by similarity score descending.
        similarity_tuples.sort(key=lambda tuple: tuple[0], reverse=True)

        # Return top results up to limit. The results are converted to a dictionary.
        return [{
            "score": similarity_tuple[0],
            "title": similarity_tuple[1]["title"],
            "description": similarity_tuple[1]["description"]
        } for similarity_tuple in similarity_tuples[:limit]]


def verify_embeddings():
    semantic_search = SemanticSearch()
    with open("data/movies.json", 'r') as f:
        # Load the movies
        movies_json = json.load(f)
        documents = movies_json["movies"]

        # Load or create embeddings
        embeddings = semantic_search.load_or_create_embeddings(documents)

        # Print the docs and embeddings
        print(f"Number of docs:   {len(documents)}")
        print(
            f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def embed_query_text(query):
    semantic_search = SemanticSearch()

    embedding = semantic_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def embed_text(text):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_model():
    semantic_search = SemanticSearch()
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")
