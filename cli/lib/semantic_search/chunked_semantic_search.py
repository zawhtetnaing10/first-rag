import json
import os
import re

import numpy as np

from lib.semantic_search.semantic_search import SemanticSearch
from lib.semantic_search.semantic_search import cosine_similarity


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self) -> None:
        super().__init__()
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents):
        """
        Build chunk embeddings from documents
        """

        # Populate documents and document_map
        self.documents = documents
        for document in documents:
            self.document_map[document["id"]] = document

        # Chunks
        all_chunks = []

        # Meta
        chunk_metadata = []

        for movie_idx, document in enumerate(documents):
            description = document["description"]
            # If description is empty skip
            if len(description) == 0:
                continue

            # Semantic chunk the description
            chunks = semantic_chunk(description, 4, 1)

            # Add to all_chunks and metadata
            for chunk_idx, chunk in enumerate(chunks):
                # Add to all chunks
                all_chunks.append(chunk)

                # Create the dictionary for metadata and add to chunk_metadata
                metadata = {
                    "movie_idx": movie_idx,
                    "chunk_idx": chunk_idx,
                    "total_chunks": len(chunks)
                }
                chunk_metadata.append(metadata)

        # Create embeddings using the model and save.
        self.chunk_embeddings = self.model.encode(all_chunks)
        # Save the metadata
        self.chunk_metadata = chunk_metadata

        # Save to disk
        # Chunk embeddings
        chunk_embeddings_path = "cache/chunk_embeddings.npy"
        with open(chunk_embeddings_path, 'wb') as ce:
            np.save(ce, self.chunk_embeddings)

        if os.path.exists(chunk_embeddings_path):
            print("Chunk embeddings successfully saved")

        # Chunk Meta data
        chunk_metadata_path = "cache/chunk_metadata.json"
        with open(chunk_metadata_path, 'w', encoding='utf-8') as f:
            json.dump({"chunks": chunk_metadata,
                      "total_chunks": len(all_chunks)}, f, indent=2)

        if os.path.exists(chunk_metadata_path):
            print("Chunk metadata saved successfully.")

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        """
            If chunk embeddings are already stored on disk, load them into memory.
            If not build them again.
        """

        # Populate documents and document_map
        self.documents = documents
        for document in documents:
            self.document_map[document["id"]] = document

        # If the files exist, open the files
        chunk_embeddings_path = "cache/chunk_embeddings.npy"
        chunk_metadata_path = "cache/chunk_metadata.json"
        if os.path.exists(chunk_embeddings_path) and os.path.exists(chunk_metadata_path):
            with open(chunk_embeddings_path, 'rb') as ce, open(chunk_metadata_path, 'r') as f:
                # Load chunk embeddings
                self.chunk_embeddings = np.load(ce)

                # Load chunk meta data
                root_metadata = json.load(f)
                self.chunk_metadata = root_metadata["chunks"]

                return self.chunk_embeddings
        else:
            return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10):
        """
            Searches the chunks using the query given
        """
        query_embedding = self.generate_embedding(query)

        chunk_scores = []

        for index, chunk_embedding in enumerate(self.chunk_embeddings):
            # Calculate cosine similarity
            score = cosine_similarity(chunk_embedding, query_embedding)

            # Get the metadata
            metadata = self.chunk_metadata[index]

            # Create a dictionary for score
            score_dict = {
                "chunk_idx": metadata["chunk_idx"],
                "movie_idx": metadata["movie_idx"],
                "score": score
            }

            # Append to chunk_scores
            chunk_scores.append(score_dict)

        # Movie idx to score dict
        movie_idx_score_dict = {}

        for chunk_score in chunk_scores:
            movie_idx = chunk_score["movie_idx"]
            score = chunk_score["score"]

            # If movie_idx is already there,Only update the score if current score is higher than previous score.
            # If not, just add the movie_idx together with the score
            if movie_idx in movie_idx_score_dict:
                previous_score = movie_idx_score_dict[movie_idx]
                if score > previous_score:
                    movie_idx_score_dict[movie_idx] = score
            else:
                movie_idx_score_dict[movie_idx] = score

        # Map dict into tuples for sorting
        movie_idx_score_tuples = []
        for key, value in movie_idx_score_dict.items():
            movie_idx_score_tuples.append((key, value))

        # Sort by score
        movie_idx_score_tuples.sort(key=lambda tuple: tuple[1], reverse=True)

        # Get the first limit items
        filtered_tuples = movie_idx_score_tuples[:limit]

        # Map the filtered tuples into the required results.
        result = []
        for tuple in filtered_tuples:
            # This is movie_idx not movie_id. So the doc must be retrieved from self.documents
            movie_doc = self.documents[tuple[0]]
            score = tuple[1]
            result_dict = {
                "id": movie_doc["id"],
                "title": movie_doc["title"],
                # Take the first 100 characters
                "description": movie_doc["description"][:100],
                "score": score
            }
            result.append(result_dict)

        return result


def semantic_chunk(text: str, chunk_size: int, overlap: int):
    """
        Chunk the text semantically
    """

    # Trim the text
    trimmed_text = text.strip()
    if len(trimmed_text) == 0:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", trimmed_text)

    # After splitting the sentences. If there's only one sentence and it doesn't end with punctuation.
    # Treat the whole text as a sentence.
    if len(sentences) == 1:
        first_sentence = sentences[0]
        if not first_sentence.endswith(".") and not first_sentence.endswith("!") and not first_sentence.endswith("?"):
            sentences = [trimmed_text]

    resulting_list = []
    temp_list = []
    for sentence in sentences:
        if len(temp_list) < chunk_size:
            # If still less than chunk size, append the sentence
            trimmed_sentence = sentence.strip()
            if len(trimmed_sentence) > 0:
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
            trimmed_sentence = sentence.strip()
            if len(trimmed_sentence) > 0:
                temp_list.append(sentence)

    # Add the leftover sentences
    if len(temp_list) > 0:
        resulting_list.append(temp_list)

    # Join the inner lists to string
    result = [
        ' '.join(string_list) for string_list in resulting_list
    ]

    return result

# def semantic_chunk(
#     text: str,
#     max_chunk_size: int,
#     overlap: int,
# ) -> list[str]:
#     text = text.strip()

#     if not text:
#         return []

#     sentences = re.split(r"(?<=[.!?])\s+", text)

#     if len(sentences) == 1 and not text.endswith((".", "!", "?")):
#         sentences = [text]

#     chunks = []
#     i = 0
#     n_sentences = len(sentences)

#     while i < n_sentences - overlap:
#         chunk_sentences = sentences[i: i + max_chunk_size]
#         cleaned_sentences = []
#         for chunk_sentence in chunk_sentences:
#             cleaned_sentences.append(chunk_sentence.strip())
#         if not cleaned_sentences:
#             continue
#         chunk = " ".join(cleaned_sentences)
#         chunks.append(chunk)
#         i += max_chunk_size - overlap

#     return chunks
