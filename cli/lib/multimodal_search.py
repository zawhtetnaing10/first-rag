import json
from PIL import Image
from sentence_transformers import SentenceTransformer
import lib.semantic_search.semantic_search as semantic_search


class MultimodalSearch():
    def __init__(self, documents, model_name="clip-ViT-B-32"):
        self.documents = documents
        self.texts = [
            f"{doc['title']}: {doc['description']}" for doc in documents
        ]
        self.model: SentenceTransformer = SentenceTransformer(model_name)

        # Generate text embeddings
        self.text_embeddings = self.model.encode(
            self.texts, show_progress_bar=True)

    def embed_image(self, image_path):
        # Open and encode image
        image = Image.open(image_path)
        image_embeddings = self.model.encode([image])

        # Return the embedding
        return image_embeddings[0]

    def search_with_image(self, image_path):
        image_embedding = self.embed_image(image_path)

        # Calculate cosine similarities
        scores = []
        for text_embedding in self.text_embeddings:
            score = semantic_search.cosine_similarity(
                image_embedding, text_embedding)
            scores.append(score)

        # Doc score tuples
        doc_score_tuples = []
        for index, score in enumerate(scores):
            doc_score_tuples.append((self.documents[index], score))

        # Sort the docs with scores
        doc_score_tuples.sort(key=lambda tuple: tuple[1], reverse=True)

        # Limit to 5 items
        doc_score_tuples = doc_score_tuples[:5]

        # Result
        result = [
            {
                "id": tuple[0]["id"],
                "title": tuple[0]["title"],
                "description": tuple[0]["description"],
                "similarity": tuple[1]
            }
            for tuple in doc_score_tuples
        ]

        return result


def image_search_command(image_path):
    with open("data/movies.json", 'r') as f:
        # Load the movies
        movies_json = json.load(f)
        documents = movies_json["movies"]

        search = MultimodalSearch(documents=documents)

        return search.search_with_image(image_path)


def verify_image_embedding(image_path: str):
    search = MultimodalSearch()
    embedding = search.embed_image(image_path)

    print(f"Embedding shape: {embedding.shape[0]} dimensions")
