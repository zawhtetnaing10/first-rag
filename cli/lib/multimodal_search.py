from PIL import Image
from sentence_transformers import SentenceTransformer


class MultimodalSearch():
    def __init__(self, model_name="clip-ViT-B-32"):
        self.model: SentenceTransformer = SentenceTransformer(model_name)

    def embed_image(self, image_path):
        # Open and encode image
        image = Image.open(image_path)
        image_embeddings = self.model.encode([image])

        # Return the embedding
        return image_embeddings


def verify_image_embedding(image_path: str):
    search = MultimodalSearch()
    embedding = search.embed_image(image_path)

    print(f"Embedding shape: {embedding.shape[0]} dimensions")
