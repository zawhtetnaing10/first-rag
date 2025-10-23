import argparse
import json
import mimetypes
from lib.hybrid_search.hybrid_search import HybridSearch
from google import genai
import lib.utils.genai_utils as genai_utils


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--image",
        type=str,
        help="Path to the image file.",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="A text query to rewrite.",
    )

    args = parser.parse_args()
    image_path = args.image
    query = args.query

    # Get mime type
    mime, _ = mimetypes.guess_type(image_path)
    mime = mime or "image/jpeg"

    with open(image_path, 'rb') as f:
        image_bytes = f.read()

        client = genai_utils.get_gemini_client()
        system_prompt = f"""
    Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary
"""
        parts = [
            system_prompt,
            genai.types.Part.from_bytes(data=image_bytes, mime_type=mime),
            query.strip()
        ]
        response = genai_utils.generate_contents_with_parts(parts, client)

        # Print the rewritten prompt
        print(f"Rewritten query: {response.text.strip()}")
        if response.usage_metadata is not None:
            print(
                f"Total tokens:    {response.usage_metadata.total_token_count}")


if __name__ == "__main__":
    main()
