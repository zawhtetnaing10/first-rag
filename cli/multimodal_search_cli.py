import argparse
import json
import lib.multimodal_search as multimodal_search


def main():
    parser = argparse.ArgumentParser(description="Multi Modal Search")
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands")

    # Verify Image Embeddings
    verify_image_embedding_parser = subparsers.add_parser(
        "verify_image_embedding", help="Verify the image embeddings."
    )
    verify_image_embedding_parser.add_argument(
        "image",
        type=str,
        help="Path to the image file.",
    )

    # Image Search
    image_search_parser = subparsers.add_parser(
        "image_search", help="Search using an image."
    )
    image_search_parser.add_argument(
        "image",
        type=str,
        help="Path to the image file.",
    )

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            multimodal_search.verify_image_embedding(args.image)
        case "image_search":
            results = multimodal_search.image_search_command(args.image)

            # Print out the results
            for index, result in enumerate(results):
                print(
                    f"{index + 1}. {result["title"]} (similarity: {result["similarity"]:.3f})")
                print(f"   {result["description"][:100]}...")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
