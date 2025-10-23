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
        "image_path",
        type=str,
        help="Path to the image file.",
    )

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            multimodal_search.verify_image_embedding(args.image_path)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
