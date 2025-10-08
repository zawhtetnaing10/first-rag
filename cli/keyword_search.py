import json
import string
from nltk.stem import PorterStemmer

import inverted_index


def keyword_search(query, inv_index):
    # Runs from root
    movie_file_path = "data/movies.json"
    stop_words_path = "data/stopwords.txt"

    try:
        with open(movie_file_path, 'r') as f, open(stop_words_path, 'r') as s:
            # Load movies
            data = json.load(f)
            movie_list = data['movies']

            # Load stop words
            stop_words_content = s.read()
            stop_words = stop_words_content.splitlines()

            # Process the query token.
            match_query_tokens = process_text(query, stop_words)

            # Search result
            # result = []

            # Add to result
            # for movie in movie_list:

            #     # Process text for movie title.
            #     match_movie_title_tokens = process_text(
            #         movie["title"], stop_words)

            #     # Check if atleast one of the tokens from the query matches one of the tokens from movie title.
            #     quit = False
            #     for query_token in match_query_tokens:
            #         for match_token in match_movie_title_tokens:
            #             if query_token in match_token:
            #                 result.append(movie)
            #                 # This is to break out of the outer loop.
            #                 quit = True
            #                 break
            #         if quit:
            #             break

            # Add doc ids for each query token.
            doc_ids = set()
            quit = False
            for query_token in match_query_tokens:
                doc_ids_for_token = inv_index.get_documents(query_token)
                for doc_id in doc_ids_for_token:
                    doc_ids.add(doc_id)
                    # If length of doc_ids reach the limit. Break out of the whole loop.
                    if len(doc_ids) >= 5:
                        quit = True
                        break
                if quit:
                    break

            # Once doc_ids have been aquired, map each doc_id to a movie doc
            # Search Result
            result = []
            for doc_id in doc_ids:
                movie_doc = inv_index.docmap[doc_id]
                result.append(movie_doc)

            # Sort the result
            result.sort(key=lambda movie: int(movie["id"]))

            # Print the first five movies -- Old
            # for index, movie in enumerate(result[:5]):
            #     print(f"{index + 1}. {movie["title"]}")

            # Print the movie ids together with movie titles
            for movie in result:
                print(f"{movie["id"]}. {movie["title"]}")

    except FileNotFoundError:
        print(f"File not found {movie_file_path}")
    except json.JSONDecodeError:
        print("Failed to decode json")


# Set up text processing
def process_text(text: str, stopwords):

    # Case sensitivity
    case_insensitive = text.lower()

    # Remove punctuation
    trans_table = str.maketrans("", "", string.punctuation)
    removed_punctuation = case_insensitive.translate(trans_table)

    # Tokenization
    tokens = list(filter(lambda token: len(token) >
                  0, removed_punctuation.split(" ")))

    # Remove stop words
    stop_words_removed = list(
        filter(lambda token: token not in stopwords, tokens))

    # Reduce each token to it's root.
    stemmer = PorterStemmer()
    stemmed_tokens = list(
        map(lambda token: stemmer.stem(token), stop_words_removed))

    return stemmed_tokens
