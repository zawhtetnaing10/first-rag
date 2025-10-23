def get_enhance_spell_prompt(query) -> str:
    """
        Get a prompt for LLM to correct the spelling errors.
    """
    return f"""Fix any spelling errors in this movie search query.

Only correct obvious typos. Don't change correctly spelled words.

Query: "{query}"

If no errors, return the original query.
Corrected:"""


def get_enhance_rewrite_prompt(query) -> str:
    """
        Get a prompt for LLM to rewrite a more concise query.
    """
    return f"""Rewrite this movie search query to be more specific and searchable.

Original: "{query}"

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep it concise (under 10 words)
- It should be a google style search query that's very specific
- Don't use boolean logic

Examples:

- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

Rewritten query:"""


def get_enhance_expand_prompt(query) -> str:
    """
        Get a prompt for LLM to expand the query
    """
    return f"""Expand this movie search query with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
This will be appended to the original query.

Examples:

- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

Query: "{query}"
"""


def get_individual_rerank_prompt(query, doc):
    """
        Get a prompt for LLM to generate a new score for reranking
    """
    return f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("description", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Give me ONLY the number in your response, no other text or explanation.

Score:"""


def get_batch_rerank_prompt(query, docs):
    """
        Get a prompt for LLM to batch rerank the documents by their ids
    """
    return f"""Rank these movies by relevance to the search query.

Query: "{query}"

Movies:
{docs}

Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. Remove any strings like "json" or "```" from your response since, it'll be parsed not displayed. For example:

[75, 12, 34, 2, 1]
"""


def get_evaluate_prompt(query, formatted_results: dict):
    """
        Get the prompt to evaluate the formatted results of 
    """
    return f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{chr(10).join(str(formatted_results))}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers out than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""


def get_rag_prompt(query, results):
    """
        Get a prompt for RAG.
    """
    return f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Documents:
{results}

Provide a comprehensive answer that addresses the query:"""


def get_summarize_prompt(query, results):
    """
        Gets a prompt to summarize the rrf searc hresponse
    """
    return f"""
Provide information useful to this query by synthesizing information from multiple search results in detail.
The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
This should be tailored to Hoopla users. Hoopla is a movie streaming service.
Query: {query}
Search Results:
{results}
Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
"""


def get_citations_prompt(query, results):
    """
        Get a prompt to include citations in the result.
    """
    return f"""Answer the question or provide information based on the provided documents.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

Query: {query}

Documents:
{results}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources using [1], [2], etc. format when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the documents, say "I don't have enough information"
- Be direct and informative

Answer:"""


def get_question_prompt(question, context):
    return f"""Answer the user's question based on the provided movies that are available on Hoopla.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Question: {question}

Documents:
{context}

Instructions:
- Answer questions directly and concisely
- Be casual and conversational
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation

Answer:"""
