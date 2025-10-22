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
