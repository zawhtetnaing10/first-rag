import os
from dotenv import load_dotenv
from google import genai


def get_gemini_client():
    # Initialize the LLM
    # Load api key from env
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    print(f"Using key {api_key[:6]}...")

    # Prepare genai client.
    client = genai.Client(api_key=api_key)

    return client


def generate_contents(prompt, client: genai.Client) -> str:
    """
        Generate text from LLM using the prompt.
    """
    response = client.models.generate_content(
        model="gemini-2.0-flash-001", contents=prompt)
    return response.text


def generate_contents_with_parts(parts, client: genai.Client) -> genai.types.GenerateContentResponse:
    """
        Generate response from llm using parts.
    """
    response = client.models.generate_content(
        model="gemini-2.0-flash-001", contents=parts)
    return response


def remove_hardcoded_json_symbols(response_text: str) -> str:
    """
        Remove ``` and json from string for json.loads()
    """
    return response_text.replace("```", "").replace("json", "")
