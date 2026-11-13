"""Small shared wrapper for Google Gemini text generation."""

import os
import time

from google import genai


def generate_text(prompt: str) -> str:
    """Send a prompt to the configured Gemini model and return plain text."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set. Add it to your .env file.")

    client = genai.Client(api_key=api_key)
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
                contents=prompt,
            )
            break
        except genai.errors.APIError as exc:
            status_code = getattr(exc, "code", 0)
            retryable = status_code in {408, 429, 500, 502, 503, 504}
            if not retryable or attempt == 2:
                raise
            time.sleep(2 ** attempt)

    if not response.text:
        raise ValueError("Gemini returned an empty response. Please try again.")
    return response.text.strip()
