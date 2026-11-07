"""Wrapper around the local Ollama runtime. TUT-56, TUT-57.

One base model serves both agents; what differs is the prompt profile, which is
passed in on every call. See docs/architecure.md section 5.1.

The host is pinned to loopback on purpose. The ollama client otherwise honours
the OLLAMA_HOST environment variable, so a stray setting could send prompts --
which carry retrieved course text -- to another machine. Pinning it here keeps
the "no outbound access" property of the model structural.
"""

from dataclasses import dataclass

import httpx
import ollama

MODEL = "llama3.2:3b"
HOST = "http://127.0.0.1:11434"


@dataclass(frozen=True)
class Profile:
    """A prompt profile: the system prompt and sampling settings for one agent."""

    name: str
    system_prompt: str
    temperature: float = 0.2


class LLMError(Exception):
    """The model could not be reached or returned an error. The message is
    display-safe: it never carries the prompt, which may contain course text."""


_client = ollama.Client(host=HOST)


def chat(profile: Profile, user_message: str, schema: dict | None = None) -> str:
    """Send one user message under a profile and return the reply text.

    When schema is given, Ollama constrains the output to that JSON schema. The
    caller must still validate the result: a small model can produce valid JSON
    whose content is wrong.
    """
    try:
        response = _client.chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": profile.system_prompt},
                {"role": "user", "content": user_message},
            ],
            format=schema,
            options={"temperature": profile.temperature},
        )
    except (ollama.ResponseError, ConnectionError, httpx.HTTPError) as exc:
        # Do not chain the library message into ours: it can echo the request.
        raise LLMError(
            f"The local model is not available. Is Ollama running with {MODEL} pulled?"
        ) from exc
    return response.message.content or ""
