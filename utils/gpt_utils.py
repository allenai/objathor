import time
from typing import List, Dict, Any, Union, Callable

import openai


GPT_TIMEOUT_SECONDS = 10
GPT_SLEEP_AFTER_ISSUE_SECONDS = 5
DEFAULT_MAX_ATTEMPTS = 10

DEFAULT_CHAT = "gpt-4"
DEFAULT_EMBED = "text-embedding-ada-002"

_OPENAI_CLIENT = None


def client():
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        _OPENAI_CLIENT = openai.OpenAI(timeout=GPT_TIMEOUT_SECONDS)
    return _OPENAI_CLIENT


def access_gpt_with_retries(
    func: Callable[[], Any], max_attempts: int = DEFAULT_MAX_ATTEMPTS
):
    for _ in range(max_attempts):
        try:
            return func()
        except openai.APITimeoutError:
            print(
                f"OpenAI timeout error, sleeping for {GPT_SLEEP_AFTER_ISSUE_SECONDS} seconds and retrying.",
                flush=True,
            )
        except openai.RateLimitError:
            print(
                f"Rate limited, sleeping for {GPT_SLEEP_AFTER_ISSUE_SECONDS} seconds  and retrying.",
                flush=True,
            )

        time.sleep(GPT_SLEEP_AFTER_ISSUE_SECONDS)

    raise RuntimeError(f"Failed to get answer after max_attempts ({max_attempts})")


def get_embedding(
    text: str, model: str = DEFAULT_EMBED, max_attempts: int = DEFAULT_MAX_ATTEMPTS
) -> List[float]:
    def embedding_create() -> List[float]:
        return (
            client()
            .embeddings.create(
                input=[text.replace("\n", " ")],
                model=model,
            )
            .data[0]
            .embedding
        )

    return access_gpt_with_retries(func=embedding_create, max_attempts=max_attempts)


def get_answer(
    prompt: List[Dict[str, Any]],
    query: Union[str, List[Dict[str, str]]],
    model: str = DEFAULT_CHAT,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    **chat_completion_cfg: Any,
) -> str:
    messages = prompt + ([{"role": "user", "content": query}] if query != "" else [])

    def chat_completion_create() -> str:
        res = (
            client()
            .chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=2000,
                temperature=0.0,
                **chat_completion_cfg,
            )
            .choices[0]
            .message.content
        )
        return res

    return access_gpt_with_retries(
        func=chat_completion_create, max_attempts=max_attempts
    )
