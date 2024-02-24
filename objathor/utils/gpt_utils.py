import time
from typing import List, Any, Callable, Sequence

import openai
from tqdm import tqdm

from objathor.utils.queries import Message, ComposedMessage

GPT_TIMEOUT_SECONDS = 10
GPT_SLEEP_AFTER_ISSUE_SECONDS = 5
DEFAULT_MAX_ATTEMPTS = 10

DEFAULT_EMBED = "text-embedding-ada-002"

_OPENAI_CLIENT = None


def client() -> openai.OpenAI:
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


def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    import tiktoken

    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def get_embeddings_from_texts(
    texts: Sequence[str],
    model: str = DEFAULT_EMBED,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    chunk_size: int = 1000,
) -> List[List[float]]:
    embs = []
    for i in tqdm(list(range(0, len(texts), chunk_size))):
        chunk = texts[i : i + chunk_size]

        def embedding_create() -> List[List[float]]:
            return [
                d.embedding
                for d in client()
                .embeddings.create(
                    input=[text.replace("\n", " ") for text in chunk],
                    model=model,
                )
                .data
            ]

        embs.extend(
            access_gpt_with_retries(func=embedding_create, max_attempts=max_attempts)
        )

    return embs


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
    prompt: Sequence[Message],
    dialog: Sequence[Message],
    model: str,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    verbose: bool = True,
    **chat_completion_cfg: Any,
) -> str:
    def message_to_content(msg):
        return msg.gpt() if isinstance(msg, ComposedMessage) else [msg.gpt()]

    messages = [
        dict(role=msg.role, content=message_to_content(msg)) for msg in prompt
    ] + [dict(role=msg.role, content=message_to_content(msg)) for msg in dialog]

    def chat_completion_create() -> str:
        all_kwargs = dict(
            model=model,
            messages=messages,
            max_tokens=2000,
            temperature=0.0,
        )
        all_kwargs.update(chat_completion_cfg)
        completion = client().chat.completions.create(**all_kwargs)
        res = completion.choices[0].message.content

        if verbose:
            pt = completion.usage.prompt_tokens
            ct = completion.usage.completion_tokens
            print(
                f"Prompt tokens: {pt}."
                f" Completion tokens: {ct}."
                f" Approx cost: ${(pt * 0.01 + ct * 0.03)/1000:.2g}."
            )

        return res

    return access_gpt_with_retries(
        func=chat_completion_create, max_attempts=max_attempts
    )
