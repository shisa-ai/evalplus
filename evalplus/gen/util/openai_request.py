import time

import openai
from openai.types.chat import ChatCompletion


def make_request(
    client: openai.Client,
    message: str,
    model: str,
    max_tokens: int = 512,
    temperature: float = 1,
    n: int = 1,
    **kwargs
) -> ChatCompletion:
    # EvalPlus historically used a sampling-heavy configuration here.
    # For compatibility with OpenAI-compatible gateways (including
    # Gemini's /v1beta/openai bridge) we stick to the standard
    # `max_tokens` parameter instead of the newer
    # `max_completion_tokens`, which some providers do not yet
    # recognize on the chat.completions endpoint.
    kwargs["top_p"] = 0.95
    kwargs["max_tokens"] = max_tokens
    if model.startswith("o1-"):  # pop top-p and max_completion_tokens
        kwargs.pop("top_p")
        kwargs.pop("max_tokens")
        temperature = 1.0  # o1 models do not support temperature

    return client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": message},
        ],
        temperature=temperature,
        n=n,
        **kwargs
    )


def make_auto_request(*args, **kwargs) -> ChatCompletion:
    ret = None
    while ret is None:
        try:
            ret = make_request(*args, **kwargs)
        except openai.RateLimitError:
            print("Rate limit exceeded. Waiting...")
            time.sleep(5)
        except openai.APIConnectionError:
            print("API connection error. Waiting...")
            time.sleep(5)
        except openai.APIError as e:
            print(e)
        except Exception as e:
            print("Unknown error. Waiting...")
            print(e)
            time.sleep(1)
    return ret
