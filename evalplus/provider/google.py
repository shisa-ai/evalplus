import json
import os
import time
from pathlib import Path
from traceback import print_exc
from typing import List

from evalplus.provider.base import DecoderBase

# Prefer the newer google-genai client when available so we can use
# ThinkingConfig and consistent safety controls. Fall back to the older
# google.generativeai client otherwise.
try:
    import google.genai as genai_native  # type: ignore[import]
    from google.genai import types as genai_types  # type: ignore[import]

    HAVE_NATIVE_GENAI = True
except Exception:  # pragma: no cover - optional dependency
    HAVE_NATIVE_GENAI = False
    genai_native = None  # type: ignore[assignment]
    genai_types = None  # type: ignore[assignment]

import google.generativeai as genai_legacy  # type: ignore[import]
from google.api_core.exceptions import GoogleAPICallError, ResourceExhausted  # type: ignore[import]


def make_request(
    client: genai_legacy.GenerativeModel,
    messages: List,
    temperature: float,
    n: int,
    max_new_tokens: int = 2048,
) -> genai_legacy.types.GenerateContentResponse:
    messages = [{"role": m["role"], "parts": [m["content"]]} for m in messages]
    response = client.generate_content(
        messages,
        generation_config=genai_legacy.types.GenerationConfig(
            candidate_count=n,
            max_output_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.95,
        ),
        safety_settings=[
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        ],
    )

    # Optional debug dump for native Gemini responses so we can inspect
    # the full shape (finish reasons, safety metadata, etc.) when models
    # return empty content. Controlled via the same env knobs used for
    # the OpenAI provider in Multieval.
    if os.getenv("EVALPLUS_DUMP_COMPLETION") == "1":
        try:
            dump_root = Path(os.getenv("EVALPLUS_DUMP_DIR", "."))
            dump_dir = dump_root.joinpath("evalplus_debug")
            dump_dir.mkdir(parents=True, exist_ok=True)
            dump_path = dump_dir / "last_gemini_response.json"
            try:
                payload = response.to_dict()  # type: ignore[attr-defined]
            except Exception:
                try:
                    payload = response._result  # type: ignore[attr-defined]
                except Exception:
                    payload = str(response)
            dump_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            # Never interfere with evaluation if debugging fails.
            pass

    return response


def make_auto_request(*args, **kwargs) -> genai_legacy.types.GenerateContentResponse:
    ret = None
    while ret is None:
        try:
            ret = make_request(*args, **kwargs)
        except ResourceExhausted as e:
            print("Rate limit exceeded. Waiting...", e.message)
            time.sleep(10)
        except GoogleAPICallError as e:
            print(e.message)
            time.sleep(1)
        except Exception:
            print("Unknown error. Waiting...")
            print_exc()
            time.sleep(1)
    return ret


class GeminiDecoder(DecoderBase):
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self.use_native = HAVE_NATIVE_GENAI

        if self.use_native:
            api_key = (
                os.getenv("GOOGLE_API_KEY")
                or os.getenv("GEMINI_API_KEY")
                or os.getenv("OPENAI_API_KEY")
            )
            if not api_key:
                raise RuntimeError("Missing GOOGLE_API_KEY/GEMINI_API_KEY/OPENAI_API_KEY for Gemini client")
            self.native_client = genai_native.Client(api_key=api_key)  # type: ignore[call-arg]
            self.client = None
        else:
            genai_legacy.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            self.client = genai_legacy.GenerativeModel(name)
            self.native_client = None

    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be positive for sampling"
        batch_size = min(self.batch_size, num_samples, 8)
        message = self.instruction_prefix + f"\n```python\n{prompt.strip()}\n```"

        # Native google-genai path with explicit thinking budget and
        # safety controls, mirroring the Shaberi/JP TL Bench helpers.
        if self.use_native and self.native_client is not None and genai_types is not None:
            evaluation_temperature = self.temperature if self.temperature is not None else 0.0

            safety_settings = [
                genai_types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                genai_types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                genai_types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                genai_types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
            ]
            # Gemini 3 Pro Preview requires a positive thinking budget;
            # a budget of 0 raises INVALID_ARGUMENT. Use a modest default
            # (128) so the model can operate in thinking mode without
            # consuming the entire context window on reasoning tokens.
            thinking_budget = 128
            thinking_config = genai_types.ThinkingConfig(thinking_budget=thinking_budget)

            gen_config = genai_types.GenerateContentConfig(
                temperature=evaluation_temperature,
                safety_settings=safety_settings,
                thinking_config=thinking_config,
                max_output_tokens=self.max_new_tokens,
            )

            ret_texts: List[str] = []
            for _ in range(batch_size):
                response = self.native_client.models.generate_content(  # type: ignore[union-attr]
                    model=self.name,
                    contents=message,
                    config=gen_config,
                )

                # Optional debug dump for the latest native response.
                if os.getenv("EVALPLUS_DUMP_COMPLETION") == "1":
                    try:
                        dump_root = Path(os.getenv("EVALPLUS_DUMP_DIR", "."))
                        dump_dir = dump_root.joinpath("evalplus_debug")
                        dump_dir.mkdir(parents=True, exist_ok=True)
                        dump_path = dump_dir / "last_gemini_response.json"
                        try:
                            payload = response.to_dict()  # type: ignore[attr-defined]
                        except Exception:
                            payload = str(response)
                        dump_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                    except Exception:
                        pass

                text = getattr(response, "text", None)
                if not text:
                    print("Empty response!")
                    ret_texts.append("")
                else:
                    ret_texts.append(str(text))

            return ret_texts + [""] * (batch_size - len(ret_texts))

        # Legacy google.generativeai path (used when google-genai is not
        # available in the environment).
        replies = make_auto_request(
            self.client,
            [{"role": "user", "content": message}],
            n=batch_size,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
        )

        if len(replies.candidates) != batch_size:
            print(
                f"WARNING: Expected {batch_size} outputs but got {len(replies.candidates)}"
            )

        ret_texts = []
        for candidate in replies.candidates:
            parts = candidate.content.parts
            if parts:
                ret_texts.append(parts[0].text)
            else:
                print("Empty response!")
                ret_texts.append("")
                print(f"{candidate.safety_ratings = }")

        return ret_texts + [""] * (batch_size - len(ret_texts))

    def is_direct_completion(self) -> bool:
        return False
