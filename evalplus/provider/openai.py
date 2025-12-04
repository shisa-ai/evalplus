import json
import os
from pathlib import Path
from typing import List

import httpx
import openai

from evalplus.gen.util import openai_request
from evalplus.provider.base import DecoderBase
from evalplus.provider.utility import concurrent_call


class OpenAIChatDecoder(DecoderBase):
    def __init__(
        self, name: str, base_url=None, verify_certificate=True, **kwargs
    ) -> None:
        super().__init__(name, **kwargs)
        self.base_url = base_url
        self.verify_certificate = verify_certificate

    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be positive for sampling"
        batch_size = min(self.batch_size, num_samples)
        prompt = self.instruction_prefix + f"\n```python\n{prompt.strip()}\n```"

        # use concurrency based batching for o1 and deepseek models
        if self.name.startswith("o1-") or self.name == "deepseek-chat":
            return self._codegen_batch_via_concurrency(prompt, num_samples)

        return self._codegen_api_batch(prompt, batch_size)

    def _codegen_api_batch(self, prompt: str, batch_size: int) -> List[str]:
        client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "none"),
            base_url=self.base_url,
            http_client=httpx.Client(verify=self.verify_certificate),
        )

        ret = openai_request.make_auto_request(
            client,
            message=prompt,
            model=self.name,
            # For OpenAI-compatible chat completions (including Gemini via
            # the /v1beta/openai bridge), use max_tokens rather than the
            # newer max_completion_tokens knob. This matches how other
            # multieval integrations talk to Gemini and avoids payload
            # field mismatches on stricter gateways.
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            n=batch_size,
        )

        # Optional debug dump: when EVALPLUS_DUMP_COMPLETION=1 is set in
        # the environment, write the raw completion payload for the first
        # batch to disk so we can inspect how OpenAI-compatible backends
        # (e.g., Gemini via /v1beta/openai) shape message/content fields.
        if os.getenv("EVALPLUS_DUMP_COMPLETION") == "1":
            try:
                dump_dir = Path(os.getenv("EVALPLUS_DUMP_DIR", ".")).joinpath(
                    "evalplus_debug"
                )
                dump_dir.mkdir(parents=True, exist_ok=True)
                dump_path = dump_dir / "last_completion.json"
                dump_path.write_text(
                    json.dumps(ret.model_dump(), indent=2),
                    encoding="utf-8",
                )
            except Exception:
                # Never let debugging interfere with evaluation.
                pass

        outputs = []
        for item in ret.choices:
            message = getattr(item, "message", None)
            value = None

            if message is not None:
                # Primary content channel
                value = getattr(message, "content", None)

                # Some OpenAI-compatible backends (including Gemini via
                # the /v1beta/openai bridge) may instead populate a
                # `refusal` field when content is blocked.
                if value is None:
                    refusal = getattr(message, "refusal", None)
                    if isinstance(refusal, str):
                        value = refusal

                # Handle list-style content blocks (Responses/content
                # parts) where each part may be a dict or an object
                # exposing a `.text` attribute.
                if isinstance(value, list):
                    parts = []
                    for part in value:
                        text = None
                        if isinstance(part, dict):
                            text = part.get("text")
                        else:
                            text = getattr(part, "text", None)
                        if text:
                            # Newer SDKs wrap text in an object with a
                            # `.value` attribute.
                            text_value = getattr(text, "value", text)
                            parts.append(str(text_value))
                    value = "".join(parts) if parts else ""

            # Ensure downstream code always receives a string. When the
            # backend returns neither content nor refusal text we treat
            # it as an empty solution so EvalPlus marks the sample as a
            # failure rather than crashing.
            outputs.append(value if isinstance(value, str) else ("" if value is None else str(value)))

        return outputs

    def _codegen_batch_via_concurrency(self, prompt: str, batch_size: int) -> List[str]:
        batches = concurrent_call(
            batch_size, self._codegen_api_batch, prompt, batch_size=1
        )
        return [b[0] for b in batches]

    def is_direct_completion(self) -> bool:
        return False
