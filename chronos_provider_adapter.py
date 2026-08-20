from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from openai import OpenAI

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    model: str
    env_key: str
    base_url: Optional[str]


PROVIDERS: Dict[str, ProviderSpec] = {
    "openai": ProviderSpec(
        name="openai",
        model="gpt-5.6-sol",
        env_key="OPENAI_API_KEY",
        base_url=None,
    ),
    "gemini": ProviderSpec(
        name="gemini",
        model="gemini-3.6-flash",
        env_key="GEMINI_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    ),
    "deepseek": ProviderSpec(
        name="deepseek",
        model="deepseek-v4-flash",
        env_key="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com",
    ),
    "claude": ProviderSpec(
        name="claude",
        model="claude-sonnet-4-6",
        env_key="ANTHROPIC_API_KEY",
        base_url=None,
    ),
}


class JsonlAuditLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, obj: dict) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


class ProviderClient:
    """
    Provider client wrapper for OpenAI, Gemini, DeepSeek, and Claude.

    OpenAI, Gemini, and DeepSeek use OpenAI-compatible Chat Completions.
    Claude uses Anthropic's native Messages API. The adapter preserves the
    same system/task text and user/assistant message trajectory.
    """

    def __init__(
        self,
        provider: str,
        model: Optional[str] = None,
        max_retries: int = 6,
        audit_path: Optional[Path] = None,
    ):
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}")

        spec = PROVIDERS[provider]
        api_key = os.getenv(spec.env_key)
        if not api_key:
            raise RuntimeError(
                f"{spec.env_key} is not set. "
                f"Set it before running provider={provider}."
            )

        self.provider = provider
        self.model = model or spec.model

        if provider == "claude":
            if Anthropic is None:
                raise RuntimeError(
                    "The anthropic package is not installed. Run: "
                    "python -m pip install -U anthropic"
                )
            self.client = Anthropic(
                api_key=api_key,
                max_retries=max_retries,
                timeout=120.0,
            )
        else:
            kwargs = {
                "api_key": api_key,
                "max_retries": max_retries,
                "timeout": 120.0,
            }
            if spec.base_url:
                kwargs["base_url"] = spec.base_url
            self.client = OpenAI(**kwargs)

        self.audit = JsonlAuditLog(
            audit_path or Path(f"{provider}_api_audit.jsonl")
        )

    def model_exists(self) -> bool:
        """
        Best-effort preflight. Some accounts/providers may restrict model-list
        access, so a failed preflight is not treated as proof of unavailability.
        """
        try:
            if self.provider == "claude":
                # Anthropic's Models API uses model_id in current SDKs.
                try:
                    self.client.models.retrieve(model_id=self.model)
                except TypeError:
                    self.client.models.retrieve(self.model)
            else:
                self.client.models.retrieve(self.model)
            return True
        except Exception:
            return False


def _flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                # Responses-style text item
                if item.get("type") in ("input_text", "text"):
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(x for x in parts if x)

    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)

    return str(content)


def normalize_messages(
    instructions: Optional[str],
    input_messages: Iterable[dict],
) -> List[dict]:
    messages: List[dict] = []
    if instructions:
        messages.append({"role": "system", "content": instructions})

    for msg in input_messages:
        role = msg.get("role", "user")
        content = _flatten_content(msg.get("content", ""))
        messages.append({"role": role, "content": content})

    return messages


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_json_text(text: str) -> dict:
    text = _strip_code_fence(text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Conservative recovery for providers that prepend a short sentence.
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start:end + 1])

    raise ValueError(f"Model did not return valid JSON: {text[:500]!r}")


def _usage_dict(response: Any) -> dict:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}

    out = {}
    for name in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "input_tokens",
        "output_tokens",
    ):
        value = getattr(usage, name, None)
        if value is not None:
            out[name] = value
    return out


def call_json_compat(
    *,
    client: ProviderClient,
    model: str,
    instructions: str,
    input_messages: List[dict],
    ledger=None,
    job_id: Optional[str] = None,
    stage: Optional[str] = None,
    pause_s: float = 0.0,
    **_: Any,
) -> dict:
    """
    Drop-in replacement for chronos_v8.call_json.

    The signature intentionally matches the original helper so that existing
    chronos_v8.run_stateful_trial() and run_control_trial() can be reused
    without changing their prompts or agent-state logic.
    """
    messages = normalize_messages(instructions, input_messages)

    # Every Chronos prompt already requests JSON. Adding this line makes JSON
    # mode robust across providers without changing the task content.
    if messages:
        messages[-1]["content"] += (
            "\n\nReturn exactly one valid JSON object and no markdown."
        )

    if client.provider == "claude":
        # Anthropic Messages API takes the system prompt separately and does
        # not accept role="system" inside messages. Keep the same task text
        # and user/assistant trajectory used by the other providers.
        claude_messages = normalize_messages(None, input_messages)
        if claude_messages:
            claude_messages[-1]["content"] += (
                "\n\nReturn exactly one valid JSON object and no markdown."
            )
        request = {
            "model": model,
            "system": instructions or "",
            "messages": claude_messages,
            "max_tokens": 1024,
        }
    else:
        request = {
            "model": model,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }

    attempts = 6
    last_exc = None
    started = time.time()

    for attempt in range(1, attempts + 1):
        try:
            if client.provider == "claude":
                response = client.client.messages.create(**request)
                text = "".join(
                    getattr(block, "text", "")
                    for block in response.content
                    if getattr(block, "type", None) == "text"
                )
            else:
                try:
                    response = client.client.chat.completions.create(**request)
                except Exception as first_error:
                    # Some compatibility layers may reject response_format even
                    # though the model can still obey the JSON-only instruction.
                    if "response_format" not in str(first_error).lower():
                        raise
                    fallback = dict(request)
                    fallback.pop("response_format", None)
                    response = client.client.chat.completions.create(**fallback)

                message = response.choices[0].message
                text = message.content or ""

            obj = parse_json_text(text)

            client.audit.append({
                "job_id": job_id,
                "stage": stage,
                "provider": client.provider,
                "model": model,
                "attempt": attempt,
                "latency_s": time.time() - started,
                "usage": _usage_dict(response),
                "response": obj,
            })

            if pause_s:
                time.sleep(pause_s)
            return obj

        except Exception as exc:
            last_exc = exc
            client.audit.append({
                "job_id": job_id,
                "stage": stage,
                "provider": client.provider,
                "model": model,
                "attempt": attempt,
                "error": repr(exc),
            })
            if attempt < attempts:
                time.sleep(min(2 ** (attempt - 1), 20))

    raise RuntimeError(
        f"API call failed after {attempts} attempts "
        f"for {client.provider}/{model}: {last_exc}"
    )


def patch_chronos_v8(v8_module) -> None:
    """
    Preserve chronos_v8's exact experiment prompts and stateful logic, but
    route every LLM call through the multi-provider adapter.
    """
    v8_module.call_json = call_json_compat
