from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
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
    "codex": ProviderSpec(
        name="codex",
        model="gpt-5.6-terra",
        env_key="",
        base_url=None,
    ),
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


FINAL_SCHEMA = {
    "type": "object",
    "properties": {
        "final_choice": {"type": "string", "enum": ["A", "B", "C"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "brief_reason": {"type": "string"},
    },
    "required": ["final_choice", "confidence", "brief_reason"],
    "additionalProperties": False,
}

STATE_SCHEMA = {
    "type": "object",
    "properties": {
        "current_choice": {"type": "string", "enum": ["A", "B", "C"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "state_summary": {"type": "string"},
        "brief_reason": {"type": "string"},
    },
    "required": [
        "current_choice",
        "confidence",
        "state_summary",
        "brief_reason",
    ],
    "additionalProperties": False,
}

CODEX_TOOL_EVENT_TYPES = {
    "collab_tool_call",
    "command_execution",
    "file_change",
    "mcp_tool_call",
    "web_search",
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
    Provider client wrapper for Codex, OpenAI, Gemini, DeepSeek, and Claude.

    OpenAI, Gemini, and DeepSeek use OpenAI-compatible Chat Completions.
    Claude uses Anthropic's native Messages API. Codex uses an isolated,
    subscription-authenticated CLI agent and is recorded as a distinct
    execution mode.
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
        self.provider = provider
        self.model = model or spec.model

        if provider == "codex":
            self.client = None
            self.audit = JsonlAuditLog(
                audit_path or Path(f"{provider}_api_audit.jsonl")
            )
            return

        api_key = os.getenv(spec.env_key)
        if not api_key:
            raise RuntimeError(
                f"{spec.env_key} is not set. "
                f"Set it before running provider={provider}."
            )

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
        if self.provider == "codex":
            if shutil.which("codex") is None:
                return False
            try:
                login = subprocess.run(
                    ["codex", "login", "status"],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                if login.returncode != 0:
                    return False

                catalog = subprocess.run(
                    ["codex", "debug", "models"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if catalog.returncode != 0:
                    return False
                models = json.loads(catalog.stdout).get("models", [])
                return any(
                    isinstance(item, dict)
                    and item.get("slug") == self.model
                    for item in models
                )
            except (
                OSError,
                subprocess.SubprocessError,
                TypeError,
                ValueError,
            ):
                return False
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


def parse_json_text(text: str) -> dict:
    try:
        obj = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Model did not return valid JSON: {text[:500]!r}"
        ) from exc

    if not isinstance(obj, dict):
        raise ValueError("Model response must be a JSON object.")
    return obj


def _schema_for_stage(stage: Optional[str]) -> dict:
    if stage and stage.startswith("stateful_step_"):
        return STATE_SCHEMA
    return FINAL_SCHEMA


def _validate_response(obj: dict, stage: Optional[str]) -> None:
    schema = _schema_for_stage(stage)
    required = set(schema["required"])
    if set(obj) != required:
        raise ValueError(
            f"Invalid response fields: expected={sorted(required)}, "
            f"got={sorted(obj)}"
        )

    choice_key = (
        "current_choice" if "current_choice" in required else "final_choice"
    )
    if obj[choice_key] not in ("A", "B", "C"):
        raise ValueError(f"Invalid choice: {obj[choice_key]!r}")

    confidence = obj["confidence"]
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0 <= confidence <= 1
    ):
        raise ValueError(f"Invalid confidence: {confidence!r}")

    for name in required - {choice_key, "confidence"}:
        if not isinstance(obj[name], str):
            raise ValueError(f"Invalid {name}: expected a string.")


def _codex_auth_path() -> Path:
    codex_home = Path(
        os.environ.get("CODEX_HOME", Path.home() / ".codex")
    )
    auth_path = codex_home / "auth.json"
    if not auth_path.is_file():
        raise RuntimeError(
            "Codex subscription authentication was not found. "
            "Run `codex login` first; isolated execution requires auth.json."
        )
    return auth_path.resolve()


def _codex_prompt(input_messages: List[dict]) -> str:
    lines = [
        "Use only the supplied conversation. Do not call tools, inspect files, "
        "search the web, or delegate work."
    ]
    for msg in input_messages:
        role = msg.get("role", "user").upper()
        content = _flatten_content(msg.get("content", ""))
        lines.append(f"<{role}>\n{content}\n</{role}>")
    lines.append("Return only the required JSON object.")
    return "\n\n".join(lines)


def _run_codex(
    *,
    model: str,
    instructions: str,
    input_messages: List[dict],
    stage: Optional[str],
) -> tuple[str, dict, List[str]]:
    with tempfile.TemporaryDirectory(prefix="chronos-codex-") as tmp:
        root = Path(tmp)
        workspace = root / "workspace"
        isolated_home = root / "codex-home"
        workspace.mkdir()
        isolated_home.mkdir(mode=0o700)
        (isolated_home / "auth.json").symlink_to(_codex_auth_path())

        output_path = root / "response.json"
        schema_path = root / "schema.json"
        schema_path.write_text(
            json.dumps(_schema_for_stage(stage)),
            encoding="utf-8",
        )

        cmd = [
            "codex",
            "exec",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--skip-git-repo-check",
            "--json",
            "--sandbox",
            "read-only",
            "-C",
            str(workspace),
            "-c",
            f"developer_instructions={json.dumps(instructions or '')}",
            "-c",
            "project_doc_max_bytes=0",
            "-c",
            "features.plugins=false",
            "--output-schema",
            str(schema_path),
            "-o",
            str(output_path),
            "-m",
            model,
            _codex_prompt(input_messages),
        ]

        env = os.environ.copy()
        env["CODEX_HOME"] = str(isolated_home)
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                "codex exec timed out after 120 seconds"
            ) from exc

        if result.returncode != 0:
            error = result.stderr.strip()[-2000:] or "no stderr"
            raise RuntimeError(
                f"codex exec exited {result.returncode}: {error}"
            )

        events = [
            json.loads(line)
            for line in result.stdout.splitlines()
            if line.strip()
        ]
        tool_calls = sorted({
            event.get("item", {}).get("type")
            for event in events
            if event.get("type") in ("item.started", "item.completed")
            and event.get("item", {}).get("type")
            in CODEX_TOOL_EVENT_TYPES
        })
        if tool_calls:
            raise RuntimeError(
                "Codex used disallowed side-channel tools: "
                + ", ".join(tool_calls)
            )

        failures = [
            event
            for event in events
            if event.get("type") in ("error", "turn.failed")
        ]
        if failures:
            raise RuntimeError(f"Codex turn failed: {failures[-1]}")

        usage = {}
        for event in events:
            if event.get("type") == "turn.completed":
                usage = event.get("usage", {})

        text = (
            output_path.read_text(encoding="utf-8").strip()
            if output_path.exists()
            else ""
        )
        if not text:
            messages = [
                event.get("item", {}).get("text", "")
                for event in events
                if event.get("type") == "item.completed"
                and event.get("item", {}).get("type") == "agent_message"
            ]
            text = messages[-1].strip() if messages else ""

        return text, usage, tool_calls


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
    elif client.provider == "codex":
        request = {"model": model}
    else:
        request = {
            "model": model,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }

    attempts = 6
    last_exc = None
    for attempt in range(1, attempts + 1):
        started = time.monotonic()
        tool_calls: List[str] = []
        try:
            usage = {}
            if client.provider == "claude":
                response = client.client.messages.create(**request)
                text = "".join(
                    getattr(block, "text", "")
                    for block in response.content
                    if getattr(block, "type", None) == "text"
                )
                usage = _usage_dict(response)
            elif client.provider == "codex":
                text, usage, tool_calls = _run_codex(
                    model=model,
                    instructions=instructions,
                    input_messages=input_messages,
                    stage=stage,
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
                usage = _usage_dict(response)

            obj = parse_json_text(text)
            _validate_response(obj, stage)

        except Exception as exc:
            last_exc = exc
            error_entry = {
                "job_id": job_id,
                "stage": stage,
                "provider": client.provider,
                "model": model,
                "attempt": attempt,
                "latency_s": time.monotonic() - started,
                "error": repr(exc),
            }
            if client.provider == "codex":
                error_entry["execution_mode"] = "codex_cli_agent"
            client.audit.append(error_entry)
            if attempt < attempts:
                time.sleep(min(2 ** (attempt - 1), 20))
            continue

        success_entry = {
            "job_id": job_id,
            "stage": stage,
            "provider": client.provider,
            "model": model,
            "attempt": attempt,
            "latency_s": time.monotonic() - started,
            "usage": usage,
            "response": obj,
        }
        if client.provider == "codex":
            success_entry.update(
                {
                    "execution_mode": "codex_cli_agent",
                    "side_channel_tool_calls": tool_calls,
                }
            )
        client.audit.append(success_entry)

        if ledger is not None:
            ledger.record(job_id or "", stage or "")

        if pause_s:
            time.sleep(pause_s)
        return obj

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
