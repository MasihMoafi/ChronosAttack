import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import chronos_multimodel_full as runner
from chronos_provider_adapter import parse_json_text


class FakeProviderClient:
    def __init__(self, provider, model, audit_path):
        self.provider = provider
        self.model = model

    def model_exists(self):
        return True


class ChronosExtensionTests(unittest.TestCase):
    def run_preflight(self, *args):
        argv = [
            "chronos_multimodel_full.py",
            "--providers",
            "codex",
            "--preflight-only",
            *args,
        ]
        with (
            patch.object(sys, "argv", argv),
            patch.object(runner, "ProviderClient", FakeProviderClient),
            redirect_stdout(StringIO()),
        ):
            runner.main()

    def test_json_parser_requires_one_bare_object(self):
        expected = {
            "final_choice": "A",
            "confidence": 0.5,
            "brief_reason": "test",
        }
        text = json.dumps(expected)
        self.assertEqual(parse_json_text(text), expected)
        for wrapped in (f"```json\n{text}\n```", f"prefix {text} suffix"):
            with self.assertRaises(ValueError):
                parse_json_text(wrapped)

    def test_resume_accepts_only_matching_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            common = ("--resume", "--out", str(out))
            self.run_preflight(*common)
            self.run_preflight(*common)
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                self.run_preflight(
                    *common,
                    "--codex-model",
                    "gpt-5.6-luna",
                )

    def test_resume_rejects_nonempty_directory_without_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            out.mkdir()
            (out / "checkpoint.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "without run_manifest"):
                self.run_preflight("--resume", "--out", str(out))

    def test_tmux_runner_resumes_only_after_manifest_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bin_dir = root / "bin"
            out = root / "run"
            bin_dir.mkdir()

            fake_tmux = bin_dir / "tmux"
            fake_tmux.write_text(
                """#!/usr/bin/env bash
if [[ $1 == has-session ]]; then exit 1; fi
while [[ $1 != bash ]]; do shift; done
exec "$@"
""",
                encoding="utf-8",
            )
            fake_tmux.chmod(0o755)

            fake_python = bin_dir / "python3"
            fake_python.write_text(
                """#!/usr/bin/env bash
resume=0
while (( $# )); do
    case $1 in
        --out) out=$2; shift 2 ;;
        --resume) resume=1; shift ;;
        *) shift ;;
    esac
done
if (( resume )) && [[ ! -f $out/run_manifest.json ]]; then exit 23; fi
if (( resume )); then
    echo resume
else
    echo '{}' > "$out/run_manifest.json"
    echo fresh
fi
""",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)

            env = os.environ.copy()
            env["PATH"] = f"{bin_dir}:{env['PATH']}"
            command = [
                str(Path("run_experiments_tmux.sh").resolve()),
                str(out),
                "barrier",
                "gpt-5.6-terra",
            ]
            first = subprocess.run(command, capture_output=True, text=True, env=env)
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            second = subprocess.run(command, capture_output=True, text=True, env=env)
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            self.assertEqual(
                (out / "runner.log").read_text(encoding="utf-8").strip(),
                "resume",
            )


if __name__ == "__main__":
    unittest.main()
