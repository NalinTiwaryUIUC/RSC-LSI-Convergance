"""
Guardrails for Slurm/bash entrypoints: syntax, shellcheck, and nounset-safe optional env vars.

Run: pytest tests/test_shell_scripts.py -q
Install: pip install -r requirements.txt -r requirements-dev.txt
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _iter_shell_scripts() -> list[Path]:
    return sorted(SCRIPTS_DIR.glob("*.sh"))


def _find_shellcheck() -> str | None:
    import shutil

    p = shutil.which("shellcheck")
    if p:
        return p
    # pip install shellcheck-py places shellcheck next to the Python binary
    candidate = Path(sys.executable).resolve().parent / "shellcheck"
    if candidate.is_file():
        return str(candidate)
    return None


def _file_uses_nounset(content: str) -> bool:
    """True if the script enables bash nounset (-u), including via set -euo pipefail."""
    for line in content.splitlines():
        code = line.split("#", 1)[0].strip()
        if not code.startswith("set "):
            continue
        if "nounset" in code:
            return True
        # Flags like -u, -eu, -euo include 'u' as nounset
        if re.search(r"-[a-zA-Z]*u[a-zA-Z]*(\s|$)", code):
            return True
    return False


# Optional cluster/project env vars: expanding "$VAR" under set -u fails if unset.
_UNSAFE_OPTIONAL_ENV = re.compile(
    r"\[\s+-[nz]\s+"  # [ -n | [ -z
    r'"'
    r"\$("
    + "|".join(
        (
            "RSC_CONV_DIR",
            "SLURM_SUBMIT_DIR",
        )
    )
    + r')"'
    r"\s*\]"
)


def test_all_shell_scripts_pass_bash_syntax_check() -> None:
    for path in _iter_shell_scripts():
        r = subprocess.run(
            ["bash", "-n", str(path)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, f"bash -n failed for {path}:\n{r.stderr or r.stdout}"


def test_shellcheck_when_available() -> None:
    shellcheck = _find_shellcheck()
    if shellcheck is None:
        pytest.fail(
            "shellcheck not found. Install dev deps: pip install -r requirements-dev.txt "
            "(or install shellcheck from your OS package manager)."
        )
    for path in _iter_shell_scripts():
        r = subprocess.run(
            [shellcheck, "-S", "warning", str(path)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, (
            f"shellcheck -S warning failed for {path}:\n{r.stdout or r.stderr}"
        )


def test_nounset_scripts_do_not_expand_unset_optional_env_unsafely() -> None:
    """
    Under `set -u`, `[ -n "$RSC_CONV_DIR" ]` aborts before the test if the var is unset.
    Use `[ -n "${RSC_CONV_DIR:-}" ]` instead (same for SLURM_SUBMIT_DIR).
    """
    for path in _iter_shell_scripts():
        text = path.read_text(encoding="utf-8")
        if not _file_uses_nounset(text):
            continue
        m = _UNSAFE_OPTIONAL_ENV.search(text)
        assert m is None, (
            f"{path}: under nounset, optional env expansion must use "
            f'${{{m.group(1)}:-}} in [ -n/-z ] tests, not "${m.group(1)}" (match: {m.group(0)!r})'
        )


def test_run_pretrain_proj_root_block_matches_expected_safe_pattern() -> None:
    """Regression guard for the Slurm failure: RSC_CONV_DIR: unbound variable."""
    path = SCRIPTS_DIR / "run_pretrain.sh"
    text = path.read_text(encoding="utf-8")
    assert _file_uses_nounset(text), "run_pretrain.sh should use nounset (set -euo pipefail)"
    assert '[ -n "${RSC_CONV_DIR:-}" ]' in text
    assert '[ -n "${SLURM_SUBMIT_DIR:-}" ]' in text
    assert '[ -n "$RSC_CONV_DIR"' not in text
    assert '[ -n "$SLURM_SUBMIT_DIR"' not in text
