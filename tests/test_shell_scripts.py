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


def _slurm_batch_script_paths() -> list[Path]:
    """Scripts with real #SBATCH directives (first ~30 lines)."""
    out: list[Path] = []
    for path in _iter_shell_scripts():
        lines = path.read_text(encoding="utf-8").splitlines()[:30]
        if any(l.startswith("#SBATCH --") for l in lines):
            out.append(path)
    return out


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


# Under set -u, `[ -n "$VAR" ]` / `[[ -n "$VAR" ]]` expands $VAR before the test; unset => error.
# Safe: `[ -n "${VAR:-}" ]` (also applies to -z).
_UNSAFE_BARE_DOLLAR_IN_TEST = re.compile(r'(?:\[|\[\[)\s+-[nz]\s+"\$(?!\{)')


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


def test_nounset_scripts_do_not_use_unsafe_test_expansions() -> None:
    """
    Catches WIDTH/RSC_CONV_DIR-style failures: under nounset, `[ -n "$VAR" ]` is invalid
    when VAR may be unset; use `[ -n "${VAR:-}" ]` or assign defaults first.
    """
    for path in _iter_shell_scripts():
        text = path.read_text(encoding="utf-8")
        if not _file_uses_nounset(text):
            continue
        bad: list[str] = []
        for line in text.splitlines():
            code = line.split("#", 1)[0]
            if _UNSAFE_BARE_DOLLAR_IN_TEST.search(code):
                bad.append(line.strip())
        assert not bad, f"{path}: unsafe [ -n/-z ] expansion under nounset:\n" + "\n".join(
            bad
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


def test_run_pretrain_binds_optional_env_before_use() -> None:
    """
    Defense in depth: optional env vars must be assigned VAR="${VAR:-}" before any
    ARGS+=(... "$VAR") under set -u (catches WIDTH-style bugs that static [ -n ] scans miss).
    """
    path = SCRIPTS_DIR / "run_pretrain.sh"
    text = path.read_text(encoding="utf-8")
    for name in (
        "WIDTH",
        "N_TRAIN",
        "ALPHA",
        "PRETRAIN_STEPS",
        "PRETRAIN_LR",
        "PRETRAIN_WEIGHT_DECAY",
        "OUTPUT",
        "ARCH",
        "NUM_BLOCKS",
        "SNAPSHOT_STEPS",
        "SNAPSHOT_EVERY",
        "SNAPSHOT_DIR",
        "DATA_DIR",
        "ROOT",
        "DATASET_SEED",
        "PRETRAIN_SEED",
        "BN_CALIBRATION_MB",
        "VERIFY",
    ):
        assert f'{name}="${{{name}:-}}"' in text, f"run_pretrain.sh must bind {name} before ARGS block"


def test_slurm_batch_scripts_resolve_proj_dir_for_spool_execution() -> None:
    """
    Slurm executes a *copy* of the batch script under /var/spool/slurmd/...; dirname(BASH_SOURCE)/..
    is then NOT the repo. Every Slurm batch entrypoint must prefer RSC_CONV_DIR, then SLURM_SUBMIT_DIR
    (directory from which sbatch was run), then BASH_SOURCE fallback (local ./scripts/foo.sh).
    """
    required_substrings = (
        '[ -n "${RSC_CONV_DIR:-}" ]',
        '[ -n "${SLURM_SUBMIT_DIR:-}" ]',
        'PROJ_DIR="$RSC_CONV_DIR"',
        'PROJ_DIR="$SLURM_SUBMIT_DIR"',
        'dirname "${BASH_SOURCE[0]}"',
    )
    for path in _slurm_batch_script_paths():
        text = path.read_text(encoding="utf-8")
        missing = [s for s in required_substrings if s not in text]
        assert not missing, (
            f"{path.name}: add the standard PROJ_DIR block (see run_pretrain.sh). Missing fragments: {missing}"
        )


def test_proj_dir_prefers_slurm_submit_dir_when_set() -> None:
    """Runtime: when SLURM_SUBMIT_DIR is set, PROJ_DIR must match it (simulates cluster submit dir)."""
    snippet = r"""
set -euo pipefail
if [ -n "${RSC_CONV_DIR:-}" ]; then
    PROJ_DIR="$RSC_CONV_DIR"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJ_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJ_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
printf '%s' "$PROJ_DIR"
"""
    want = str(REPO_ROOT)
    r = subprocess.run(
        ["bash", "-c", snippet],
        cwd=str(REPO_ROOT),
        env={
            "PATH": "/usr/bin:/bin",
            "HOME": "/tmp",
            "SLURM_SUBMIT_DIR": want,
        },
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout == want, f"expected PROJ_DIR={want!r}, got {r.stdout!r}"


def test_run_pretrain_args_block_runs_under_nounset_with_empty_env() -> None:
    """Runtime smoke: same logic as ARGS build, with every optional name unset."""
    snippet = r"""
set -euo pipefail
WIDTH="${WIDTH:-}"
N_TRAIN="${N_TRAIN:-}"
[ -n "${WIDTH:-}" ] && ARGS+=(--width "$WIDTH")
[ -n "${N_TRAIN:-}" ] && ARGS+=(--n_train "$N_TRAIN")
: "${ARGS[@]:-}"
"""
    r = subprocess.run(
        ["bash", "-c", snippet],
        cwd=str(REPO_ROOT),
        env={"PATH": "/usr/bin:/bin", "HOME": "/tmp"},
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr + r.stdout
