"""Repository hygiene checks — run in CI and before any commit.

Answers one question: could anything sensitive, enormous, or unreproducible
reach a commit? Ignore rules are easy to get subtly wrong, and a secret in git
history is effectively permanent, so this is asserted rather than assumed.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Anything above this is almost certainly data that should not be versioned.
MAX_TRACKED_FILE_KB = 512
MAX_TOTAL_TRACKED_MB = 20

#: Paths that must never appear in the index, whatever the ignore file says.
FORBIDDEN_PATTERNS = [
    re.compile(r"(^|/)\.env$"),
    re.compile(r"(^|/)\.env\.(?!example)"),
    re.compile(r"\.(pem|key|p12|pfx|keystore)$"),
    re.compile(r"(^|/)(data|logs|artifacts)/"),
    re.compile(r"\.(pkl|joblib|onnx|pt|pth|h5|ckpt|safetensors)$"),
    re.compile(r"\.venv/"),
    re.compile(r"\.parquet$"),
]

#: High-signal secret shapes. Deliberately narrow: a scanner that cries wolf
#: gets disabled, and then it protects nothing.
SECRET_PATTERNS = [
    # A bare 0x + 64 hex is NOT a reliable private-key signal: it is also the
    # shape of every Ethereum transaction hash and every Polymarket
    # conditionId, both of which are public and appear throughout the fixtures.
    # Require an actual credential context instead.
    (
        re.compile(
            r"(?i)(private[_-]?key|privkey|secret[_-]?key|signing[_-]?key)"
            r"\s*[=:]\s*['\"]?0x[a-fA-F0-9]{64}"
        ),
        "private key",
    ),
    (re.compile(r"(?i)(api[_-]?key|secret|passphrase|password)\s*[=:]\s*['\"][^'\"\s]{12,}"),
     "hard-coded credential"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "PEM private key"),
    (re.compile(r"(?i)aws_secret_access_key\s*[=:]"), "AWS secret"),
    (re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"), "GitHub token"),
]

#: Files allowed to contain secret-shaped text: examples, docs, and the
#: scanner's own pattern list.
SECRET_SCAN_EXEMPT = {
    ".env.example",
    "scripts/validate_repo_hygiene.py",
    "docs/OPERATIONS.md",
    "SECURITY.md",
}

FAILURES: list[str] = []


def check(name: str, passed: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if passed else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))
    if not passed:
        FAILURES.append(f"{name}: {detail}")
    return passed


def tracked_files() -> list[str]:
    """Files git would actually commit, honouring .gitignore."""
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def main() -> int:
    files = tracked_files()
    print(f"\n{len(files)} file(s) would be tracked\n")

    print("1. Forbidden paths")
    offenders = [
        f for f in files if any(pattern.search(f) for pattern in FORBIDDEN_PATTERNS)
    ]
    check("no secrets, data, logs, or model artifacts tracked", not offenders,
          ", ".join(offenders[:5]))

    print("\n2. File sizes")
    oversized = []
    total = 0
    for name in files:
        path = ROOT / name
        if not path.is_file():
            continue
        size = path.stat().st_size
        total += size
        if size > MAX_TRACKED_FILE_KB * 1024:
            oversized.append(f"{name} ({size / 1024:.0f} KB)")
    check(f"no file exceeds {MAX_TRACKED_FILE_KB} KB", not oversized,
          ", ".join(oversized[:3]))
    check(f"repository under {MAX_TOTAL_TRACKED_MB} MB", total < MAX_TOTAL_TRACKED_MB * 1024**2,
          f"{total / 1024**2:.2f} MB")

    print("\n3. Secret scan of tracked content")
    found: list[str] = []
    for name in files:
        if name in SECRET_SCAN_EXEMPT:
            continue
        path = ROOT / name
        if not path.is_file() or path.suffix in {".gz", ".parquet", ".pkl", ".png"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for pattern, label in SECRET_PATTERNS:
            match = pattern.search(text)
            if match:
                found.append(f"{name}: {label}")
                break
    check("no credential-shaped strings in tracked files", not found,
          "; ".join(found[:5]))

    print("\n4. Required files present")
    for required in (".gitignore", ".gitattributes", "README.md", "pyproject.toml"):
        check(f"{required} exists", (ROOT / required).is_file())

    print("\n5. Fixtures needed by the determinism tests are tracked")
    for needed in ("tests/fixtures/replay_baseline.json",):
        check(f"{needed} tracked", needed in files)
    archive_fixtures = [f for f in files if f.startswith("tests/fixtures/archive/")]
    check("archive fixtures tracked", bool(archive_fixtures),
          f"{len(archive_fixtures)} file(s)")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} hygiene check(s) failed:")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print("Repository hygiene OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
