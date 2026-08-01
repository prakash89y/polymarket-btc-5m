"""Dependency vulnerability audit.

The problem this solves precisely: `pmbtc` is installed editable and is not
published to PyPI, so pip-audit cannot resolve it. Under `--strict` — which
means "fail if dependency *collection* fails on any dependency" — that is fatal.

The obvious fix, `--skip-editable`, does not work: skipping counts as a
collection failure, so `--strict --skip-editable` still fails. And it would be
the wrong mechanism anyway, because it skips *every* editable install. If a
third-party package were ever installed editable it would be silently dropped
from the audit, which is exactly the kind of quiet hole a security check must
not have.

So instead of subtracting the local package from the audit, this builds the
audit set explicitly: every installed **non-editable** distribution, pinned, and
audits precisely that with `--no-deps`. `pip list` already reports the full
transitive closure, so nothing is missed — the set is complete by construction
rather than by resolution.

Two guards keep the exclusion honest:

* the only editable distribution must be the local project, so "editable" cannot
  quietly come to mean "some dependency we forgot about";
* the audit set must be non-trivially large, so a broken `pip list` cannot pass
  the check by auditing nothing.

Strictness is fully preserved: any collection failure among the third-party
packages still fails the build, and any known vulnerability fails it too.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: The one distribution allowed to be editable: this project.
LOCAL_PACKAGE = "pmbtc"

#: A healthy environment has dozens of transitive dependencies. Far fewer means
#: `pip list` failed or the environment is not installed, and auditing three
#: packages must not read as success.
MIN_EXPECTED_PACKAGES = 20


def run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args, capture_output=True, text=True, check=False, **kwargs  # type: ignore[arg-type]
    )


def editable_distributions() -> list[str]:
    """Names of every distribution installed in editable mode."""
    result = run([sys.executable, "-m", "pip", "list", "--format=json"])
    if result.returncode != 0:
        print(f"ERROR: pip list failed:\n{result.stderr}", file=sys.stderr)
        raise SystemExit(2)
    return [
        package["name"]
        for package in json.loads(result.stdout)
        if package.get("editable_project_location")
    ]


def third_party_requirements() -> list[str]:
    """Pinned `name==version` for every installed non-editable distribution."""
    result = run(
        [sys.executable, "-m", "pip", "list", "--format=freeze", "--exclude-editable"]
    )
    if result.returncode != 0:
        print(f"ERROR: pip list failed:\n{result.stderr}", file=sys.stderr)
        raise SystemExit(2)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ignore-vuln", action="append", default=[],
        help="Vulnerability ID to ignore. Every use needs a written reason in "
             "SECURITY.md; unexplained suppressions are how a scan becomes theatre.",
    )
    args = parser.parse_args()

    print("1. Editable distributions")
    editable = editable_distributions()
    print(f"   found: {editable or 'none'}")
    unexpected = [name for name in editable if name.lower() != LOCAL_PACKAGE]
    if unexpected:
        # A third-party package installed editable would be invisible to the
        # audit set below, so refuse rather than quietly under-audit.
        print(
            f"   FAIL  unexpected editable distribution(s): {unexpected}. "
            "Only the local project may be editable, or the audit would silently "
            "skip a real dependency.",
            file=sys.stderr,
        )
        return 1
    print(f"   OK    only the local project ({LOCAL_PACKAGE}) is editable")

    print("\n2. Building the third-party audit set")
    requirements = third_party_requirements()
    print(f"   {len(requirements)} pinned package(s)")
    if any(line.lower().startswith(LOCAL_PACKAGE) for line in requirements):
        print(f"   FAIL  {LOCAL_PACKAGE} leaked into the audit set", file=sys.stderr)
        return 1
    if len(requirements) < MIN_EXPECTED_PACKAGES:
        print(
            f"   FAIL  only {len(requirements)} package(s); expected at least "
            f"{MIN_EXPECTED_PACKAGES}. Is the environment installed?",
            file=sys.stderr,
        )
        return 1
    print(f"   OK    {LOCAL_PACKAGE} excluded; set looks complete")

    print("\n3. Auditing")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "audit-requirements.txt"
        path.write_text("\n".join(requirements) + "\n", encoding="utf-8")

        command = [
            sys.executable, "-m", "pip_audit",
            # Strict: any dependency whose data cannot be collected fails the
            # run. Preserved exactly — it is only the *set* that changed.
            "--strict",
            # The set is already the full transitive closure from pip list, so
            # no further resolution is needed or wanted.
            "--no-deps",
            "--requirement", str(path),
            "--desc", "on",
        ]
        for vuln in args.ignore_vuln:
            command += ["--ignore-vuln", vuln]

        result = run(command)
        # pip-audit writes its findings to stderr, including the "no known
        # vulnerabilities" line, so both streams are surfaced.
        if result.stdout.strip():
            print(result.stdout.rstrip())
        if result.stderr.strip():
            print(result.stderr.rstrip())

    if result.returncode != 0:
        print(
            f"\nFAIL  audit reported vulnerabilities or a collection failure "
            f"(exit {result.returncode}).",
            file=sys.stderr,
        )
        return result.returncode

    print(f"\nOK    {len(requirements)} third-party package(s) audited, no known "
          "vulnerabilities.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
