"""End-to-end validation of the GitHub integration.

Checks that the workflow is actually wired, not merely present: that every
required CI check exists and feeds the aggregate gate, that provenance reaches a
model card, and that nothing sensitive can be committed.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
FAILURES: list[str] = []


def check(name: str, passed: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if passed else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))
    if not passed:
        FAILURES.append(f"{name}: {detail}")
    return passed


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip()


def main() -> int:
    print("\n1. Repository")
    check("git repository initialised", (ROOT / ".git").is_dir())
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    check("on main", branch == "main", branch)
    branches = set(git("branch", "--format=%(refname:short)").split())
    check("develop branch exists", "develop" in branches, ", ".join(sorted(branches)))
    tags = git("tag").split()
    check("release tag present", any(t.startswith("v") for t in tags), ", ".join(tags))
    check("working tree clean", not git("status", "--porcelain"),
          git("status", "--porcelain")[:120])

    print("\n2. Required files")
    required = [
        ".gitignore", ".gitattributes", "CHANGELOG.md", "CONTRIBUTING.md",
        "SECURITY.md", "mkdocs.yml",
        ".github/CODEOWNERS", ".github/dependabot.yml",
        ".github/pull_request_template.md", ".github/branch-protection.md",
        ".github/ISSUE_TEMPLATE/bug_report.yml",
        ".github/ISSUE_TEMPLATE/feature_request.yml",
        ".github/ISSUE_TEMPLATE/config.yml",
        ".github/workflows/ci.yml", ".github/workflows/release.yml",
        ".github/workflows/docs.yml",
        "docs/GITHUB_WORKFLOW.md", "docs/ARCHITECTURE.md",
        "docs/OPERATIONS.md", "docs/DEPLOYMENT.md",
    ]
    for name in required:
        check(name, (ROOT / name).is_file())

    print("\n3. CI covers every required check")
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    text = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    for label, needle in [
        ("Ruff", "ruff check"),
        ("MyPy", "mypy"),
        ("unit tests", "pytest -q -m \"not network\""),
        ("integration tests", "test_integration_pipeline"),
        ("replay determinism", "TestCommittedBaseline"),
        ("feature schema validation", "validate_schemas.py"),
        ("dataset schema validation", "TestLeakageDetection"),
        ("model registry validation", "test_training.py"),
        ("readiness validation", "check_readiness"),
        ("secret scan", "gitleaks"),
        ("dependency scan", "pip-audit"),
    ]:
        check(f"CI runs {label}", needle in text)

    jobs = set(workflow["jobs"])
    gate = set(workflow["jobs"]["ci-complete"]["needs"])
    check("aggregate gate depends on every job", gate == jobs - {"ci-complete"},
          f"{len(gate)} of {len(jobs) - 1}")

    print("\n4. Experiment tracking")
    from pmbtc.models.registry import ModelCard
    from pmbtc.ops.gitinfo import collect

    provenance = collect(ROOT)
    check("git provenance available", provenance.available, provenance.describe)
    check("commit recorded", len(provenance.commit) == 40, provenance.short_commit)
    check("branch recorded", provenance.branch == "main", provenance.branch)
    check("tag recorded", bool(provenance.tag), provenance.tag or "(none)")
    check("clean tree is reproducible", provenance.reproducible)

    card = ModelCard(
        model_id="x", model_type="t", dataset_version="d", feature_schema_version="f",
        config_hash="c", hyperparameter_hash="h", trained_at_ms=0,
        git_commit=provenance.commit, git_branch=provenance.branch,
        git_tag=provenance.tag, git_dirty=provenance.dirty,
    )
    payload = card.as_dict()
    check("model card carries git block",
          set(payload["git"]) >= {"commit", "branch", "tag", "dirty", "reproducible"})
    check("commit is part of experiment identity",
          card.reproducibility_key() != ModelCard(
              model_id="x", model_type="t", dataset_version="d",
              feature_schema_version="f", config_hash="c",
              hyperparameter_hash="h", trained_at_ms=0,
          ).reproducibility_key())

    print("\n5. Nothing sensitive is committed")
    tracked = git("ls-files").splitlines()
    forbidden = [
        f for f in tracked
        if re.search(r"(^|/)(data|logs|artifacts)/|\.env$|\.(pem|key|pkl)$", f)
    ]
    check("no data, logs, secrets, or model artifacts tracked", not forbidden,
          ", ".join(forbidden[:5]))
    check("tracked file count is sane", 50 < len(tracked) < 400, f"{len(tracked)} files")

    print("\n6. Large-file policy")
    check("Git LFS not enabled", not (ROOT / ".lfsconfig").exists()
          and "filter=lfs" not in (ROOT / ".gitattributes").read_text(encoding="utf-8"))
    check("LFS decision documented",
          "Git LFS is deliberately not enabled"
          in (ROOT / "docs/GITHUB_WORKFLOW.md").read_text(encoding="utf-8"))

    print("\n7. Documentation site")
    mkdocs = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    check("mkdocs strict mode enabled", "strict: true" in mkdocs)

    # Every root-level document referenced by the site must exist inside docs/,
    # because strict mode refuses a link that escapes the docs tree.
    for mirrored in ("CHANGELOG.md", "CONTRIBUTING.md", "SECURITY.md",
                     "BRANCH_PROTECTION.md"):
        path = ROOT / "docs" / mirrored
        check(f"docs/{mirrored} mirrored", path.is_file())
        if path.is_file():
            check(f"docs/{mirrored} marked generated",
                  path.read_text(encoding="utf-8").startswith("<!-- Generated from"))

    link_pattern = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
    escaping: list[str] = []
    for doc in sorted((ROOT / "docs").rglob("*.md")):
        for match in link_pattern.finditer(doc.read_text(encoding="utf-8")):
            target = match.group(1).strip()
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            candidate, _, _ = target.partition("#")
            if candidate and not (doc.parent / candidate).resolve().is_file():
                escaping.append(f"{doc.name} -> {target}")
    check("every internal docs link resolves", not escaping, "; ".join(escaping[:5]))

    print("\n8. Versioning")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    versions = re.findall(r"^## \[(\d+\.\d+\.\d+)\]", changelog, re.M)
    check("changelog has module releases", len(versions) >= 7, ", ".join(versions))
    for tag in tags:
        if tag.startswith("v"):
            check(f"tag {tag} has a changelog section", tag[1:] in versions)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed:")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print("GitHub integration validated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
