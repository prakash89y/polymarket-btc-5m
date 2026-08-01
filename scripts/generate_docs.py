"""Generate documentation from the code, so it cannot drift.

Everything here is derived: the feature dictionary from the registry, the module
map from package docstrings, the API reference from signatures, and the schema
table from the running versions. A hand-maintained version of any of these is
wrong within a week.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

#: Documents that live at the repository root because that is where GitHub and
#: contributors expect them, but which also belong in the published site.
#:
#: MkDocs runs in strict mode and refuses links that escape `docs/` — correctly,
#: because such a link is broken on the published site even though it resolves
#: on GitHub. So rather than weakening the check, the sources are mirrored into
#: `docs/` at generation time and their internal links are rewritten to match
#: the flattened layout.
#:
#: Source of truth stays at the root. The copies are generated, and CI fails if
#: they drift.
MIRRORED_DOCUMENTS: dict[str, str] = {
    "CHANGELOG.md": "CHANGELOG.md",
    "CONTRIBUTING.md": "CONTRIBUTING.md",
    "SECURITY.md": "SECURITY.md",
    ".github/branch-protection.md": "BRANCH_PROTECTION.md",
}

#: Link rewrites applied to mirrored content. Each source path becomes the
#: name it has inside `docs/`.
_LINK_REWRITES: dict[str, str] = {
    "docs/GITHUB_WORKFLOW.md": "GITHUB_WORKFLOW.md",
    "docs/ARCHITECTURE.md": "ARCHITECTURE.md",
    "docs/OPERATIONS.md": "OPERATIONS.md",
    "docs/DEPLOYMENT.md": "DEPLOYMENT.md",
    "docs/features.md": "features.md",
    "../.github/branch-protection.md": "BRANCH_PROTECTION.md",
    ".github/branch-protection.md": "BRANCH_PROTECTION.md",
    "../CHANGELOG.md": "CHANGELOG.md",
    "../CONTRIBUTING.md": "CONTRIBUTING.md",
    "../SECURITY.md": "SECURITY.md",
}

_MARKDOWN_LINK = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")


def rewrite_links(text: str) -> str:
    """Point markdown links at their location inside the flattened docs tree."""

    def replace(match: re.Match[str]) -> str:
        label, target = match.group(1), match.group(2)
        # Split any anchor or title so only the path is rewritten.
        path, _, suffix = target.partition("#")
        rewritten = _LINK_REWRITES.get(path.strip())
        if rewritten is None:
            return match.group(0)
        return f"[{label}]({rewritten}{'#' + suffix if suffix else ''})"

    return _MARKDOWN_LINK.sub(replace, text)


def mirror_root_documents() -> list[Path]:
    """Copy root-level documents into `docs/`, rewriting their links.

    A banner marks each copy as generated, so an editor who lands on it from the
    site knows to change the source instead.
    """
    written: list[Path] = []
    for source_name, target_name in MIRRORED_DOCUMENTS.items():
        source = ROOT / source_name
        if not source.is_file():
            print(f"  WARNING: {source_name} missing; skipping mirror")
            continue
        banner = (
            f"<!-- Generated from {source_name} by scripts/generate_docs.py. "
            "Edit the source, not this copy. -->\n\n"
        )
        target = DOCS / target_name
        target.write_text(banner + rewrite_links(source.read_text(encoding="utf-8")),
                          encoding="utf-8")
        written.append(target)
    return written


def check_internal_links() -> list[str]:
    """Every relative markdown link in `docs/` must resolve inside `docs/`.

    This is the invariant MkDocs enforces in strict mode, checked here too so a
    broken link is caught by `generate_docs.py` rather than only by a CI job
    that runs later.
    """
    problems: list[str] = []
    for path in sorted(DOCS.rglob("*.md")):
        for match in _MARKDOWN_LINK.finditer(path.read_text(encoding="utf-8")):
            target = match.group(2).strip()
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            candidate, _, _ = target.partition("#")
            if not candidate:
                continue
            resolved = (path.parent / candidate).resolve()
            if not resolved.exists():
                problems.append(f"{path.relative_to(ROOT)} -> {target}")
            elif DOCS.resolve() not in resolved.parents and resolved != DOCS.resolve():
                problems.append(
                    f"{path.relative_to(ROOT)} -> {target} (escapes docs/)"
                )
    return problems


def generate_feature_docs() -> Path:
    from pmbtc.features.docs import require_documentation, write_docs
    from pmbtc.features.families import all_features
    from pmbtc.features.graph import build_graph

    graph = build_graph(all_features())
    require_documentation(graph)
    return write_docs(graph, DOCS / "features.md")


def generate_schema_docs() -> Path:
    from pmbtc.dataset.schema import DATASET_RECORD_VERSION, FEATURE_SCHEMA_VERSION
    from pmbtc.features.provider import (
        ENGINEERED_FEATURE_NAMES,
        canonical_feature_schema_version,
    )
    from pmbtc.settlement.parser import PARSER_VERSION

    lines = [
        "# Schema versions",
        "",
        "Four versions move independently because they answer different",
        "questions. A change to any of them alters what stored data *means*.",
        "",
        "| schema | version | changes when |",
        "|---|---|---|",
        f"| Feature set | `{canonical_feature_schema_version()}` | "
        "a feature's formula, inputs, or units change (hashed automatically) |",
        f"| Dataset features | `{FEATURE_SCHEMA_VERSION}` | "
        "the observation/snapshot shape changes |",
        f"| Dataset records | `{DATASET_RECORD_VERSION}` | "
        "the market record shape changes |",
        f"| Settlement parser | `{PARSER_VERSION}` | "
        "settlement detection or normalisation changes |",
        "",
        f"The feature set currently contains **{len(ENGINEERED_FEATURE_NAMES)} features**.",
        "",
        "## Why the feature version is a hash",
        "",
        "It is computed from the declarations themselves — names, formulas,",
        "inputs, units, tiers. It therefore cannot be forgotten: editing a",
        "formula changes the version whether or not anyone remembers to bump it.",
        "The other three are hand-maintained and checked by",
        "`scripts/validate_schemas.py`.",
        "",
    ]
    path = DOCS / "SCHEMAS.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def generate_module_docs() -> Path:
    """Module map, taken from package docstrings."""
    import pmbtc

    lines = [
        "# Module reference",
        "",
        "Generated from package docstrings. Edit the code, not this file.",
        "",
    ]
    packages = [
        "pmbtc.settlement", "pmbtc.gamma", "pmbtc.dataset",
        "pmbtc.live", "pmbtc.features", "pmbtc.models", "pmbtc.ops",
    ]
    for name in packages:
        try:
            module = importlib.import_module(name)
        except ImportError as exc:
            lines += [f"## `{name}`", "", f"*unavailable: {exc}*", ""]
            continue
        doc = inspect.getdoc(module) or "*(no docstring)*"
        lines += [f"## `{name}`", "", doc, ""]

        submodules = sorted(
            m.name for m in pkgutil.iter_modules(module.__path__)
            if not m.name.startswith("_")
        )
        if submodules:
            lines.append("| submodule | summary |")
            lines.append("|---|---|")
            for sub in submodules:
                try:
                    child = importlib.import_module(f"{name}.{sub}")
                    summary = (inspect.getdoc(child) or "").split("\n")[0]
                except Exception:
                    summary = ""
                lines.append(f"| `{sub}` | {summary} |")
            lines.append("")
    del pmbtc
    path = DOCS / "MODULES.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def generate_api_docs() -> Path:
    """Public API surface: what each package exports."""
    lines = [
        "# API reference",
        "",
        "The public surface of each package, taken from `__all__`.",
        "",
    ]
    for name in ("pmbtc.settlement", "pmbtc.gamma", "pmbtc.dataset",
                 "pmbtc.live", "pmbtc.features", "pmbtc.models", "pmbtc.ops"):
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        exports = sorted(getattr(module, "__all__", []))
        if not exports:
            continue
        lines += [f"## `{name}`", ""]
        for export in exports:
            obj = getattr(module, export, None)
            doc = (inspect.getdoc(obj) or "").split("\n")[0] if obj else ""
            kind = "class" if inspect.isclass(obj) else (
                "function" if inspect.isfunction(obj) else "object"
            )
            lines.append(f"- **`{export}`** *({kind})* — {doc}")
        lines.append("")
    path = DOCS / "API.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def generate_index() -> Path:
    lines = [
        "# pmbtc documentation",
        "",
        "Autonomous trading system for Polymarket Bitcoin 5-minute Up/Down",
        "markets.",
        "",
        "## Start here",
        "",
        "- [Engineering workflow](GITHUB_WORKFLOW.md) — branches, versioning, gates",
        "- [Architecture](ARCHITECTURE.md) — how the pieces fit",
        "- [Operations guide](OPERATIONS.md) — running and monitoring the collector",
        "- [Deployment guide](DEPLOYMENT.md) — getting it running somewhere real",
        "",
        "## Reference",
        "",
        "- [Feature dictionary](features.md) — every feature, with its formula",
        "- [Schema versions](SCHEMAS.md)",
        "- [Module reference](MODULES.md)",
        "- [API reference](API.md)",
        "",
        "## Project documents",
        "",
        "Mirrored from the repository root at build time, so every link on this",
        "site resolves. Edit the source files, not these copies.",
        "",
        "- [Changelog](CHANGELOG.md)",
        "- [Contributing](CONTRIBUTING.md)",
        "- [Security policy](SECURITY.md)",
        "- [Branch protection](BRANCH_PROTECTION.md)",
        "",
    ]
    path = DOCS / "index.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    DOCS.mkdir(parents=True, exist_ok=True)
    written = [
        generate_feature_docs(),
        generate_schema_docs(),
        generate_module_docs(),
        generate_api_docs(),
        generate_index(),
        *mirror_root_documents(),
    ]
    for path in written:
        print(f"  wrote {path.relative_to(ROOT)} ({path.stat().st_size:,} bytes)")
    print(f"\n{len(written)} document(s) generated.")

    problems = check_internal_links()
    if problems:
        print(f"\n{len(problems)} unresolved documentation link(s):")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("All internal documentation links resolve inside docs/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
