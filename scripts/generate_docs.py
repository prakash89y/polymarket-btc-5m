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
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


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
        f"| Dataset features | `{FEATURE_SCHEMA_VERSION}` | the observation/snapshot shape changes |",
        f"| Dataset records | `{DATASET_RECORD_VERSION}` | the market record shape changes |",
        f"| Settlement parser | `{PARSER_VERSION}` | settlement detection or normalisation changes |",
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
        "- [Changelog](../CHANGELOG.md)",
        "- [Contributing](../CONTRIBUTING.md)",
        "- [Security policy](../SECURITY.md)",
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
    ]
    for path in written:
        print(f"  wrote {path.relative_to(ROOT)} ({path.stat().st_size:,} bytes)")
    print(f"\n{len(written)} document(s) generated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
