"""Shared fixtures.

Every test that touches config goes through :func:`load_config` with an explicit
path so the suite is independent of the working directory, and clears the
process-wide caches so no test can leak configuration into the next.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from pmbtc.config import Config, load_config, reset_config
from pmbtc.logging_setup import reset_logging

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIPPED_CONFIG = REPO_ROOT / "config" / "config.yaml"


@pytest.fixture(autouse=True)
def _clean_state() -> Iterator[None]:
    """Isolate global state and any PMBTC_* env vars leaking in from the shell."""
    saved = {k: v for k, v in os.environ.items() if k.startswith("PMBTC_")}
    for key in saved:
        del os.environ[key]
    reset_config()
    reset_logging()
    yield
    for key in [k for k in os.environ if k.startswith("PMBTC_")]:
        del os.environ[key]
    os.environ.update(saved)
    reset_config()
    reset_logging()


@pytest.fixture
def shipped_config() -> Config:
    """The configuration that actually ships in ``config/config.yaml``.

    Loading it in the suite means a broken default file fails CI rather than
    a deployment.
    """
    return load_config(SHIPPED_CONFIG)


@pytest.fixture
def tmp_config(tmp_path: Path) -> Config:
    """A config rooted in a temp dir, safe to create directories under."""
    return load_config(SHIPPED_CONFIG, app={"base_dir": str(tmp_path), "mode": "paper"})
