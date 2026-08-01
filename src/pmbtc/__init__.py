"""pmbtc — autonomous trading bot for Polymarket Bitcoin 5-minute Up/Down markets.

Module map (built in this order; each is approved before the next begins):

1.  ``config`` / ``logging_setup`` / ``constants`` / ``utils``  — architecture
2.  ``settlement``   — settlement verification engine (the immutable core)
3.  ``gamma``        — Gamma API market parser
4.  ``collectors.historical`` — historical data collector
5.  ``collectors.live``       — live market data collector
6.  ``features``     — feature engineering pipeline (tiered + evidence-selected)
7.  ``models``       — model training, calibration, ensembling
8.  ``backtest``     — backtesting framework
9.  ``paper``        — paper trading engine
10. ``execution``    — live execution engine
11. ``monitoring``   — dashboard and alerting
12. ``retrain``      — continuous retraining and champion/challenger promotion
"""

from __future__ import annotations

__version__ = "0.8.3"

__all__ = ["__version__"]
