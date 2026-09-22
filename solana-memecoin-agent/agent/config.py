"""Configuration loading and validation.

Fails loudly at startup rather than quietly at 3am with real money on the line.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(Exception):
    """Raised when the config is malformed or internally inconsistent."""


@dataclass(frozen=True)
class Costs:
    taker_fee_pct: float
    slippage_pct: float
    spread_pct: float

    @property
    def round_trip_pct(self) -> float:
        """Total cost of getting in and back out again, as a percent.

        This is the number that decides whether a S$100 account has any
        business trading at all. At the defaults it is 1.5% -- meaning a
        position must move 1.5% in your favour just to break even.
        """
        return 2 * (self.taker_fee_pct + self.slippage_pct + self.spread_pct)


@dataclass(frozen=True)
class Risk:
    max_position_pct: float
    max_total_exposure_pct: float
    min_trade_sgd: float
    max_trades_per_day: int
    daily_fee_budget_pct: float
    stop_loss_pct: float
    take_profit_pct: float
    max_drawdown_pct: float
    cooldown_minutes: int
    max_notional_sgd: float


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    base_currency: str
    starting_equity: float
    allowlist: tuple[str, ...]
    timeframe: str
    lookback_candles: int
    source: str
    replay_file: str
    live_endpoint: str
    costs: Costs
    risk: Risk
    brain_engine: str
    model: str
    temperature: float
    max_tokens: int
    fallback_to_deterministic: bool
    live_enabled: bool
    live_exchange: str
    interval_minutes: int
    max_iterations: int
    log_dir: str

    # ---- derived -----------------------------------------------------------
    @property
    def live_armed(self) -> bool:
        """True only when every independent safety latch is open.

        Three separate things must agree: the config file, an environment
        variable with an exact confirmation phrase, and the presence of API
        credentials. Forgetting any one of them keeps you in paper mode.
        """
        return (
            self.live_enabled
            and os.getenv("LIVE_TRADING", "") == "I_UNDERSTAND_THE_RISK"
            and bool(os.getenv("EXCHANGE_API_KEY"))
            and bool(os.getenv("EXCHANGE_API_SECRET"))
        )

    @property
    def breakeven_move_pct(self) -> float:
        """How far price must move before a round trip is profitable."""
        return self.costs.round_trip_pct


def _req(d: dict, *path: str) -> Any:
    node: Any = d
    for key in path:
        if not isinstance(node, dict) or key not in node:
            raise ConfigError(f"missing required config key: {'.'.join(path)}")
        node = node[key]
    return node


def load(path: str | Path = "config.yaml") -> Config:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config not found: {p}")

    raw = yaml.safe_load(p.read_text()) or {}

    costs = Costs(
        taker_fee_pct=float(_req(raw, "costs", "taker_fee_pct")),
        slippage_pct=float(_req(raw, "costs", "slippage_pct")),
        spread_pct=float(_req(raw, "costs", "spread_pct")),
    )
    r = _req(raw, "risk")
    risk = Risk(
        max_position_pct=float(r["max_position_pct"]),
        max_total_exposure_pct=float(r["max_total_exposure_pct"]),
        min_trade_sgd=float(r["min_trade_sgd"]),
        max_trades_per_day=int(r["max_trades_per_day"]),
        daily_fee_budget_pct=float(r["daily_fee_budget_pct"]),
        stop_loss_pct=float(r["stop_loss_pct"]),
        take_profit_pct=float(r["take_profit_pct"]),
        max_drawdown_pct=float(r["max_drawdown_pct"]),
        cooldown_minutes=int(r["cooldown_minutes"]),
        max_notional_sgd=float(r["max_notional_sgd"]),
    )

    cfg = Config(
        raw=raw,
        base_currency=str(_req(raw, "account", "base_currency")),
        starting_equity=float(_req(raw, "account", "starting_equity")),
        allowlist=tuple(_req(raw, "market", "allowlist")),
        timeframe=str(raw["market"].get("timeframe", "1h")),
        lookback_candles=int(raw["market"].get("lookback_candles", 200)),
        source=str(raw["market"].get("source", "synthetic")),
        replay_file=str(raw["market"].get("replay_file", "")),
        live_endpoint=str(raw["market"].get("live_endpoint", "binance")),
        costs=costs,
        risk=risk,
        brain_engine=str(raw.get("brain", {}).get("engine", "deterministic")),
        model=str(raw.get("brain", {}).get("model", "claude-opus-5")),
        temperature=float(raw.get("brain", {}).get("temperature", 0.2)),
        max_tokens=int(raw.get("brain", {}).get("max_tokens", 1200)),
        fallback_to_deterministic=bool(
            raw.get("brain", {}).get("fallback_to_deterministic", True)
        ),
        live_enabled=bool(raw.get("live", {}).get("enabled", False)),
        live_exchange=str(raw.get("live", {}).get("exchange", "")),
        interval_minutes=int(raw.get("run", {}).get("interval_minutes", 60)),
        max_iterations=int(raw.get("run", {}).get("max_iterations", 0)),
        log_dir=str(raw.get("run", {}).get("log_dir", "runs")),
    )
    _validate(cfg)
    return cfg


def _validate(c: Config) -> None:
    """Catch configs that are syntactically fine but financially nonsense."""
    problems: list[str] = []

    if c.starting_equity <= 0:
        problems.append("account.starting_equity must be > 0")
    if c.risk.min_trade_sgd > c.starting_equity:
        problems.append(
            f"risk.min_trade_sgd ({c.risk.min_trade_sgd}) exceeds starting "
            f"equity ({c.starting_equity}) -- no trade could ever be placed"
        )
    if c.risk.max_position_pct > c.risk.max_total_exposure_pct:
        problems.append(
            "risk.max_position_pct cannot exceed risk.max_total_exposure_pct"
        )
    if not 0 < c.risk.max_drawdown_pct <= 100:
        problems.append("risk.max_drawdown_pct must be in (0, 100]")
    if c.risk.stop_loss_pct <= c.costs.round_trip_pct:
        problems.append(
            f"risk.stop_loss_pct ({c.risk.stop_loss_pct}%) is inside the "
            f"round-trip cost ({c.costs.round_trip_pct}%) -- every position "
            f"would stop out on fees alone"
        )
    if not c.allowlist:
        problems.append("market.allowlist is empty -- nothing to trade")
    if c.source not in {"live", "replay", "synthetic"}:
        problems.append(f"market.source must be live|replay|synthetic, got {c.source!r}")
    if c.brain_engine not in {"claude", "deterministic"}:
        problems.append(f"brain.engine must be claude|deterministic, got {c.brain_engine!r}")

    # The one that catches the most self-deception:
    min_pos = c.starting_equity * c.risk.max_position_pct / 100
    if min_pos < c.risk.min_trade_sgd:
        problems.append(
            f"max_position_pct ({c.risk.max_position_pct}%) of equity is "
            f"S${min_pos:.2f}, below min_trade_sgd (S${c.risk.min_trade_sgd:.2f}). "
            f"The agent is configured to never trade."
        )

    if problems:
        raise ConfigError("invalid configuration:\n  - " + "\n  - ".join(problems))
