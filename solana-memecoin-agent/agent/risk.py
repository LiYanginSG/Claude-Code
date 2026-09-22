"""The risk engine: hard rails between the model and the money.

Design rule, and the whole point of this project:

    The LLM proposes. This module disposes.

Nothing here reads the model's reasoning, its confidence, or its enthusiasm.
These are pure functions over numbers. A model cannot talk its way past them,
because it is never asked. Every veto and every clamp is recorded, which turns
the rails into the most interesting part of the log.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum


class Action(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class Verdict(str, Enum):
    APPROVED = "APPROVED"   # passed untouched
    CLAMPED = "CLAMPED"     # allowed, but smaller than requested
    VETOED = "VETOED"       # blocked entirely
    FORCED = "FORCED"       # the engine itself is ordering this (stop/kill)


@dataclass
class Position:
    symbol: str
    qty: float
    entry_price: float
    entry_ts: int
    cost_basis_sgd: float   # what was actually paid, fees included

    def value(self, price: float) -> float:
        return self.qty * price

    def pnl_pct(self, price: float) -> float:
        if self.entry_price == 0:
            return 0.0
        return (price / self.entry_price - 1.0) * 100.0


@dataclass
class Portfolio:
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    peak_equity: float = 0.0
    trades_today: int = 0
    fees_today: float = 0.0
    fees_total: float = 0.0
    day_stamp: str = ""
    halted: bool = False
    halt_reason: str = ""
    cooldown_until: dict[str, int] = field(default_factory=dict)  # symbol -> ms

    def equity(self, prices: dict[str, float]) -> float:
        held = sum(p.value(prices.get(s, p.entry_price))
                   for s, p in self.positions.items())
        return self.cash + held

    def exposure_pct(self, prices: dict[str, float]) -> float:
        eq = self.equity(prices)
        if eq <= 0:
            return 0.0
        held = sum(p.value(prices.get(s, p.entry_price))
                   for s, p in self.positions.items())
        return held / eq * 100.0

    def roll_day(self, ts_ms: int) -> None:
        """Reset per-day counters when the UTC date changes."""
        stamp = datetime.fromtimestamp(ts_ms / 1000, timezone.utc).strftime("%Y-%m-%d")
        if stamp != self.day_stamp:
            self.day_stamp = stamp
            self.trades_today = 0
            self.fees_today = 0.0


@dataclass
class Intent:
    """What the brain wants to do."""
    action: Action
    symbol: str
    size_pct: float          # % of equity to deploy (BUY) or of position (SELL)
    confidence: float        # 0..1, for logging only -- never sizes a trade
    reasoning: str = ""
    invalidation: str = ""   # what would prove this thesis wrong


@dataclass
class Decision:
    """What the risk engine permits."""
    verdict: Verdict
    action: Action
    symbol: str
    size_sgd: float
    rails: list[str] = field(default_factory=list)   # every rule that fired
    intent: Intent | None = None

    @property
    def tradeable(self) -> bool:
        return self.action in (Action.BUY, Action.SELL) and self.size_sgd > 0


# ---------------------------------------------------------------------------
# Forced exits -- evaluated BEFORE the brain is consulted
# ---------------------------------------------------------------------------

def check_forced_exits(pf: Portfolio, prices: dict[str, float],
                       cfg, ts_ms: int) -> list[Decision]:
    """Stop-losses, take-profits, and the drawdown kill switch.

    These are not suggestions to the model. They are executed first, and the
    model is told afterwards. A stop loss you can argue with is not a stop loss.
    """
    out: list[Decision] = []
    equity = pf.equity(prices)
    pf.peak_equity = max(pf.peak_equity, equity)

    # --- kill switch: drawdown from peak equity ---------------------------
    if pf.peak_equity > 0:
        dd = (1.0 - equity / pf.peak_equity) * 100.0
        if dd >= cfg.risk.max_drawdown_pct and not pf.halted:
            pf.halted = True
            pf.halt_reason = (
                f"KILL SWITCH: drawdown {dd:.1f}% >= "
                f"{cfg.risk.max_drawdown_pct:.1f}% of peak equity "
                f"S${pf.peak_equity:.2f}"
            )
            for sym, pos in list(pf.positions.items()):
                out.append(Decision(
                    verdict=Verdict.FORCED,
                    action=Action.SELL,
                    symbol=sym,
                    size_sgd=pos.value(prices.get(sym, pos.entry_price)),
                    rails=[pf.halt_reason, "liquidate_all"],
                ))
            return out   # nothing else matters once we are halted

    # --- per-position stop loss / take profit -----------------------------
    for sym, pos in list(pf.positions.items()):
        price = prices.get(sym)
        if price is None:
            continue
        pnl = pos.pnl_pct(price)
        hit = None
        if pnl <= -cfg.risk.stop_loss_pct:
            hit = f"stop_loss: {pnl:.2f}% <= -{cfg.risk.stop_loss_pct:.1f}%"
        elif pnl >= cfg.risk.take_profit_pct:
            hit = f"take_profit: {pnl:+.2f}% >= +{cfg.risk.take_profit_pct:.1f}%"
        if hit:
            out.append(Decision(
                verdict=Verdict.FORCED,
                action=Action.SELL,
                symbol=sym,
                size_sgd=pos.value(price),
                rails=[hit],
            ))
            # A stopped-out symbol goes into cooldown so the agent cannot
            # immediately re-enter the same losing idea.
            if pnl < 0:
                pf.cooldown_until[sym] = ts_ms + cfg.risk.cooldown_minutes * 60_000
    return out


# ---------------------------------------------------------------------------
# Intent evaluation
# ---------------------------------------------------------------------------

def evaluate(intent: Intent, pf: Portfolio, prices: dict[str, float],
             cfg, ts_ms: int) -> Decision:
    """Apply every rail to a proposed trade. Returns what is actually allowed."""
    rails: list[str] = []
    equity = pf.equity(prices)

    def veto(reason: str) -> Decision:
        return Decision(Verdict.VETOED, Action.HOLD, intent.symbol, 0.0,
                        rails + [reason], intent)

    if intent.action == Action.HOLD:
        return Decision(Verdict.APPROVED, Action.HOLD, intent.symbol, 0.0,
                        ["hold: no action"], intent)

    # --- gates that apply to any trade ------------------------------------
    if pf.halted:
        return veto(f"halted: {pf.halt_reason}")

    if intent.symbol not in cfg.allowlist:
        return veto(f"allowlist: {intent.symbol} is not tradeable")

    price = prices.get(intent.symbol)
    if price is None or price <= 0:
        return veto(f"no price available for {intent.symbol}")

    # ================= SELL =================
    if intent.action == Action.SELL:
        pos = pf.positions.get(intent.symbol)
        if pos is None:
            return veto(f"no open position in {intent.symbol} to sell")
        frac = max(0.0, min(intent.size_pct, 100.0)) / 100.0
        size = pos.value(price) * frac
        if frac < 1.0 and size < cfg.risk.min_trade_sgd:
            # A partial sell too small to matter: close the lot instead of
            # paying a fee for a rounding error.
            rails.append(
                f"partial sell S${size:.2f} < min S${cfg.risk.min_trade_sgd:.2f}: "
                f"closing full position"
            )
            size = pos.value(price)
        return Decision(
            Verdict.CLAMPED if rails else Verdict.APPROVED,
            Action.SELL, intent.symbol, size, rails or ["sell approved"], intent)

    # ================= BUY =================
    # Fee-budget rails. At S$100 these bind long before anything else does.
    if pf.trades_today >= cfg.risk.max_trades_per_day:
        return veto(
            f"daily trade cap: {pf.trades_today}/{cfg.risk.max_trades_per_day} used"
        )

    fee_budget = equity * cfg.risk.daily_fee_budget_pct / 100.0
    if pf.fees_today >= fee_budget:
        return veto(
            f"daily fee budget spent: S${pf.fees_today:.2f} >= S${fee_budget:.2f}"
        )

    cd = pf.cooldown_until.get(intent.symbol, 0)
    if ts_ms < cd:
        mins = int((cd - ts_ms) / 60_000)
        return veto(f"cooldown: {intent.symbol} locked for another {mins} min")

    # --- sizing: start from what was asked, then clamp down repeatedly ----
    requested_pct = max(0.0, min(intent.size_pct, 100.0))
    size = equity * requested_pct / 100.0
    original = size

    cap_position = equity * cfg.risk.max_position_pct / 100.0
    existing = pf.positions[intent.symbol].value(price) if intent.symbol in pf.positions else 0.0
    room_position = max(0.0, cap_position - existing)
    if size > room_position:
        size = room_position
        rails.append(
            f"max_position {cfg.risk.max_position_pct:.0f}% of equity: "
            f"capped to S${size:.2f}"
        )

    cap_total = equity * cfg.risk.max_total_exposure_pct / 100.0
    deployed = sum(p.value(prices.get(s, p.entry_price)) for s, p in pf.positions.items())
    room_total = max(0.0, cap_total - deployed)
    if size > room_total:
        size = room_total
        rails.append(
            f"max_total_exposure {cfg.risk.max_total_exposure_pct:.0f}%: "
            f"capped to S${size:.2f}"
        )

    room_notional = max(0.0, cfg.risk.max_notional_sgd - deployed)
    if size > room_notional:
        size = room_notional
        rails.append(f"max_notional S${cfg.risk.max_notional_sgd:.2f}: capped to S${size:.2f}")

    if size > pf.cash:
        size = pf.cash
        rails.append(f"insufficient cash: capped to S${size:.2f}")

    # --- the floor: below this, fees dominate and the trade is pointless --
    if size < cfg.risk.min_trade_sgd:
        return veto(
            f"below min_trade S${cfg.risk.min_trade_sgd:.2f} "
            f"(sized S${size:.2f}) -- fees would dominate"
        )

    verdict = Verdict.CLAMPED if size < original - 1e-9 else Verdict.APPROVED
    if verdict == Verdict.APPROVED:
        rails.append("buy approved at requested size")
    return Decision(verdict, Action.BUY, intent.symbol, size, rails, intent)
