"""Order execution.

PaperBroker models costs pessimistically on purpose. The most common way a
trading bot lies to its owner is by assuming it fills at the mid price with no
fee. At a S$100 account size that single assumption is the difference between
a backtest that shows +12% and a reality that shows -18%.

LiveBroker is deliberately hard to reach: it refuses to construct unless every
independent safety latch is open, and it imports ccxt lazily so paper mode
never even loads the dependency.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone

from .risk import Action, Decision, Portfolio, Position


@dataclass(frozen=True)
class Fill:
    ts: int
    symbol: str
    action: Action
    qty: float
    price: float            # the price actually paid, after slippage+spread
    ref_price: float        # the price that was quoted
    gross_sgd: float
    fee_sgd: float
    net_sgd: float          # cash actually leaving (BUY) or arriving (SELL)
    realised_pnl_sgd: float = 0.0

    @property
    def slippage_cost_sgd(self) -> float:
        return abs(self.price - self.ref_price) * self.qty


class BrokerError(Exception):
    pass


class PaperBroker:
    """Simulated execution with fees, slippage, and spread applied every time."""

    name = "paper"

    def __init__(self, cfg):
        self.cfg = cfg
        self.c = cfg.costs

    def _effective_price(self, ref: float, action: Action) -> float:
        """Buys fill above the quote, sells below it. Never in your favour."""
        drag = (self.c.slippage_pct + self.c.spread_pct) / 100.0
        return ref * (1 + drag) if action is Action.BUY else ref * (1 - drag)

    def execute(self, decision: Decision, pf: Portfolio, ref_price: float,
                ts: int) -> Fill | None:
        if not decision.tradeable:
            return None

        price = self._effective_price(ref_price, decision.action)

        if decision.action is Action.BUY:
            # decision.size_sgd is the total cash we are willing to part with,
            # so the fee must come out of it rather than be added on top.
            fee = decision.size_sgd * self.c.taker_fee_pct / 100.0
            spend = decision.size_sgd - fee
            if spend <= 0 or price <= 0:
                return None
            qty = spend / price

            if decision.size_sgd > pf.cash + 1e-9:
                raise BrokerError(
                    f"execution would overdraw cash: need S${decision.size_sgd:.2f}, "
                    f"have S${pf.cash:.2f}"
                )

            pf.cash -= decision.size_sgd
            existing = pf.positions.get(decision.symbol)
            if existing:
                total_qty = existing.qty + qty
                pf.positions[decision.symbol] = Position(
                    symbol=decision.symbol,
                    qty=total_qty,
                    # weighted average entry, so P&L on a scaled-in position
                    # is measured against what was actually paid
                    entry_price=(existing.entry_price * existing.qty + price * qty) / total_qty,
                    entry_ts=existing.entry_ts,
                    cost_basis_sgd=existing.cost_basis_sgd + decision.size_sgd,
                )
            else:
                pf.positions[decision.symbol] = Position(
                    decision.symbol, qty, price, ts, decision.size_sgd)

            pf.trades_today += 1
            pf.fees_today += fee
            pf.fees_total += fee
            return Fill(ts, decision.symbol, Action.BUY, qty, price, ref_price,
                        spend, fee, decision.size_sgd)

        # ---- SELL ----
        pos = pf.positions.get(decision.symbol)
        if pos is None:
            return None

        qty = min(decision.size_sgd / price, pos.qty) if price > 0 else 0.0
        if qty <= 0:
            return None
        gross = qty * price
        fee = gross * self.c.taker_fee_pct / 100.0
        proceeds = gross - fee

        frac = qty / pos.qty if pos.qty else 1.0
        basis = pos.cost_basis_sgd * frac
        realised = proceeds - basis

        pf.cash += proceeds
        remaining = pos.qty - qty
        if remaining <= 1e-12:
            del pf.positions[decision.symbol]
        else:
            pf.positions[decision.symbol] = Position(
                pos.symbol, remaining, pos.entry_price, pos.entry_ts,
                pos.cost_basis_sgd - basis)

        pf.trades_today += 1
        pf.fees_today += fee
        pf.fees_total += fee
        return Fill(ts, decision.symbol, Action.SELL, qty, price, ref_price,
                    gross, fee, proceeds, realised)


class LiveBroker:
    """Real orders against a real exchange. Guarded three ways.

    This class will not construct unless config, an environment confirmation
    phrase, and API credentials all agree. That is intentional friction.
    """

    name = "live"

    def __init__(self, cfg):
        if not cfg.live_armed:
            raise BrokerError(
                "LIVE MODE NOT ARMED. All three are required:\n"
                "  1. live.enabled: true in config.yaml\n"
                "  2. LIVE_TRADING=I_UNDERSTAND_THE_RISK in your environment\n"
                "  3. EXCHANGE_API_KEY and EXCHANGE_API_SECRET set\n"
                "Refusing to trade real money."
            )
        try:
            import ccxt                                    # noqa: PLC0415
        except ImportError as e:
            raise BrokerError(
                "live mode needs ccxt: pip install ccxt"
            ) from e

        if not cfg.live_exchange:
            raise BrokerError("live.exchange is empty in config.yaml")
        if not hasattr(ccxt, cfg.live_exchange):
            raise BrokerError(f"ccxt has no exchange {cfg.live_exchange!r}")

        self.cfg = cfg
        self.client = getattr(ccxt, cfg.live_exchange)({
            "apiKey": os.environ["EXCHANGE_API_KEY"],
            "secret": os.environ["EXCHANGE_API_SECRET"],
            "enableRateLimit": True,
        })

    def execute(self, decision: Decision, pf: Portfolio, ref_price: float,
                ts: int) -> Fill | None:
        if not decision.tradeable:
            return None
        if decision.size_sgd > self.cfg.risk.max_notional_sgd:
            raise BrokerError(
                f"refusing live order of S${decision.size_sgd:.2f}: "
                f"above max_notional S${self.cfg.risk.max_notional_sgd:.2f}"
            )

        side = "buy" if decision.action is Action.BUY else "sell"
        qty = decision.size_sgd / ref_price
        order = self.client.create_order(decision.symbol, "market", side, qty)

        filled_price = float(order.get("average") or order.get("price") or ref_price)
        filled_qty = float(order.get("filled") or qty)
        fee_info = order.get("fee") or {}
        fee = float(fee_info.get("cost") or
                    decision.size_sgd * self.cfg.costs.taker_fee_pct / 100.0)
        gross = filled_qty * filled_price

        pf.trades_today += 1
        pf.fees_today += fee
        pf.fees_total += fee
        return Fill(ts, decision.symbol, decision.action, filled_qty,
                    filled_price, ref_price, gross, fee,
                    gross + fee if side == "buy" else gross - fee)


def make_broker(cfg):
    """Return a live broker only when every latch is open; paper otherwise."""
    if cfg.live_armed:
        return LiveBroker(cfg)
    return PaperBroker(cfg)
