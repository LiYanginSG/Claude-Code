"""Tests for the risk engine.

Every test here describes money that does NOT get lost.

NOTE: these cover the venue-agnostic rails (sizing caps, fee budget, trade
caps, cooldowns, drawdown kill switch). The memecoin-specific gates --
freeze authority, LP burn, honeypot simulation -- live in safety.py and are
tested separately, because a stop-loss cannot save you from a token you are
not permitted to sell.
"""
import pytest

from agent.risk import (Action, Intent, Portfolio, Position, Verdict,
                        check_forced_exits, evaluate)

TS = 1_700_000_000_000
PRICES = {"BTC/USDT": 50_000.0, "ETH/USDT": 3_000.0}


def pf(cash=100.0, **kw):
    p = Portfolio(cash=cash, **kw)
    p.peak_equity = cash
    return p


def buy(symbol="BTC/USDT", size_pct=25.0, confidence=0.9):
    return Intent(Action.BUY, symbol, size_pct, confidence, "test")


# --- sizing rails ----------------------------------------------------------

def test_buy_at_allowed_size_is_approved(cfg):
    d = evaluate(buy(size_pct=25.0), pf(), PRICES, cfg, TS)
    assert d.verdict is Verdict.APPROVED
    assert d.size_sgd == pytest.approx(25.0)


def test_oversized_buy_is_clamped_not_rejected(cfg):
    d = evaluate(buy(size_pct=100.0), pf(), PRICES, cfg, TS)
    assert d.verdict is Verdict.CLAMPED
    assert d.size_sgd == pytest.approx(25.0)
    assert any("max_position" in r for r in d.rails)


def test_confidence_never_increases_size(cfg):
    """A model claiming 100% confidence gets the same size as one at 10%."""
    lo = evaluate(buy(size_pct=100.0, confidence=0.1), pf(), PRICES, cfg, TS)
    hi = evaluate(buy(size_pct=100.0, confidence=1.0), pf(), PRICES, cfg, TS)
    assert lo.size_sgd == hi.size_sgd


def test_total_exposure_cap_across_symbols(cfg):
    p = pf(cash=50.0)
    p.positions["BTC/USDT"] = Position("BTC/USDT", 50.0 / 50_000, 50_000, TS, 50.0)
    assert evaluate(buy("ETH/USDT", 100.0), p, PRICES, cfg, TS).verdict is Verdict.VETOED


def test_cannot_exceed_cash(cfg):
    assert evaluate(buy(size_pct=100.0), pf(cash=10.0), PRICES, cfg, TS).verdict is Verdict.VETOED


def test_dust_trade_is_vetoed(cfg):
    d = evaluate(buy(size_pct=5.0), pf(), PRICES, cfg, TS)
    assert d.verdict is Verdict.VETOED
    assert any("min_trade" in r for r in d.rails)


# --- gating rails ----------------------------------------------------------

def test_symbol_not_on_allowlist_is_vetoed(cfg):
    d = evaluate(buy("DOGE/USDT", 25.0), pf(), PRICES, cfg, TS)
    assert d.verdict is Verdict.VETOED
    assert any("allowlist" in r for r in d.rails)


def test_daily_trade_cap(cfg):
    p = pf(); p.trades_today = cfg.risk.max_trades_per_day
    assert evaluate(buy(), p, PRICES, cfg, TS).verdict is Verdict.VETOED


def test_daily_fee_budget(cfg):
    p = pf(); p.fees_today = 99.0
    d = evaluate(buy(), p, PRICES, cfg, TS)
    assert d.verdict is Verdict.VETOED
    assert any("fee budget" in r for r in d.rails)


def test_cooldown_blocks_revenge_trade(cfg):
    p = pf(); p.cooldown_until["BTC/USDT"] = TS + 3_600_000
    d = evaluate(buy(), p, PRICES, cfg, TS)
    assert d.verdict is Verdict.VETOED
    assert any("cooldown" in r for r in d.rails)


def test_halted_portfolio_rejects_everything(cfg):
    p = pf(); p.halted = True; p.halt_reason = "kill switch"
    assert evaluate(buy(), p, PRICES, cfg, TS).verdict is Verdict.VETOED


def test_sell_without_position_is_vetoed(cfg):
    d = evaluate(Intent(Action.SELL, "BTC/USDT", 100.0, 0.9), pf(), PRICES, cfg, TS)
    assert d.verdict is Verdict.VETOED


# --- forced exits ----------------------------------------------------------

def test_stop_loss_fires_without_asking_the_model(cfg):
    p = pf(cash=75.0)
    p.positions["BTC/USDT"] = Position("BTC/USDT", 25.0 / 50_000, 50_000, TS, 25.0)
    exits = check_forced_exits(p, {"BTC/USDT": 45_000.0}, cfg, TS)
    assert len(exits) == 1
    assert exits[0].verdict is Verdict.FORCED and exits[0].action is Action.SELL
    assert any("stop_loss" in r for r in exits[0].rails)


def test_stop_out_sets_cooldown(cfg):
    p = pf(cash=75.0)
    p.positions["BTC/USDT"] = Position("BTC/USDT", 25.0 / 50_000, 50_000, TS, 25.0)
    check_forced_exits(p, {"BTC/USDT": 45_000.0}, cfg, TS)
    assert p.cooldown_until["BTC/USDT"] > TS


def test_take_profit_fires(cfg):
    p = pf(cash=75.0)
    p.positions["BTC/USDT"] = Position("BTC/USDT", 25.0 / 50_000, 50_000, TS, 25.0)
    exits = check_forced_exits(p, {"BTC/USDT": 60_000.0}, cfg, TS)
    assert any("take_profit" in r for r in exits[0].rails)


def test_kill_switch_liquidates_and_halts(cfg):
    p = Portfolio(cash=10.0); p.peak_equity = 100.0
    p.positions["BTC/USDT"] = Position("BTC/USDT", 50.0 / 50_000, 50_000, TS, 50.0)
    exits = check_forced_exits(p, {"BTC/USDT": 25_000.0}, cfg, TS)
    assert p.halted and "KILL SWITCH" in p.halt_reason
    assert all(d.action is Action.SELL and d.verdict is Verdict.FORCED for d in exits)


def test_no_kill_switch_within_tolerance(cfg):
    p = Portfolio(cash=90.0); p.peak_equity = 100.0
    assert check_forced_exits(p, PRICES, cfg, TS) == []
    assert not p.halted


# --- accounting ------------------------------------------------------------

def test_day_roll_resets_counters(cfg):
    p = pf(); p.trades_today, p.fees_today = 2, 5.0
    p.roll_day(TS); p.roll_day(TS + 86_400_000 * 2)
    assert p.trades_today == 0 and p.fees_today == 0.0


def test_peak_equity_only_ratchets_up(cfg):
    p = pf(cash=100.0)
    check_forced_exits(p, PRICES, cfg, TS)
    p.cash = 80.0
    check_forced_exits(p, PRICES, cfg, TS)
    assert p.peak_equity == 100.0
