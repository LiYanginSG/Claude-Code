# solana-memecoin-agent

**Status: work in progress.** Venue-agnostic risk engine + paper broker are
built and tested. The Solana/Jupiter execution layer is designed but not yet
written. See "What's missing" below.

## What this is

An autonomous trading agent scoped to a small, deliberately-capped wallet.
Its organising principle:

> **The LLM proposes. The risk engine disposes.**

The model never sizes a trade. It emits an intent; `agent/risk.py` applies
hard numeric rails and returns what is actually permitted. Every veto and
clamp is logged, which makes the rails the most informative part of the run.

## Why the rails are layered

| Layer | Enforced by | Bypassable by a bug in this repo? |
|---|---|---|
| Wallet funding | You. Only fund what you'll lose. | No |
| CDP Policy Engine | Coinbase infrastructure, server-side | **No** |
| `agent/risk.py` | This codebase | Yes |

The middle layer matters most for an autonomous agent. Token metadata
(names, descriptions, socials) is attacker-controlled text that flows into a
model holding signing authority. A server-side policy restricting the wallet
to Jupiter program IDs means even a fully hijacked agent can swap but cannot
transfer funds out.

## Built and tested

- `agent/config.py` — config loading with financial sanity checks. Refuses
  configs that are syntactically valid but economically impossible (e.g. a
  stop-loss tighter than round-trip costs).
- `agent/risk.py` — position caps, total-exposure caps, daily trade and fee
  budgets, per-position stop/take-profit, post-loss cooldowns, and a
  drawdown kill switch that liquidates and halts. 19 tests.
- `agent/broker.py` — `PaperBroker` with pessimistic cost modelling (fee +
  slippage + spread on every fill, never in your favour).

## What's missing

- `agent/solrpc.py` — SPL mint account parsing (mint/freeze authority)
- `agent/safety.py` — the gates that actually matter for memecoins:
  freeze authority revoked, mint authority revoked, LP burned, holder
  concentration, round-trip quote simulation (honeypot detection)
- `agent/jupiter.py` — quote/swap
- `agent/wallet.py` — CDP Solana account + policy provisioning
- `agent/brain.py`, `agent/ledger.py`, `agent/report.py`, `run.py`

## Important caveat on the current risk engine

The rails in `risk.py` are built for **market risk** and assume you can
always sell near the quoted price. Memecoins break that assumption: a
non-revoked freeze authority means your token account can be frozen and the
stop-loss becomes a no-op. Those gates must run *before* the buy, not after.
That is what `safety.py` is for, and until it exists this agent should not
be pointed at memecoins.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # then fill it in
pytest -q
```

Paper mode is the default and requires no keys. Live mode requires three
independent latches to be open; see `.env.example`.
