# TradeBot

[![CI](https://github.com/ronenme44-jpg/TradeBot/actions/workflows/ci.yml/badge.svg)](https://github.com/ronenme44-jpg/TradeBot/actions/workflows/ci.yml)

TradeBot is a Python trading-bot showcase built around intraday candle ingestion,
graphical pattern detection, indicator confirmation, risk management, persistence,
and market-session scheduling.

This public version keeps the original project structure and file names, but limits
the strategy library to ten classical graphical patterns so the code is easy to
review, run, and discuss.

The goal of this version is to demonstrate practical Python engineering around market
data ingestion, pattern detection, signal confirmation, risk management, persistence,
and market-session scheduling.

## Quick Review

If you are reviewing the project, start with these files:

1. `tradeBot_main.py` - the main runtime loop and trade-management flow.
2. `tradeBot_Pattern_Test.py` - the training/test loop with session-aware scheduling.
3. `tradeBot_Graphical_Pattern.py` - the reduced graphical pattern detector set.
4. `tradeBot_indicators_pattern.py` - the indicator confirmation layer.
5. `tradeBot_Indicator_Rules.py` - the reinforcement-key schema for each pattern.
6. `tests/test_public_contract.py` - contract tests that protect the public pattern set and market-session behavior.

## What It Does

- Collects intraday OHLCV candles for a US equity with `yfinance`.
- Detects a curated set of long and short graphical price patterns.
- Confirms detected patterns with indicator-specific rule checks.
- Tracks open trades and updates ladder-based TP/SL levels.
- Maintains reinforcement-style pattern scores from closed trade outcomes.
- Runs only during the regular US market session and handles delayed free data.
- Keeps generated runtime state out of Git.

## Included Pattern Set

Long patterns:

- `bullish_engulfing`
- `hammer`
- `morning_star`
- `piercing_pattern`
- `three_white_soldiers_simple`

Short patterns:

- `bearish_engulfing`
- `shooting_star`
- `evening_star`
- `three_black_crows`
- `dark_cloud_cover`

This public showcase intentionally keeps the strategy surface compact and readable.

## Architecture

```text
generate_shared_candles_stocks.py
        |
        v
shared_candles.csv
        |
        v
tradeBot_main.py / tradeBot_Pattern_Test.py
        |
        +--> tradeBot_Graphical_Pattern.py
        +--> tradeBot_indicators_pattern.py
        +--> tradeBot_Indicator_Rules.py
        +--> tradeBot_Thresholds.py
        +--> reinforcement_manager.py
```

## Main Files

- `tradeBot_main.py` - main bot loop using the reduced ten-pattern set.
- `tradeBot_Pattern_Test.py` - training/test bot using the same ten-pattern set.
- `tradeBot_Graphical_Pattern.py` - graphical detection functions for selected patterns.
- `tradeBot_indicators_pattern.py` - confirmation functions for selected patterns.
- `tradeBot_Indicator_Rules.py` - expected indicator schema used for reinforcement keys.
- `tradeBot_Thresholds.py` - ladder thresholds for TP/SL management.
- `tradeBot_pattern_registry.py` - timeframe registry helper retained for compatibility.
- `reinforcement_manager.py` - persistence helpers for pattern scoring.
- `terminal_display.py` - compact terminal status rendering.
- `generate_shared_candles_stocks.py` - `yfinance` candle collector.
- `ttp_adapter.py` - optional adapter for JSONL trade-tape style data.
- `config.py` - fee configuration only. No API keys are required.
- `tests/test_public_contract.py` - lightweight tests for imports, pattern alignment, reinforcement keys, and session windows.
- `.github/workflows/ci.yml` - GitHub Actions workflow that compiles the code and runs the tests.

## Setup

Requirements:

- Python 3.10+
- Internet access for `yfinance` candle downloads

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

First fetch candles:

```bash
python generate_shared_candles_stocks.py
```

Then run the main bot:

```bash
python tradeBot_main.py
```

Or run the training/test bot:

```bash
python tradeBot_Pattern_Test.py
```

Run the lightweight contract tests:

```bash
python -m unittest discover -s tests
```

By default the symbol is `AAPL`. To test another equity, update `SYMBOL` in:

- `generate_shared_candles_stocks.py`
- `tradeBot_main.py`
- `tradeBot_Pattern_Test.py`

## Configuration

The public configuration is intentionally small:

- `config.py` contains only fee settings.
- No API keys are required for the public version.
- `SYMBOL` defaults to `AAPL`.
- Runtime state is stored in generated JSON files that Git ignores.
- The public version does not ship broker credentials or a random-data fallback.
  Run `generate_shared_candles_stocks.py` before starting either bot.

## Runtime Behavior

The runtime modules use regular weekday US equity market hours:

- Market timezone: `America/New_York`
- Open: `09:30`
- Close: `16:00`

The collector and bots sleep outside the active market window and wake at the next
regular open. A short post-close grace window is included for delayed free market
data, so the code does not treat delayed candles as an immediate feed failure.

## Generated Files

The bot can create runtime files such as:

- `shared_candles.csv`
- `Bot*_capital.json`
- `Bot*_pending_pattern.json`
- `Bot*_reinforcement_scores.json`
- `*_alerts.jsonl`

These files are ignored by Git and are not part of the public repository.

## Portfolio Notes

This project demonstrates:

- Data validation and defensive handling of malformed candle rows.
- Separation between graphical detection, indicator confirmation, and risk logic.
- Stateful bot operation with JSON persistence.
- Timezone-aware market-session scheduling.
- A reduced public strategy surface while preserving runnable project behavior.

## Limitations

- This version is intentionally limited to ten patterns.
- The candle collector uses `yfinance`, which can be delayed or unavailable.
- The market calendar is weekday/session based and does not model exchange holidays or half-days.
- This is not connected to a production broker execution system.
- No profitability claim is made.

## Disclaimer

This project is for engineering portfolio review and research discussion only.
It is not financial advice and it is not a production trading system.
