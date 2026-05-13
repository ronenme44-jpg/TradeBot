"""Contract tests for the reduced public trading-bot repository."""

from __future__ import annotations

import importlib
import pathlib
import sys
import unittest
from datetime import datetime, timezone

import pandas as pd


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

EXPECTED_PATTERNS = [
    "bullish_engulfing",
    "hammer",
    "morning_star",
    "piercing_pattern",
    "three_white_soldiers_simple",
    "bearish_engulfing",
    "shooting_star",
    "evening_star",
    "three_black_crows",
    "dark_cloud_cover",
]

RUNTIME_FILES = [
    "shared_candles.csv",
    "Bot1_capital.json",
    "Bot1_pending_pattern.json",
    "Bot1_reinforcement_scores.json",
    "Bot1_closed_trades_log.json",
    "BotTest_capital.json",
    "BotTest_pending_pattern.json",
    "BotTest_reinforcement_scores.json",
]


class PublicRepoContractTest(unittest.TestCase):
    def test_imports_do_not_create_runtime_files(self) -> None:
        """Importing modules should not start network work or write local state."""
        before = {name: (REPO_ROOT / name).exists() for name in RUNTIME_FILES}

        importlib.import_module("generate_shared_candles_stocks")
        importlib.import_module("tradeBot_main")
        importlib.import_module("tradeBot_Pattern_Test")

        after = {name: (REPO_ROOT / name).exists() for name in RUNTIME_FILES}
        self.assertEqual(before, after)

    def test_public_pattern_contract_is_aligned(self) -> None:
        """The reduced public version must stay aligned at exactly ten patterns."""
        graphical = importlib.import_module("tradeBot_Graphical_Pattern")
        indicators = importlib.import_module("tradeBot_indicators_pattern")
        rules = importlib.import_module("tradeBot_Indicator_Rules")
        main_bot = importlib.import_module("tradeBot_main")
        training_bot = importlib.import_module("tradeBot_Pattern_Test")

        self.assertEqual([item["name"] for item in main_bot.pattern_list], EXPECTED_PATTERNS)
        self.assertEqual([item["name"] for item in training_bot.pattern_list], EXPECTED_PATTERNS)
        self.assertEqual(list(rules.pattern_indicator_rules), EXPECTED_PATTERNS)
        self.assertEqual(list(indicators.confirmation_functions), EXPECTED_PATTERNS)

        long_names = [name for name, _func in graphical.LONG_PATTERN_FUNCS]
        short_names = [name for name, _func in graphical.SHORT_PATTERN_FUNCS]
        self.assertEqual(long_names + short_names, EXPECTED_PATTERNS)

    def test_reinforcement_key_contains_declared_indicators(self) -> None:
        """Reinforcement keys should be deterministic and use the rule order."""
        rules = importlib.import_module("tradeBot_Indicator_Rules")
        outcomes = {
            "rsi": True,
            "volume": False,
            "volatility": True,
            "trend": True,
        }
        key = rules.build_pattern_key("bullish_engulfing", outcomes)
        self.assertEqual(
            key,
            "bullish_engulfing|rsi:1,volume:0,volatility:1,trend:1",
        )

    def test_market_session_sleep_windows_are_aligned(self) -> None:
        """Runtime modules should stop polling outside the market active window."""
        modules = [
            importlib.import_module("generate_shared_candles_stocks"),
            importlib.import_module("tradeBot_main"),
            importlib.import_module("tradeBot_Pattern_Test"),
        ]

        # 2026-05-08 is a Friday. 13:30 UTC is 09:30 in New York.
        pre_open_warmup_utc = datetime(2026, 5, 8, 13, 29, tzinfo=timezone.utc)
        market_open_utc = datetime(2026, 5, 8, 13, 30, tzinfo=timezone.utc)
        post_close_grace_utc = datetime(2026, 5, 8, 20, 10, tzinfo=timezone.utc)
        after_grace_utc = datetime(2026, 5, 8, 20, 22, tzinfo=timezone.utc)

        collector = modules[0]
        collector_warmup = collector.market_session_status(pre_open_warmup_utc)
        self.assertTrue(collector_warmup["is_pre_open_warmup"])
        self.assertFalse(collector_warmup["is_open"])
        self.assertTrue(collector_warmup["is_active_window"])

        for bot_module in modules[1:]:
            bot_warmup = bot_module.market_session_status(pre_open_warmup_utc)
            self.assertFalse(bot_warmup["is_open"])
            self.assertFalse(bot_warmup["is_active_window"])

        for module in modules:
            with self.subTest(module=module.__name__):
                open_status = module.market_session_status(market_open_utc)
                self.assertTrue(open_status["is_open"])
                self.assertTrue(open_status["is_active_window"])

                grace_status = module.market_session_status(post_close_grace_utc)
                self.assertFalse(grace_status["is_open"])
                self.assertTrue(grace_status["is_post_close_grace"])
                self.assertTrue(grace_status["is_active_window"])

                closed_status = module.market_session_status(after_grace_utc)
                self.assertFalse(closed_status["is_open"])
                self.assertFalse(closed_status["is_active_window"])
                self.assertEqual(
                    closed_status["next_open_dt"].strftime("%Y-%m-%d %H:%M"),
                    "2026-05-11 09:30",
                )

    def test_closed_trade_logging_policy_is_explicit(self) -> None:
        """Main bot keeps closed-trade logs; training bot keeps that path disabled."""
        main_bot = importlib.import_module("tradeBot_main")
        training_bot = importlib.import_module("tradeBot_Pattern_Test")

        self.assertTrue(hasattr(main_bot, "update_closed_trade_log"))
        self.assertTrue(callable(main_bot.update_closed_trade_log))

        main_source = (REPO_ROOT / "tradeBot_main.py").read_text(encoding="utf-8")
        training_source = (REPO_ROOT / "tradeBot_Pattern_Test.py").read_text(encoding="utf-8")

        self.assertIn("init_file(CLOSED_LOG_PATH, DEFAULT_CLOSED_LOG)", main_source)
        self.assertIn("update_closed_trade_log({**open_trade})", main_source)
        self.assertIn("# init_file(CLOSED_LOG_PATH, DEFAULT_CLOSED_LOG)", training_source)
        self.assertIn("# update_closed_trade_log({**open_trade})", training_source)
        self.assertFalse(hasattr(training_bot, "update_closed_trade_log"))

    def test_api_fallback_never_fabricates_candles(self) -> None:
        """Missing CSV/API config should fail closed instead of returning random OHLCV."""
        modules = [
            importlib.import_module("tradeBot_main"),
            importlib.import_module("tradeBot_Pattern_Test"),
        ]

        for module in modules:
            with self.subTest(module=module.__name__):
                original_log_alert = module.log_alert
                module.log_alert = lambda *args, **kwargs: None
                try:
                    self.assertIsNone(module.load_candles_from_api(limit=5))
                finally:
                    module.log_alert = original_log_alert

    def test_current_session_unfilled_gaps_block_entries(self) -> None:
        """Large unfilled gaps in the latest session should close the entry gate."""
        training_bot = importlib.import_module("tradeBot_Pattern_Test")

        gap_df = pd.DataFrame({
            "timestamp": [
                "2026-05-08T13:30:00Z",
                "2026-05-08T13:31:00Z",
                "2026-05-08T13:35:00Z",
            ]
        })
        report = training_bot.detect_unfilled_candle_gap_issue(gap_df, interval_minutes=1)
        self.assertTrue(report["block_entries"])
        self.assertEqual(report["missing_total"], 3)
        self.assertEqual(report["max_missing_in_gap"], 3)

        previous_day_gap_df = pd.DataFrame({
            "timestamp": [
                "2026-05-07T13:30:00Z",
                "2026-05-07T13:35:00Z",
                "2026-05-08T13:30:00Z",
                "2026-05-08T13:31:00Z",
            ]
        })
        report = training_bot.detect_unfilled_candle_gap_issue(previous_day_gap_df, interval_minutes=1)
        self.assertFalse(report["block_entries"])
        self.assertEqual(report["missing_total"], 0)


if __name__ == "__main__":
    unittest.main()
