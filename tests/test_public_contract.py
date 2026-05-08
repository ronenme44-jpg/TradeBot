"""Contract tests for the reduced public trading-bot repository."""

from __future__ import annotations

import importlib
import pathlib
import sys
import unittest


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


if __name__ == "__main__":
    unittest.main()
