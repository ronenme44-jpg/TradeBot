"""Public fee configuration for the reduced trading-bot showcase.

The public repository intentionally keeps configuration minimal and excludes
broker credentials or private runtime settings.
"""

# Fees are stored as fractions, not percentages.
# Example: 0.001 means 0.10% per side.
CRYPTO_FEE_MAKER = 0.001
CRYPTO_FEE_TAKER = 0.001

# The public bots import FEE_BUY and FEE_SELL below.
# Change this value to "maker" only if your test setup should use maker fees.
CRYPTO_FEE_MODE = "taker"

# Keep one final buy/sell fee API for the rest of the codebase.
FEE_BUY = CRYPTO_FEE_TAKER if CRYPTO_FEE_MODE == "taker" else CRYPTO_FEE_MAKER
FEE_SELL = CRYPTO_FEE_TAKER if CRYPTO_FEE_MODE == "taker" else CRYPTO_FEE_MAKER
