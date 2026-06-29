"""
CRIS DSSM Configuration Module
Central configuration for all system parameters
"""
from pathlib import Path
from typing import Dict, List

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
NEWS_DATA_PATH = DATA_DIR / "news" / "cryptonews.csv"
MARKET_DATA_DIR = DATA_DIR / "market"

# Model parameters
SENTIMENT_MODEL = "ProsusAI/finbert"
ROLLING_WINDOW = 7  # For shock computation
STATE_SPACE_HORIZON = [1, 7, 30]  # Days ahead for forecasting

# Kalman filter parameters
KALMAN_PROCESS_NOISE = 1e-5
KALMAN_MEASUREMENT_NOISE = 1e-5

# Risk categories
RISK_CATEGORIES = {
    "Very High": (0.75, 1.0),
    "High": (0.5, 0.75),
    "Moderate": (0.25, 0.5),
    "Low": (0.0, 0.25),
}

# Cryptocurrency aliases and mappings (matched to actual CSV file names)
CRYPTO_ALIASES: Dict[str, str] = {
    # Bitcoin
    "bitcoin": "bitcoin",
    "btc": "bitcoin",
    # Ethereum
    "ethereum": "ethereum",
    "eth": "ethereum",
    # Ripple/XRP
    "ripple": "xrp",
    "xrp": "xrp",
    # Solana
    "solana": "solana",
    "sol": "solana",
    # Cardano
    "cardano": "cardano",
    "ada": "cardano",
    # Litecoin
    "litecoin": "litecoin",
    "ltc": "litecoin",
    # Dogecoin
    "dogecoin": "dogecoin",
    "doge": "dogecoin",
    # Polkadot
    "polkadot": "polkadot",
    "dot": "polkadot",
    # BNB
    "binance": "BNB",
    "bnb": "BNB",
    # Chainlink
    "chainlink": "chainlink",
    "link": "chainlink",
    # Tron
    "tron": "tron",
    "trx": "tron",
    # Stellar
    "stellar": "stellar",
    "xlm": "stellar",
    # Monero
    "monero": "monero",
    "xmr": "monero",
    # VeChain
    "vechain": "VeChain",
    "vet": "VeChain",
    # Polygon
    "polygon": "polygon",
    "matic": "polygon",
    # Additional coins
    "avalanche": "avalanche",
    "avax": "avalanche",
    "cosmos": "cosmos",
    "atom": "cosmos",
    "near": "near protocol",
    "tezos": "Tezos",
    "xtz": "Tezos",
    "filecoin": "Filecoin",
    "fil": "Filecoin",
    "theta": "Theta Network",
    "theta network": "Theta Network",
    "maker": "Maker",
    "mkr": "Maker",
    "aave": "Aave",
    "compound": "Compound",
    "uniswap": "uniswap",
    "uni": "uniswap",
    "curve": "Curve DAO Token",
}

# Coins to exclude from analysis
EXCLUDED_COINS = {"Current Crypto leaderboard"}

# Dashboard settings
DASHBOARD_HOST = "127.0.0.1"
DASHBOARD_PORT = 8050
DASHBOARD_DEBUG = True

# Logging
LOG_LEVEL = "INFO"
