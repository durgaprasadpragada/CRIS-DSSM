"""
Data Loading Module
Handles loading and normalization of news and market data
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class NewsDataLoader:
    """Load and validate news data from CSV"""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df = None

    def load(self) -> pd.DataFrame:
        """Load news data from CSV"""
        if not os.path.exists(self.filepath):
            logger.warning(f"News file not found: {self.filepath}")
            return pd.DataFrame(columns=['date', 'title', 'text'])

        df = pd.read_csv(self.filepath)

        # Normalize column names
        col_map = {}
        for col in df.columns:
            col_lower = col.lower().strip()
            if col_lower in ['date', 'time', 'timestamp']:
                col_map[col] = 'date'
            elif col_lower in ['title', 'headline']:
                col_map[col] = 'title'
            elif col_lower in ['text', 'body', 'content', 'article']:
                col_map[col] = 'text'
            elif col_lower in ['url', 'link']:
                col_map[col] = 'url'
            elif col_lower in ['source']:
                col_map[col] = 'source'

        df = df.rename(columns=col_map)

        # Ensure required columns
        if 'date' not in df.columns:
            df['date'] = datetime.now()
        if 'title' not in df.columns:
            df['title'] = ""
        if 'text' not in df.columns:
            df['text'] = ""

        # Parse dates
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # Keep only required columns
        df = df[['date', 'title', 'text']].copy()
        df = df.dropna(subset=['date'])
        df = df.sort_values('date').reset_index(drop=True)

        logger.info(f"Loaded {len(df)} news articles from {self.filepath}")
        self.df = df
        return df

    def get_df(self) -> pd.DataFrame:
        """Get loaded dataframe"""
        if self.df is None:
            return self.load()
        return self.df


class MarketDataLoader:
    """Load and normalize market data from multiple CSV files"""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.markets: Dict[str, pd.DataFrame] = {}

    def load_all(self) -> Dict[str, pd.DataFrame]:
        """Load all market CSV files"""
        if not os.path.isdir(self.data_dir):
            logger.warning(f"Market data directory not found: {self.data_dir}")
            return {}

        files = [f for f in os.listdir(
            self.data_dir) if f.lower().endswith('.csv')]
        logger.info(f"Found {len(files)} market data files")

        for filename in files:
            filepath = os.path.join(self.data_dir, filename)
            coin_name = os.path.splitext(filename)[0]

            # Skip unwanted files
            if coin_name.lower() == "current crypto leaderboard":
                continue

            try:
                df = self._load_and_normalize(filepath)
                if df is not None and not df.empty:
                    self.markets[coin_name] = df
                    logger.info(f"Loaded {coin_name}: {len(df)} records")
            except Exception as e:
                logger.warning(f"Failed to load {filename}: {e}")

        logger.info(
            f"Successfully loaded {len(self.markets)} cryptocurrency markets")
        return self.markets

    def _load_and_normalize(self, filepath: str) -> Optional[pd.DataFrame]:
        """Load and normalize a single market CSV file"""
        df = pd.read_csv(filepath)

        if df.empty:
            return None

        # Find date column
        date_col = None
        for col in df.columns:
            if col.lower() in ['date', 'time', 'timestamp', 'datetime']:
                date_col = col
                break

        if date_col is None:
            date_col = df.columns[0]

        # Find close/price column
        close_col = None
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in ['close', 'close_price', 'price', 'adj close', 'adj_close']:
                close_col = col
                break

        if close_col is None:
            # Find any numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                close_col = numeric_cols[-1]  # Usually last numeric is close

        if close_col is None:
            return None

        # Parse dates
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except:
            return None

        # Select and normalize columns
        df = df[[date_col, close_col]].copy()
        df = df.rename(columns={date_col: 'date', close_col: 'close'})
        df = df.sort_values('date').reset_index(drop=True)

        # Remove duplicates and NaN
        df = df.drop_duplicates(subset=['date'])
        df = df.dropna()

        # Ensure numeric close prices
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df = df.dropna(subset=['close'])

        return df if not df.empty else None

    def get_market(self, coin_name: str) -> Optional[pd.DataFrame]:
        """Get a specific market dataframe"""
        return self.markets.get(coin_name)

    def get_all_markets(self) -> Dict[str, pd.DataFrame]:
        """Get all loaded markets"""
        return self.markets

    def get_coin_names(self) -> list:
        """Get list of loaded coin names"""
        return sorted(list(self.markets.keys()))


class DataValidator:
    """Validate data quality and completeness"""

    @staticmethod
    def validate_news(df: pd.DataFrame) -> Tuple[bool, str]:
        """Validate news dataframe"""
        if df.empty:
            return False, "News dataframe is empty"

        required_cols = ['date', 'title', 'text']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            return False, f"Missing columns: {missing}"

        return True, "News data valid"

    @staticmethod
    def validate_market(df: pd.DataFrame, coin_name: str) -> Tuple[bool, str]:
        """Validate market dataframe"""
        if df.empty:
            return False, f"{coin_name}: dataframe is empty"

        required_cols = ['date', 'close']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            return False, f"{coin_name}: Missing columns {missing}"

        if len(df) < 3:
            return False, f"{coin_name}: Too few data points ({len(df)})"

        return True, f"{coin_name}: Valid"
