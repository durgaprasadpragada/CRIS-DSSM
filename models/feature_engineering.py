"""
Market Feature Engineering Module
Compute market features for dynamic state space modeling
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class MarketFeatureEngineer:
    """Compute market-derived features from OHLCV data"""

    def __init__(self):
        pass

    def compute_returns(self, df: pd.DataFrame) -> pd.Series:
        """Compute daily returns"""
        return df['close'].pct_change()

    def compute_log_returns(self, df: pd.DataFrame) -> pd.Series:
        """Compute log returns"""
        return np.log(df['close'] / df['close'].shift(1))

    def compute_volatility(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Compute rolling volatility (standard deviation of returns)
        This is the key risk metric we're trying to predict
        """
        returns = self.compute_returns(df)
        volatility = returns.rolling(window=window, min_periods=1).std()
        return volatility

    def compute_rolling_mean(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Compute rolling mean price"""
        return df['close'].rolling(window=window, min_periods=1).mean()

    def compute_rolling_std(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Compute rolling standard deviation of price"""
        return df['close'].rolling(window=window, min_periods=1).std()

    def compute_momentum(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Compute momentum (price change over window)"""
        return df['close'].diff(window)

    def compute_price_change_pct(self, df: pd.DataFrame) -> pd.Series:
        """Compute percentage change from previous day"""
        return df['close'].pct_change() * 100

    def compute_moving_average(self, df: pd.DataFrame, window: int = 50) -> pd.Series:
        """Compute exponential moving average"""
        return df['close'].ewm(span=window, adjust=False).mean()

    def compute_rsi(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """
        Compute Relative Strength Index
        RSI = 100 - 100/(1 + RS)
        where RS = avg_gain / avg_loss
        """
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(
            window=window, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)
                ).rolling(window=window, min_periods=1).mean()

        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Neutral RSI for initial period

    def compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all market features"""
        features = pd.DataFrame(index=df.index)
        features['date'] = df['date']
        features['close'] = df['close']
        features['returns'] = self.compute_returns(df)
        features['log_returns'] = self.compute_log_returns(df)
        features['volatility'] = self.compute_volatility(df)
        features['rolling_mean'] = self.compute_rolling_mean(df)
        features['rolling_std'] = self.compute_rolling_std(df)
        features['momentum'] = self.compute_momentum(df)
        features['price_change_pct'] = self.compute_price_change_pct(df)
        features['sma_50'] = self.compute_moving_average(df, window=50)
        features['rsi'] = self.compute_rsi(df)

        # Fill NaN from rolling calculations
        features = features.bfill().ffill()

        return features

    @staticmethod
    def normalize_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        """Normalize features to [0, 1] using min-max scaling"""
        df_norm = df.copy()
        for col in feature_cols:
            if col in df_norm.columns:
                min_val = df_norm[col].min()
                max_val = df_norm[col].max()
                if max_val - min_val > 0:
                    df_norm[col] = (df_norm[col] - min_val) / \
                        (max_val - min_val)
        return df_norm


class DataMerger:
    """Merge news sentiment with market data"""

    @staticmethod
    def merge(market_df: pd.DataFrame,
              sentiment_shocks: pd.DataFrame,
              coin_name: str) -> pd.DataFrame:
        """
        Merge market features with sentiment shocks

        Args:
            market_df: DataFrame with date and market features
            sentiment_shocks: DataFrame with date and shock values
            coin_name: Name of cryptocurrency

        Returns:
            Merged DataFrame with aligned dates
        """
        merged = market_df.copy()
        merged['date'] = pd.to_datetime(merged['date']).dt.date

        # Add sentiment shock
        shock_dict = dict(zip(
            pd.to_datetime(sentiment_shocks['date']).dt.date,
            sentiment_shocks['shock']
        ))

        merged['shock'] = merged['date'].map(shock_dict).fillna(0.0)
        merged['coin'] = coin_name

        return merged
