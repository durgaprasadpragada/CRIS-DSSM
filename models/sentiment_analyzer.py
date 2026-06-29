"""
Sentiment Analysis Module
FinBERT-based sentiment analysis for financial news
"""
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class FinBERTSentimentAnalyzer:
    """
    Sentiment analysis using FinBERT
    Output: Positive, Neutral, Negative probabilities and combined sentiment score
    """

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.model_name = model_name
        self.pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self):
        """Load FinBERT model"""
        if self.pipe is not None:
            return

        try:
            from transformers import pipeline
            self.pipe = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                return_all_scores=True,
                device=0 if self.device == "cuda" else -1  # -1 for CPU
            )
            logger.info(f"Loaded FinBERT model on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load FinBERT: {e}")
            raise

    def predict(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Predict sentiment for multiple texts

        Returns: List of dicts with 'POSITIVE', 'NEUTRAL', 'NEGATIVE' scores
        """
        if self.pipe is None:
            self.load()

        # Filter empty texts
        texts = [t.strip() for t in texts if isinstance(t, str) and t.strip()]
        if not texts:
            texts = [""]

        results = self.pipe(texts)

        # Parse results into standardized format
        parsed = []
        for result in results:
            scores = {"POSITIVE": 0.0, "NEUTRAL": 0.0, "NEGATIVE": 0.0}

            if isinstance(result, list):
                for item in result:
                    label = item.get('label', '').upper()
                    score = float(item.get('score', 0.0))
                    if label in scores:
                        scores[label] = score
            elif isinstance(result, dict):
                label = result.get('label', '').upper()
                score = float(result.get('score', 0.0))
                if label in scores:
                    scores[label] = score

            parsed.append(scores)

        return parsed

    def compute_sentiment_score(self, probabilities: Dict[str, float]) -> float:
        """
        Compute combined sentiment score from probabilities
        Formula: S_t = P_pos - P_neg
        Range: [-1, 1]
        """
        pos = float(probabilities.get('POSITIVE', 0.0))
        neg = float(probabilities.get('NEGATIVE', 0.0))
        return pos - neg

    def analyze(self, texts: List[str]) -> pd.DataFrame:
        """
        Analyze texts and return comprehensive sentiment results
        """
        if not texts:
            return pd.DataFrame(columns=[
                'text', 'positive', 'neutral', 'negative', 'sentiment'
            ])

        predictions = self.predict(texts)

        results = []
        for text, probs in zip(texts, predictions):
            sentiment = self.compute_sentiment_score(probs)
            results.append({
                'text': text,
                'positive': float(probs.get('POSITIVE', 0.0)),
                'neutral': float(probs.get('NEUTRAL', 0.0)),
                'negative': float(probs.get('NEGATIVE', 0.0)),
                'sentiment': sentiment
            })

        return pd.DataFrame(results)


class SentimentShockComputer:
    """
    Compute sentiment shock from sentiment time series
    Shock measures the surprise or magnitude of change in sentiment
    """

    def __init__(self, rolling_window: int = 7, baseline_sentiment: float = 0.0):
        self.rolling_window = rolling_window
        self.baseline_sentiment = baseline_sentiment

    def compute_shock(self, sentiment_series: pd.Series) -> pd.Series:
        """
        Compute shock as deviation from baseline
        Shock = S_t - Baseline

        Large positive shock: Good news
        Large negative shock: Bad news
        Small shock: Neutral/expected news
        """
        if len(sentiment_series) < 1:
            return sentiment_series

        # Use baseline sentiment (historical average or neutral)
        shock = sentiment_series - self.baseline_sentiment
        return shock

    def compute_shock_from_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute shock from dataframe with sentiment column"""
        df = df.copy()
        df['shock'] = self.compute_shock(df['sentiment'])
        return df

    def update_baseline(self, new_baseline: float):
        """Update the baseline sentiment for shock calculation"""
        self.baseline_sentiment = new_baseline


class CoinDetector:
    """Detect and map cryptocurrencies mentioned in text"""

    def __init__(self, crypto_aliases: Dict[str, str] = None):
        """
        Initialize with cryptocurrency aliases/mappings
        aliases: dict mapping 'coin_mention' -> 'standardized_name'
        """
        self.aliases = crypto_aliases or self._default_aliases()

    @staticmethod
    def _default_aliases() -> Dict[str, str]:
        """Default cryptocurrency aliases - mapped to actual market data file names"""
        return {
            # Bitcoin
            'bitcoin': 'bitcoin',
            'btc': 'bitcoin',
            # Ethereum
            'ethereum': 'ethereum',
            'eth': 'ethereum',
            # Major coins
            'ripple': 'xrp',
            'xrp': 'xrp',
            'solana': 'solana',
            'sol': 'solana',
            'cardano': 'cardano',
            'ada': 'cardano',
            'litecoin': 'litecoin',
            'ltc': 'litecoin',
            'dogecoin': 'dogecoin',
            'doge': 'dogecoin',
            'polkadot': 'polkadot',
            'dot': 'polkadot',
            'binance': 'BNB',
            'bnb': 'BNB',
            'chainlink': 'chainlink',
            'link': 'chainlink',
            # Additional coins
            'avalanche': 'avalanche',
            'avax': 'avalanche',
            'polygon': 'polygon',
            'matic': 'polygon',
            'cosmos': 'cosmos',
            'atom': 'cosmos',
            'near': 'near protocol',
            'stellar': 'stellar',
            'xlm': 'stellar',
            'tron': 'tron',
            'trx': 'tron',
            'monero': 'monero',
            'xmr': 'monero',
            'vechain': 'VeChain',
            'vet': 'VeChain',
            'tezos': 'Tezos',
            'xtz': 'Tezos',
            'filecoin': 'Filecoin',
            'fil': 'Filecoin',
            'theta': 'Theta Network',
            'theta network': 'Theta Network',
            'maker': 'Maker',
            'mkr': 'Maker',
            'aave': 'Aave',
            'compound': 'Compound',
            'uniswap': 'uniswap',
            'uni': 'uniswap',
            'curve': 'Curve DAO Token',
            'sushi': 'Sushi',
            'yearn': 'yearn.finance',
            'yfi': 'yearn.finance',
        }

    def detect(self, text: str) -> List[str]:
        """Detect mentioned cryptocurrencies in text"""
        import re
        text_lower = text.lower()
        detected = set()

        for mention, standard_name in self.aliases.items():
            # Word boundary matching
            pattern = r'\b' + re.escape(mention) + r'\b'
            if re.search(pattern, text_lower):
                detected.add(standard_name)

        return sorted(list(detected))

    def weight_coins(self, detected_coins: List[str], all_coins: List[str], text: str = "") -> Dict[str, float]:
        """
        Weight coins based on whether they're mentioned and context
        Detected coins get high weight based on mention frequency and position
        """
        weights = {coin: 0.0 for coin in all_coins}

        if not detected_coins:
            # No detection: equal distribution
            weight = 1.0 / len(all_coins) if all_coins else 0.0
            for coin in weights:
                weights[coin] = weight
            return weights

        # Filter detected coins to those in available markets
        detected_coins = [c for c in detected_coins if c in weights]

        if not detected_coins:
            # No valid detected coins: equal distribution
            weight = 1.0 / len(all_coins) if all_coins else 0.0
            for coin in weights:
                weights[coin] = weight
            return weights

        # Count mentions in text for each detected coin
        import re
        text_lower = text.lower() if text else ""
        mention_counts = {}
        
        for coin in detected_coins:
            # Count how many times this coin is mentioned
            count = 0
            for mention, standard_name in self.aliases.items():
                if standard_name == coin:
                    pattern = r'\b' + re.escape(mention) + r'\b'
                    count += len(re.findall(pattern, text_lower))
            mention_counts[coin] = max(1, count)  # At least 1 if detected

        # Calculate weights based on mention counts
        total_mentions = sum(mention_counts.values())
        
        # Detected coins get 95% of weight distributed by mention frequency
        detected_share = 0.95
        for coin in detected_coins:
            if total_mentions > 0:
                weights[coin] = detected_share * (mention_counts[coin] / total_mentions)
            else:
                weights[coin] = detected_share / len(detected_coins)

        # Remaining coins share 5% equally
        others = [c for c in all_coins if c not in detected_coins]
        if others:
            other_share = 0.05 / len(others)
            for coin in others:
                weights[coin] = other_share

        # Normalize to ensure sum = 1.0
        total = sum(weights.values())
        if total > 0:
            for coin in weights:
                weights[coin] = weights[coin] / total

        return weights


def create_sentiment_analyzer() -> FinBERTSentimentAnalyzer:
    """Factory function"""
    return FinBERTSentimentAnalyzer()
