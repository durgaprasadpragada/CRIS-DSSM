"""
Core CRIS DSSM Pipeline
Orchestrates the complete analysis workflow
"""
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from config import (
    NEWS_DATA_PATH, MARKET_DATA_DIR, SENTIMENT_MODEL, STATE_SPACE_HORIZON,
    ROLLING_WINDOW, CRYPTO_ALIASES
)
from utils.data_loader import NewsDataLoader, MarketDataLoader, DataValidator
from utils.text_processor import TextPreprocessor
from models.sentiment_analyzer import FinBERTSentimentAnalyzer, SentimentShockComputer, CoinDetector
from models.feature_engineering import MarketFeatureEngineer, DataMerger
from models.dssm import estimate_state_space, kalman_forecast
from utils.metrics import RegressionMetrics, RiskMetrics, RiskRanker, evaluate_predictions

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CRISDSSMPipeline:
    """
    Complete CRIS DSSM Pipeline

    Steps:
    1. Load news and market data
    2. Preprocess text
    3. Run sentiment analysis (FinBERT)
    4. Compute sentiment shocks
    5. Detect coin impacts
    6. Load and engineer market features
    7. Merge sentiment and market data
    8. Estimate state space model
    9. Forecast with Kalman filter
    10. Compute risk metrics
    11. Rank and evaluate
    """

    def __init__(self):
        """Initialize pipeline components"""
        logger.info("Initializing CRIS DSSM Pipeline...")

        # Data loaders
        self.news_loader = NewsDataLoader(str(NEWS_DATA_PATH))
        self.market_loader = MarketDataLoader(str(MARKET_DATA_DIR))

        # Text processing
        self.text_preprocessor = TextPreprocessor()

        # Sentiment analysis
        self.sentiment_analyzer = FinBERTSentimentAnalyzer(SENTIMENT_MODEL)
        self.shock_computer = SentimentShockComputer(
            rolling_window=ROLLING_WINDOW)
        self.coin_detector = CoinDetector(CRYPTO_ALIASES)

        # Market features
        self.feature_engineer = MarketFeatureEngineer()

        # Model components
        self.models: Dict[str, Dict] = {}  # Trained models per coin
        self.data_validator = DataValidator()

        # Tracking
        self.progress_log: List[str] = []

    def log_progress(self, message: str):
        """Log progress message"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        msg = f"[{timestamp}] {message}"
        self.progress_log.append(msg)
        logger.info(message)

    def run(self, articles: List[str]) -> Dict:
        """
        Run complete pipeline on articles

        Args:
            articles: List of news articles (can be full text or summaries)

        Returns:
            Dictionary with complete analysis results
        """
        self.progress_log = []
        self.log_progress("Starting CRIS DSSM analysis...")

        try:
            # Step 1: Load data
            self.log_progress("Loading market data...")
            markets = self.market_loader.load_all()
            coin_names = sorted(list(markets.keys()))

            if not markets:
                self.log_progress("ERROR: No market data loaded")
                return self._empty_results()

            # Step 2: Process sentiment
            self.log_progress(f"Processing {len(articles)} articles...")
            article_df = self._process_sentiment(articles)

            if article_df.empty:
                self.log_progress("WARNING: No articles to process")
                return self._empty_results()

            # Step 3: Compute shocks
            self.log_progress("Computing sentiment shocks...")
            # Use neutral baseline (0.0) for shock calculation
            self.shock_computer.update_baseline(0.0)
            article_df = self.shock_computer.compute_shock_from_df(article_df)
            
            # Log shock statistics
            if not article_df.empty:
                self.log_progress(f"  Shock range: [{article_df['shock'].min():.3f}, {article_df['shock'].max():.3f}]")
                self.log_progress(f"  Shock mean: {article_df['shock'].mean():.3f}")

            # Step 4: Detect coin impacts
            self.log_progress("Mapping cryptocurrency impacts...")
            shock_history = self._compute_coin_impacts(article_df)

            # Step 5: Train models and compute risk
            self.log_progress("Fitting Dynamic State Space Models...")
            results = []
            overall_metrics = {'RMSE': [], 'MAE': [], 'R2': [], 'MAPE': []}

            for coin in coin_names:
                if coin not in markets:
                    continue

                market_df = markets[coin]

                try:
                    # Validate data
                    valid, msg = self.data_validator.validate_market(
                        market_df, coin)
                    if not valid:
                        logger.warning(msg)
                        continue

                    # Engineer features
                    features = self.feature_engineer.compute_all_features(
                        market_df)
                    vol_series = features['volatility'].values

                    # Get shocks for this coin
                    coin_shocks = shock_history[shock_history['coin'] == coin].set_index('date')[
                        'shock']
                    coin_shocks = coin_shocks.reindex(
                        features['date'], fill_value=0.0).values

                    # Fit model
                    if len(vol_series) >= 3:
                        result = self._fit_and_forecast_coin(
                            coin, vol_series, coin_shocks, features
                        )
                        if result:
                            results.append(result)
                            # Aggregate metrics
                            for key in ['RMSE', 'MAE', 'R2', 'MAPE']:
                                if key in result:
                                    overall_metrics[key].append(result[key])

                except Exception as e:
                    logger.warning(f"Error processing {coin}: {e}")
                    continue

            # Step 6: Compute aggregate metrics
            self.log_progress("Computing aggregate metrics...")
            agg_metrics = self._aggregate_metrics(overall_metrics)

            # Step 7: Rank coins
            self.log_progress("Ranking cryptocurrencies...")
            highest_risk, safest = RiskRanker.rank_coins(results)

            # Compile results
            self.log_progress("Analysis complete!")

            return {
                'timestamp': datetime.now().isoformat(),
                'articles_analyzed': len(articles),
                'coins_analyzed': len(results),
                'results': results,
                'highest_risk_coins': highest_risk,
                'safest_coins': safest,
                'model_metrics': agg_metrics,
                'article_sentiment': article_df[['sentiment', 'shock']].describe().to_dict(),
                'progress_log': self.progress_log,
            }

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            self.log_progress(f"ERROR: {e}")
            return self._empty_results()

    def _process_sentiment(self, articles: List[str]) -> pd.DataFrame:
        """Process sentiment for articles"""
        # Clean text
        cleaned = self.text_preprocessor.preprocess_batch(articles)

        # Analyze sentiment
        sentiment_df = self.sentiment_analyzer.analyze(cleaned)
        sentiment_df['date'] = datetime.now().date()
        
        # Log sentiment results
        for idx, row in sentiment_df.iterrows():
            self.log_progress(f"  Article {idx+1}: Pos={row['positive']:.3f}, Neu={row['neutral']:.3f}, Neg={row['negative']:.3f}, Score={row['sentiment']:.3f}")

        return sentiment_df

    def _compute_coin_impacts(self, article_df: pd.DataFrame) -> pd.DataFrame:
        """Compute impacts per coin"""
        rows = []
        all_coins = list(self.market_loader.get_all_markets().keys())
        
        for idx, row in article_df.iterrows():
            text = row['text']
            shock = row['shock']
            sentiment = row['sentiment']
            
            # Log article details
            self.log_progress(f"  Article {idx+1}: Sentiment={sentiment:.3f}, Shock={shock:.3f}")

            # Detect coins
            detected = self.coin_detector.detect(text)
            self.log_progress(f"  Detected coins: {detected}")

            if detected:
                # Weight coins with text context
                weights = self.coin_detector.weight_coins(
                    detected, all_coins, text)
                
                # Log top weights
                top_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:5]
                self.log_progress(f"  Top weights: {top_weights}")

                # Apply shock to each coin
                for coin, weight in weights.items():
                    if weight > 0.01:  # Only log significant weights
                        impact = weight * shock
                        rows.append({
                            'date': row['date'],
                            'coin': coin,
                            'shock': impact,
                            'weight': weight,
                            'original_shock': shock
                        })

        if not rows:
            # Return empty but properly structured
            self.log_progress("  No coin impacts computed")
            return pd.DataFrame(columns=['date', 'coin', 'shock', 'weight', 'original_shock'])

        df = pd.DataFrame(rows)
        # Aggregate shocks by coin and date
        df = df.groupby(['coin', 'date'], as_index=False).agg({
            'shock': 'sum',
            'weight': 'first',
            'original_shock': 'first'
        })
        
        self.log_progress(f"  Generated {len(df)} coin impact records")
        return df

    def _fit_and_forecast_coin(self, coin: str, vol_series: np.ndarray,
                               shocks: np.ndarray, features: pd.DataFrame) -> Optional[Dict]:
        """Fit DSSM and compute risk for a coin"""
        try:
            # Log input statistics
            total_shock = float(np.sum(shocks))
            max_shock = float(np.max(np.abs(shocks)))
            self.log_progress(f"  {coin}: Total shock={total_shock:.4f}, Max shock={max_shock:.4f}")
            
            # Fit state space model
            phi, beta, q, r, smoothed_z, smoothed_p = estimate_state_space(
                vol_series, shocks)

            # Get last state
            z_last = float(smoothed_z[-1])
            p_last = float(smoothed_p[-1])
            
            self.log_progress(f"  {coin}: Hidden state={z_last:.4f}, Variance={p_last:.6f}")
            self.log_progress(f"  {coin}: Model params - φ={phi:.4f}, β={beta:.4f}")

            # Forecast
            forecasts = kalman_forecast(
                z_last, p_last, phi, beta, q, STATE_SPACE_HORIZON)

            # Convert to risk scores
            current_risk = RiskMetrics.sigmoid(z_last)
            future_risks = [RiskMetrics.sigmoid(f[0]) for f in forecasts]
            
            self.log_progress(f"  {coin}: Current risk={current_risk:.4f}, 7-day forecast={future_risks[1]:.4f}")

            # Confidence
            confidence = RiskMetrics.compute_confidence(p_last)

            # Compute metrics
            y_true = vol_series[1:]
            y_pred = np.array([phi * vol_series[t-1] + beta * shocks[t]
                              for t in range(1, len(vol_series))])
            y_pred_baseline = np.array([phi * vol_series[t-1]
                                       for t in range(1, len(vol_series))])

            metrics = evaluate_predictions(y_true, y_pred, y_pred_baseline)

            # Variance across forecast
            variances = {f't+{h}': float(forecasts[i][1])
                         for i, h in enumerate(STATE_SPACE_HORIZON)}

            # Compile result
            result = {
                'coin': coin,
                'current_risk': current_risk,
                'risk_category': RiskRanker.categorize_risk(current_risk),
                'future_risks': {f't+{h}': future_risks[i] for i, h in enumerate(STATE_SPACE_HORIZON)},
                'variance': float(p_last),
                'confidence': confidence,
                'forecast_variances': variances,
                'model_params': {
                    'phi': phi,
                    'beta': beta,
                    'q': q,
                    'r': r,
                },
                'impact': total_shock,
                'max_shock': max_shock,
            }

            # Add metrics
            for key, value in metrics.items():
                result[key] = value

            return result

        except Exception as e:
            logger.warning(f"Failed to fit model for {coin}: {e}")
            return None

    def _aggregate_metrics(self, metrics_dict: Dict) -> Dict:
        """Aggregate metrics across coins"""
        agg = {}
        for key, values in metrics_dict.items():
            if values:
                agg[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                }
        return agg

    def _empty_results(self) -> Dict:
        """Return empty results structure"""
        return {
            'timestamp': datetime.now().isoformat(),
            'articles_analyzed': 0,
            'coins_analyzed': 0,
            'results': [],
            'highest_risk_coins': [],
            'safest_coins': [],
            'model_metrics': {},
            'article_sentiment': {},
            'progress_log': self.progress_log,
            'error': 'Pipeline failed to complete',
        }


def create_pipeline() -> CRISDSSMPipeline:
    """Factory function"""
    return CRISDSSMPipeline()
