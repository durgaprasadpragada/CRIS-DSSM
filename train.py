"""
CRIS-DSSM Training Pipeline
Trains Dynamic State Space Models using historical market and news data
"""
import logging
import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from config import (
    NEWS_DATA_PATH, MARKET_DATA_DIR, SENTIMENT_MODEL, 
    STATE_SPACE_HORIZON, ROLLING_WINDOW, CRYPTO_ALIASES
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

# Model save directory
MODEL_DIR = Path(__file__).parent / "models" / "trained"
MODEL_DIR.mkdir(exist_ok=True, parents=True)


class CRISDSSMTrainer:
    """
    Complete training pipeline for CRIS-DSSM
    
    Steps:
    1. Load historical market and news data
    2. Clean and engineer features
    3. Generate FinBERT sentiment for historical news
    4. Compute sentiment shocks
    5. Create coin impact mappings
    6. Train DSSM models for each coin
    7. Validate models
    8. Save trained models and parameters
    """

    def __init__(self):
        """Initialize trainer components"""
        logger.info("Initializing CRIS DSSM Trainer...")
        
        # Data loaders
        self.news_loader = NewsDataLoader(str(NEWS_DATA_PATH))
        self.market_loader = MarketDataLoader(str(MARKET_DATA_DIR))
        
        # Text processing
        self.text_preprocessor = TextPreprocessor()
        
        # Sentiment analysis
        self.sentiment_analyzer = FinBERTSentimentAnalyzer(SENTIMENT_MODEL)
        self.shock_computer = SentimentShockComputer(rolling_window=ROLLING_WINDOW)
        self.coin_detector = CoinDetector(CRYPTO_ALIASES)
        
        # Market features
        self.feature_engineer = MarketFeatureEngineer()
        
        # Data validator
        self.data_validator = DataValidator()
        
        # Storage for trained models
        self.trained_models: Dict[str, Dict] = {}
        self.validation_metrics: Dict[str, Dict] = {}
        self.trained_coins: List[str] = []
        
        # Progress tracking
        self.progress_log: List[str] = []

    def log_progress(self, message: str):
        """Log progress message"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        msg = f"[{timestamp}] {message}"
        self.progress_log.append(msg)
        logger.info(message)

    def run(self) -> Dict:
        """
        Run complete training pipeline
        
        Returns:
            Dictionary with training results and metadata
        """
        self.progress_log = []
        self.log_progress("Starting CRIS DSSM Training Pipeline...")
        
        try:
            # Step 1: Load data
            self.log_progress("Loading historical data...")
            news_df = self.news_loader.load()
            markets = self.market_loader.load_all()
            
            if news_df.empty:
                self.log_progress("ERROR: No news data loaded")
                return self._empty_results()
            
            if not markets:
                self.log_progress("ERROR: No market data loaded")
                return self._empty_results()
            
            self.log_progress(f"Loaded {len(news_df)} news articles")
            self.log_progress(f"Loaded {len(markets)} cryptocurrency datasets")
            
            # Step 2: Preprocess news text
            self.log_progress("Preprocessing news text...")
            news_df['cleaned_text'] = self.text_preprocessor.preprocess_batch(
                news_df['text'].tolist()
            )
            
            # Step 3: Generate FinBERT sentiment for historical news
            self.log_progress("Running FinBERT on historical news...")
            self.log_progress("Loading FinBERT model...")
            self.sentiment_analyzer.load()
            
            sentiment_df = self.sentiment_analyzer.analyze(
                news_df['cleaned_text'].tolist()
            )
            
            # Merge sentiment with news
            news_df = news_df.reset_index(drop=True)
            sentiment_df = sentiment_df.reset_index(drop=True)
            news_df['positive'] = sentiment_df['positive'].values
            news_df['neutral'] = sentiment_df['neutral'].values
            news_df['negative'] = sentiment_df['negative'].values
            news_df['sentiment'] = sentiment_df['sentiment'].values
            
            self.log_progress(f"Generated sentiment for {len(news_df)} articles")
            
            # Step 4: Compute sentiment shocks
            self.log_progress("Computing sentiment shocks...")
            # Use neutral baseline for shock calculation
            self.shock_computer.update_baseline(0.0)
            news_df['shock'] = self.shock_computer.compute_shock(news_df['sentiment'])
            
            self.log_progress(f"Shock range: [{news_df['shock'].min():.3f}, {news_df['shock'].max():.3f}]")
            
            # Step 5: Create coin impact mappings
            self.log_progress("Creating coin impact mappings...")
            coin_impacts = self._create_historical_impacts(news_df)
            
            # Step 6: Train models for each coin
            self.log_progress("Training DSSM models...")
            coin_names = sorted(list(markets.keys()))
            
            for coin in coin_names:
                try:
                    # Validate data
                    market_df = markets[coin]
                    valid, msg = self.data_validator.validate_market(market_df, coin)
                    if not valid:
                        logger.warning(msg)
                        continue
                    
                    # Engineer features
                    self.log_progress(f"  Training {coin}...")
                    features = self.feature_engineer.compute_all_features(market_df)
                    
                    # Merge with sentiment shocks
                    coin_shocks = coin_impacts[coin_impacts['coin'] == coin]
                    if coin_shocks.empty:
                        # No historical shocks for this coin, use neutral
                        features['shock'] = 0.0
                    else:
                        features = DataMerger.merge(features, coin_shocks, coin)
                    
                    # Train DSSM
                    vol_series = features['volatility'].values
                    shock_series = features['shock'].values
                    
                    if len(vol_series) >= 10:  # Minimum data requirement
                        model_result = self._train_coin_model(
                            coin, vol_series, shock_series, features
                        )
                        if model_result:
                            self.trained_models[coin] = model_result
                            self.trained_coins.append(coin)
                            self.log_progress(f"  ✓ {coin} trained successfully")
                    else:
                        self.log_progress(f"  ✗ {coin} skipped (insufficient data)")
                
                except Exception as e:
                    logger.warning(f"Error training {coin}: {e}")
                    continue
            
            # Step 7: Validate models
            self.log_progress("Validating models...")
            self._validate_models()
            
            # Step 8: Save trained models
            self.log_progress("Saving trained models...")
            self._save_models()
            
            # Step 9: Save metadata
            self.log_progress("Saving metadata...")
            self._save_metadata()
            
            self.log_progress("Training complete!")
            
            return {
                'timestamp': datetime.now().isoformat(),
                'coins_trained': len(self.trained_coins),
                'trained_coins': self.trained_coins,
                'validation_metrics': self.validation_metrics,
                'progress_log': self.progress_log,
                'model_dir': str(MODEL_DIR),
            }
        
        except Exception as e:
            logger.error(f"Training error: {e}")
            self.log_progress(f"ERROR: {e}")
            return self._empty_results()

    def _create_historical_impacts(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Create historical coin impact mappings"""
        rows = []
        all_coins = list(self.market_loader.get_all_markets().keys())
        
        for idx, row in news_df.iterrows():
            text = row['text']
            shock = row['shock']
            date = row.get('date', datetime.now().date())
            
            # Detect coins
            detected = self.coin_detector.detect(text)
            
            if detected:
                # Weight coins
                weights = self.coin_detector.weight_coins(detected, all_coins, text)
                
                # Apply shock to each coin
                for coin, weight in weights.items():
                    if weight > 0.01:
                        impact = weight * shock
                        rows.append({
                            'date': date,
                            'coin': coin,
                            'shock': impact,
                            'weight': weight
                        })
        
        if not rows:
            return pd.DataFrame(columns=['date', 'coin', 'shock', 'weight'])
        
        df = pd.DataFrame(rows)
        df = df.groupby(['coin', 'date'], as_index=False).agg({
            'shock': 'sum',
            'weight': 'first'
        })
        
        return df

    def _train_coin_model(self, coin: str, vol_series: np.ndarray, 
                          shocks: np.ndarray, features: pd.DataFrame) -> Optional[Dict]:
        """Train DSSM model for a single coin"""
        try:
            # Fit state space model
            phi, beta, q, r, smoothed_z, smoothed_p = estimate_state_space(
                vol_series, shocks, maxiter=200
            )
            
            # Get final state
            z_last = float(smoothed_z[-1])
            p_last = float(smoothed_p[-1])
            
            # Compute validation metrics
            y_true = vol_series[1:]
            y_pred = np.array([phi * vol_series[t-1] + beta * shocks[t]
                              for t in range(1, len(vol_series))])
            y_pred_baseline = np.array([phi * vol_series[t-1]
                                       for t in range(1, len(vol_series))])
            
            metrics = evaluate_predictions(y_true, y_pred, y_pred_baseline)
            
            # Store model parameters
            model_result = {
                'coin': coin,
                'phi': float(phi),
                'beta': float(beta),
                'q': float(q),
                'r': float(r),
                'final_state': float(z_last),
                'final_variance': float(p_last),
                'smoothed_states': smoothed_z.tolist(),
                'smoothed_variances': smoothed_p.tolist(),
                'feature_columns': features.columns.tolist(),
                'validation_metrics': metrics,
                'training_samples': len(vol_series),
            }
            
            return model_result
        
        except Exception as e:
            logger.warning(f"Failed to train model for {coin}: {e}")
            return None

    def _validate_models(self):
        """Validate trained models and compute aggregate metrics"""
        if not self.trained_models:
            return
        
        for coin, model in self.trained_models.items():
            metrics = model['validation_metrics']
            self.validation_metrics[coin] = metrics
        
        # Compute aggregate metrics
        agg_metrics = {
            'RMSE': {'mean': 0, 'std': 0, 'min': 0, 'max': 0},
            'MAE': {'mean': 0, 'std': 0, 'min': 0, 'max': 0},
            'R2': {'mean': 0, 'std': 0, 'min': 0, 'max': 0},
            'MAPE': {'mean': 0, 'std': 0, 'min': 0, 'max': 0},
        }
        
        for metric_name in ['RMSE', 'MAE', 'R2', 'MAPE']:
            values = [m[metric_name] for m in self.validation_metrics.values() if metric_name in m]
            if values:
                agg_metrics[metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                }
        
        self.validation_metrics['aggregate'] = agg_metrics

    def _save_models(self):
        """Save trained models to disk"""
        # Save each coin's model
        for coin, model in self.trained_models.items():
            model_path = MODEL_DIR / f"{coin}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        # Save all models together
        all_models_path = MODEL_DIR / "trained_models.pkl"
        with open(all_models_path, 'wb') as f:
            pickle.dump(self.trained_models, f)
        
        self.log_progress(f"Saved {len(self.trained_models)} models to {MODEL_DIR}")

    def _save_metadata(self):
        """Save training metadata"""
        # Save trained coin list
        coin_list_path = MODEL_DIR / "trained_coin_list.json"
        with open(coin_list_path, 'w') as f:
            json.dump(self.trained_coins, f, indent=2)
        
        # Save validation metrics
        metrics_path = MODEL_DIR / "validation_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.validation_metrics, f, indent=2)
        
        # Save coin aliases
        aliases_path = MODEL_DIR / "coin_aliases.json"
        with open(aliases_path, 'w') as f:
            json.dump(CRYPTO_ALIASES, f, indent=2)
        
        # Save training info
        training_info = {
            'timestamp': datetime.now().isoformat(),
            'coins_trained': len(self.trained_coins),
            'model_dir': str(MODEL_DIR),
            'sentiment_model': SENTIMENT_MODEL,
            'state_space_horizon': STATE_SPACE_HORIZON,
            'rolling_window': ROLLING_WINDOW,
        }
        
        info_path = MODEL_DIR / "training_info.json"
        with open(info_path, 'w') as f:
            json.dump(training_info, f, indent=2)
        
        self.log_progress("Saved metadata files")

    def _empty_results(self) -> Dict:
        """Return empty results structure"""
        return {
            'timestamp': datetime.now().isoformat(),
            'coins_trained': 0,
            'trained_coins': [],
            'validation_metrics': {},
            'progress_log': self.progress_log,
            'error': 'Training failed to complete',
        }


def main():
    """Main training function"""
    logger.info("=" * 60)
    logger.info("CRIS-DSSM Training Pipeline")
    logger.info("=" * 60)
    
    trainer = CRISDSSMTrainer()
    results = trainer.run()
    
    logger.info("=" * 60)
    logger.info("Training Summary")
    logger.info("=" * 60)
    logger.info(f"Coins trained: {results['coins_trained']}")
    logger.info(f"Trained coins: {results['trained_coins']}")
    
    if 'validation_metrics' in results and 'aggregate' in results['validation_metrics']:
        logger.info("Aggregate validation metrics:")
        for metric, values in results['validation_metrics']['aggregate'].items():
            logger.info(f"  {metric}: {values}")
    
    logger.info(f"Models saved to: {results.get('model_dir', 'N/A')}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
