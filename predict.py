"""
CRIS-DSSM Prediction Pipeline
Uses trained models for prediction - no training during inference
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

from config import SENTIMENT_MODEL, STATE_SPACE_HORIZON, ROLLING_WINDOW
from utils.text_processor import TextPreprocessor
from models.sentiment_analyzer import FinBERTSentimentAnalyzer, SentimentShockComputer, CoinDetector
from models.dssm import kalman_forecast
from utils.metrics import RiskMetrics, RiskRanker

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model directory
MODEL_DIR = Path(__file__).parent / "models" / "trained"


class CRISDSSMPredictor:
    """
    Prediction pipeline using trained models
    
    Steps:
    1. Load trained models
    2. Process user input (news article)
    3. Generate FinBERT sentiment
    4. Compute sentiment shock
    5. Detect affected trained coins
    6. Load trained DSSM parameters
    7. Kalman update
    8. Predict hidden risk
    9. Forecast future risk
    10. Generate visualizations
    """

    def __init__(self):
        """Initialize predictor with trained models"""
        logger.info("Initializing CRIS DSSM Predictor...")
        
        # Text processing
        self.text_preprocessor = TextPreprocessor()
        
        # Sentiment analysis
        self.sentiment_analyzer = FinBERTSentimentAnalyzer(SENTIMENT_MODEL)
        self.shock_computer = SentimentShockComputer(rolling_window=ROLLING_WINDOW)
        
        # Coin detection
        self.coin_detector = None  # Will load from trained models
        
        # Load trained models
        self.trained_models: Dict[str, Dict] = {}
        self.trained_coins: List[str] = []
        self.validation_metrics: Dict = {}
        self.coin_aliases: Dict = {}
        
        # Progress tracking
        self.progress_log: List[str] = []
        
        # Load models
        self._load_trained_models()

    def log_progress(self, message: str):
        """Log progress message"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        msg = f"[{timestamp}] {message}"
        self.progress_log.append(msg)
        logger.info(message)

    def _load_trained_models(self):
        """Load trained models from disk"""
        try:
            # Load trained coin list
            coin_list_path = MODEL_DIR / "trained_coin_list.json"
            if not coin_list_path.exists():
                raise FileNotFoundError("Trained models not found. Please run train.py first.")
            
            with open(coin_list_path, 'r') as f:
                self.trained_coins = json.load(f)
            
            self.log_progress(f"Loaded {len(self.trained_coins)} trained coins")
            
            # Load trained models
            models_path = MODEL_DIR / "trained_models.pkl"
            with open(models_path, 'rb') as f:
                self.trained_models = pickle.load(f)
            
            self.log_progress(f"Loaded {len(self.trained_models)} trained models")
            
            # Load validation metrics
            metrics_path = MODEL_DIR / "validation_metrics.json"
            with open(metrics_path, 'r') as f:
                self.validation_metrics = json.load(f)
            
            self.log_progress("Loaded validation metrics")
            
            # Load coin aliases
            aliases_path = MODEL_DIR / "coin_aliases.json"
            with open(aliases_path, 'r') as f:
                self.coin_aliases = json.load(f)
            
            self.log_progress("Loaded coin aliases")
            
            # Initialize coin detector with trained aliases
            self.coin_detector = CoinDetector(self.coin_aliases)
            
        except Exception as e:
            logger.error(f"Failed to load trained models: {e}")
            raise

    def predict(self, articles: List[str]) -> Dict:
        """
        Run prediction pipeline on articles
        
        Args:
            articles: List of news articles
            
        Returns:
            Dictionary with prediction results
        """
        self.progress_log = []
        self.log_progress("Starting CRIS DSSM Prediction...")
        
        try:
            # Step 1: Preprocess text
            self.log_progress("Preprocessing text...")
            cleaned = self.text_preprocessor.preprocess_batch(articles)
            
            # Step 2: Generate FinBERT sentiment
            self.log_progress("Running FinBERT sentiment analysis...")
            self.sentiment_analyzer.load()
            sentiment_df = self.sentiment_analyzer.analyze(cleaned)
            
            # Log sentiment results
            for idx, row in sentiment_df.iterrows():
                self.log_progress(f"  Article {idx+1}: Pos={row['positive']:.3f}, Neu={row['neutral']:.3f}, Neg={row['negative']:.3f}, Score={row['sentiment']:.3f}")
            
            # Step 3: Compute sentiment shock
            self.log_progress("Computing sentiment shock...")
            self.shock_computer.update_baseline(0.0)
            sentiment_df['shock'] = self.shock_computer.compute_shock(sentiment_df['sentiment'])
            
            for idx, row in sentiment_df.iterrows():
                self.log_progress(f"  Article {idx+1}: Shock={row['shock']:.3f}")
            
            # Step 4: Detect affected trained coins
            self.log_progress("Detecting affected cryptocurrencies...")
            affected_coins = self._detect_affected_coins(sentiment_df)
            
            if not affected_coins:
                self.log_progress("WARNING: No trained coins detected in articles")
                return self._empty_results()
            
            self.log_progress(f"Affected coins: {affected_coins}")
            
            # Step 5: Load trained parameters and predict
            self.log_progress("Loading trained parameters and predicting...")
            results = []
            
            for coin in affected_coins:
                if coin not in self.trained_models:
                    self.log_progress(f"  {coin}: Not in trained models, skipping")
                    continue
                
                try:
                    result = self._predict_coin(coin, sentiment_df)
                    if result:
                        results.append(result)
                        self.log_progress(f"  {coin}: Predicted successfully")
                except Exception as e:
                    logger.warning(f"Error predicting {coin}: {e}")
                    continue
            
            if not results:
                self.log_progress("ERROR: No predictions generated")
                return self._empty_results()
            
            # Step 6: Rank coins
            self.log_progress("Ranking cryptocurrencies by risk...")
            highest_risk, safest = RiskRanker.rank_coins(results)
            
            # Step 7: Compile results with per-coin validation metrics
            self.log_progress("Prediction complete!")
            
            # Add per-coin validation metrics to results
            for result in results:
                coin = result['coin']
                if coin in self.validation_metrics:
                    result['validation_metrics'] = self.validation_metrics[coin]
            
            return {
                'timestamp': datetime.now().isoformat(),
                'articles_analyzed': len(articles),
                'coins_predicted': len(results),
                'affected_coins': affected_coins,
                'results': results,
                'highest_risk_coins': highest_risk,
                'safest_coins': safest,
                'validation_metrics': self.validation_metrics,
                'progress_log': self.progress_log,
            }
        
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            self.log_progress(f"ERROR: {e}")
            return self._empty_results()

    def _detect_affected_coins(self, sentiment_df: pd.DataFrame) -> List[str]:
        """Detect which trained coins are affected by the articles"""
        detected_coins = set()
        
        for _, row in sentiment_df.iterrows():
            text = row['text']
            detected = self.coin_detector.detect(text)
            
            # Filter to only trained coins
            for coin in detected:
                if coin in self.trained_coins:
                    detected_coins.add(coin)
        
        # If no coins detected, return empty list (will show error)
        if not detected_coins:
            return []
        
        return sorted(list(detected_coins))

    def _predict_coin(self, coin: str, sentiment_df: pd.DataFrame) -> Optional[Dict]:
        """Predict risk for a single coin using trained model"""
        self.log_progress(f"    === Predicting {coin} ===")
        
        # Load trained model parameters
        model = self.trained_models[coin]
        
        phi = model['phi']
        beta = model['beta']
        q = model['q']
        r = model['r']
        z_last = model['final_state']
        p_last = model['final_variance']
        
        self.log_progress(f"    Previous Hidden State: {z_last:.6f}")
        self.log_progress(f"    Previous Variance: {p_last:.6f}")
        self.log_progress(f"    Model Params - φ={phi:.6f}, β={beta:.6f}, q={q:.6f}, r={r:.6f}")
        
        # Compute aggregate shock from articles
        total_shock = 0.0
        total_weight = 0.0
        
        for idx, row in sentiment_df.iterrows():
            text = row['text']
            shock = row['shock']
            pos = row['positive']
            neu = row['neutral']
            neg = row['negative']
            sent = row['sentiment']
            
            self.log_progress(f"    Article {idx+1}: Pos={pos:.3f}, Neu={neu:.3f}, Neg={neg:.3f}, Sent={sent:.3f}, Shock={shock:.3f}")
            
            # Detect if this coin is mentioned
            detected = self.coin_detector.detect(text)
            self.log_progress(f"    Detected coins in article: {detected}")
            
            if coin in detected:
                # Compute weight for this coin
                weights = self.coin_detector.weight_coins(
                    [coin], self.trained_coins, text
                )
                weight = weights.get(coin, 0.0)
                total_shock += weight * shock
                total_weight += weight
                self.log_progress(f"    {coin} weight: {weight:.4f}, contribution: {weight * shock:.4f}")
        
        # If no direct mention, use small shock
        if total_weight == 0:
            total_shock = sentiment_df['shock'].mean() * 0.1
            total_weight = 1.0
            self.log_progress(f"    No direct mention, using average shock: {total_shock:.4f}")
        
        avg_shock = total_shock / total_weight if total_weight > 0 else 0.0
        
        self.log_progress(f"    Total shock: {total_shock:.4f}, Avg shock: {avg_shock:.4f}")
        
        # Kalman update with new shock
        # Update hidden state: z_new = phi * z_old + beta * shock
        z_updated = phi * z_last + beta * avg_shock
        
        # Update variance: p_new = phi^2 * p_old + q
        p_updated = (phi ** 2) * p_last + q
        
        self.log_progress(f"    Updated Hidden State: {z_updated:.6f}")
        self.log_progress(f"    Updated Variance: {p_updated:.6f}")
        self.log_progress(f"    State Change: {z_updated - z_last:.6f}")
        
        # Forecast
        forecasts = kalman_forecast(
            z_updated, p_updated, phi, beta, q, STATE_SPACE_HORIZON
        )
        
        self.log_progress(f"    Forecasts: {[(f[0], f[1]) for f in forecasts]}")
        
        # Convert to risk scores
        current_risk = RiskMetrics.sigmoid(z_updated)
        future_risks = [RiskMetrics.sigmoid(f[0]) for f in forecasts]
        
        self.log_progress(f"    Current Risk: {current_risk:.4f}")
        self.log_progress(f"    Future Risks: {future_risks}")
        
        # Confidence
        confidence = RiskMetrics.compute_confidence(p_updated)
        self.log_progress(f"    Confidence: {confidence:.2f}%")
        
        # Forecast variances
        forecast_variances = {f't+{h}': float(forecasts[i][1])
                             for i, h in enumerate(STATE_SPACE_HORIZON)}
        
        # Compile result
        result = {
            'coin': coin,
            'current_risk': current_risk,
            'risk_category': RiskRanker.categorize_risk(current_risk),
            'future_risks': {f't+{h}': future_risks[i] for i, h in enumerate(STATE_SPACE_HORIZON)},
            'variance': float(p_updated),
            'confidence': confidence,
            'forecast_variances': forecast_variances,
            'model_params': {
                'phi': phi,
                'beta': beta,
                'q': q,
                'r': r,
            },
            'hidden_state': float(z_updated),
            'previous_state': float(z_last),
            'shock_applied': float(avg_shock),
            'state_change': float(z_updated - z_last),
        }
        
        self.log_progress(f"    === {coin} prediction complete ===")
        return result

    def _empty_results(self) -> Dict:
        """Return empty results structure"""
        return {
            'timestamp': datetime.now().isoformat(),
            'articles_analyzed': 0,
            'coins_predicted': 0,
            'affected_coins': [],
            'results': [],
            'highest_risk_coins': [],
            'safest_coins': [],
            'validation_metrics': {},
            'progress_log': self.progress_log,
            'error': 'Prediction failed to complete',
        }


def create_predictor() -> CRISDSSMPredictor:
    """Factory function"""
    return CRISDSSMPredictor()
