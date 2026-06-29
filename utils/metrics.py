import logging
from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))


def r2(y_true, y_pred):
    return float(r2_score(y_true, y_pred))


"""
Evaluation Metrics Module
Compute regression and risk estimation metrics
"""

logger = logging.getLogger(__name__)


class RegressionMetrics:
    """Compute standard regression metrics"""

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error"""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error"""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Percentage Error"""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        # Avoid division by zero
        mask = y_true != 0
        if not np.any(mask):
            return 0.0
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

    @staticmethod
    def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """R-squared (coefficient of determination)"""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

        if ss_tot == 0:
            return 0.0

        return float(1 - (ss_res / ss_tot))

    @staticmethod
    def compute_all(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute all metrics"""
        return {
            'RMSE': RegressionMetrics.rmse(y_true, y_pred),
            'MAE': RegressionMetrics.mae(y_true, y_pred),
            'MAPE': RegressionMetrics.mape(y_true, y_pred),
            'R2': RegressionMetrics.r_squared(y_true, y_pred),
        }


class RiskMetrics:
    """Risk-specific metrics"""

    @staticmethod
    def sigmoid(x: float) -> float:
        """Sigmoid function to map risk score to [0, 1]"""
        x = float(x)
        return float(1.0 / (1.0 + np.exp(-x)))

    @staticmethod
    def risk_score_from_volatility(volatility: float) -> float:
        """Convert volatility to risk score [0, 1]"""
        # Sigmoid maps unbounded volatility to [0, 1]
        return RiskMetrics.sigmoid(volatility)

    @staticmethod
    def compute_confidence(variance: float) -> float:
        """
        Compute prediction confidence from variance
        High variance = low confidence
        Confidence = 1 - sqrt(variance), clamped to [0, 1]
        """
        variance = max(0, float(variance))
        confidence = max(0.0, min(1.0, 1.0 - np.sqrt(variance)))
        return float(confidence * 100)  # Return as percentage


class RiskRanker:
    """Rank cryptocurrencies by risk"""

    RISK_CATEGORIES = {
        "Very High": (0.75, 1.0),
        "High": (0.5, 0.75),
        "Moderate": (0.25, 0.5),
        "Low": (0.0, 0.25),
    }

    @staticmethod
    def categorize_risk(score: float) -> str:
        """Categorize risk score into risk level"""
        score = float(score)
        for category, (lower, upper) in RiskRanker.RISK_CATEGORIES.items():
            if lower <= score <= upper:
                return category
        return "Unknown"

    @staticmethod
    def rank_coins(results: list) -> Tuple[list, list]:
        """
        Rank coins by risk

        Args:
            results: List of dicts with 'coin' and 'current_risk' keys

        Returns:
            (highest_risk_coins, safest_coins): Tuples of ranked results
        """
        sorted_by_risk = sorted(
            results, key=lambda x: x['current_risk'], reverse=True)

        highest_risk = sorted_by_risk[:10]
        safest = sorted_by_risk[-10:][::-1]  # Reverse to show safest first

        return highest_risk, safest


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_ablation: np.ndarray = None
) -> Dict[str, float]:
    """
    Evaluate model predictions with optional ablation study
    """
    metrics = RegressionMetrics.compute_all(y_true, y_pred)

    # Ablation: predictive power of shocks vs baseline
    if y_pred_ablation is not None:
        ablation_metrics = RegressionMetrics.compute_all(
            y_true, y_pred_ablation)
        metrics['ablation_rmse'] = ablation_metrics['RMSE']
        metrics['ablation_mae'] = ablation_metrics['MAE']
        metrics['ablation_r2'] = ablation_metrics['R2']

        # Improvement from shocks
        if ablation_metrics['RMSE'] > 0:
            metrics['rmse_improvement'] = (
                (ablation_metrics['RMSE'] - metrics['RMSE']) /
                ablation_metrics['RMSE'] * 100
            )

    return metrics
