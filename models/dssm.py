"""
Dynamic State Space Model (DSSM) for Cryptocurrency Risk Estimation

Model Specification:
    State Equation:    Z_t = φ*Z_{t-1} + β*S_t + η_t    (η_t ~ N(0, Q))
    Observation:       vol_t = Z_t + ε_t                 (ε_t ~ N(0, R))

Where:
    Z_t = Unobserved risk state at time t
    S_t = Sentiment shock at time t
    φ = Persistence of risk (state transition coefficient)
    β = Impact of sentiment on risk (shock loading)
    Q = State noise variance
    R = Observation noise variance

The model captures how sentiment shocks affect cryptocurrency risk
dynamically over time, with persistence in the underlying risk state.
"""

import numpy as np
import logging
from typing import Tuple, List
from statsmodels.tsa.statespace.mlemodel import MLEModel

logger = logging.getLogger(__name__)


class DSSMStateSpaceModel(MLEModel):
    """
    Dynamic State Space Model for risk estimation

    Equation:
        Z_t = φ*Z_{t-1} + β*S_t + η_t
        vol_t = Z_t + ε_t

    This is a univariate state space model where the hidden state Z_t
    represents the true risk, which is driven by sentiment shocks S_t
    and follows an AR(1) process.
    """

    def __init__(self, endog: np.ndarray, exog_shocks: np.ndarray):
        """
        Initialize the state space model

        Args:
            endog: Observed volatility series
            exog_shocks: Sentiment shock series
        """
        super().__init__(
            endog,
            k_states=1,        # Single hidden state (risk)
            k_posdef=1,        # Single stochastic shock
            initialization='approximate_diffuse'
        )

        # Store shocks as exogenous variable
        self.exog_shocks = np.asarray(exog_shocks, dtype=float)
        if self.exog_shocks.ndim == 1:
            self.exog_shocks = self.exog_shocks.reshape(1, -1)

    def update(self, params: np.ndarray, **kwargs):
        """Update state space matrices with parameters"""
        phi, beta, log_q, log_r = params

        # Ensure parameters are float
        phi = float(np.clip(phi, -0.999, 0.999))  # Keep stable
        beta = float(beta)
        q = float(np.exp(log_q))
        r = float(np.exp(log_r))

        # State transition: Z_t = φ*Z_{t-1}
        self['transition'] = np.array([[phi]], dtype=float)

        # Selection: shock enters state
        self['selection'] = np.array([[1.0]], dtype=float)

        # State covariance: Var(η_t)
        self['state_cov'] = np.array([[q]], dtype=float)

        # Design: how state maps to observation
        self['design'] = np.array([[1.0]], dtype=float)

        # Observation covariance: Var(ε_t)
        self['obs_cov'] = np.array([[r]], dtype=float)

        # State intercept includes shock loading: β*S_t
        self['state_intercept'] = np.atleast_2d(beta * self.exog_shocks)

    def start_params(self) -> np.ndarray:
        """Reasonable starting parameters"""
        var = np.var(self.endog) if len(self.endog) > 1 else 1.0

        # φ: persistence (typical 0.8-0.95 for risk)
        phi_init = 0.9

        # β: shock loading (typical 0.1-0.5)
        beta_init = 0.1

        # Q, R: noise variances (from data variance) - use raw values, not log
        q_init = max(var * 0.01, 1e-6)
        r_init = max(var * 0.01, 1e-6)

        return np.array([phi_init, beta_init, q_init, r_init], dtype=float)

    def transform_params(self, params: np.ndarray) -> np.ndarray:
        """Transform parameters to unconstrained space"""
        params = np.array(params, dtype=float)
        # Log-transform variance parameters to ensure positivity
        params[2:] = np.log(np.maximum(params[2:], 1e-9))
        return params

    def untransform_params(self, params: np.ndarray) -> np.ndarray:
        """Transform parameters back to constrained space"""
        params = np.array(params, dtype=float)
        # Exp-transform back
        params[2:] = np.exp(params[2:])
        return params


def estimate_state_space(
    volatility_series: np.ndarray,
    shock_series: np.ndarray,
    maxiter: int = 200
) -> Tuple[float, float, float, float, np.ndarray, np.ndarray]:
    """
    Estimate the state space model parameters using maximum likelihood

    Args:
        volatility_series: Observed volatility (dependent variable)
        shock_series: Sentiment shocks (exogenous variable)
        maxiter: Maximum iterations for optimization

    Returns:
        phi: Persistence coefficient
        beta: Shock loading coefficient
        Q: State noise variance
        R: Observation noise variance
        smoothed_states: Smoothed hidden states Z_t
        smoothed_covariance: Smoothed state covariance
    """
    y = np.asarray(volatility_series, dtype=float)
    shocks = np.asarray(shock_series, dtype=float)

    if len(y) != len(shocks):
        raise ValueError("Volatility and shock series must have same length")

    if len(y) < 3:
        raise ValueError("Need at least 3 observations for estimation")

    try:
        # Build and fit model
        model = DSSMStateSpaceModel(y, shocks)
        results = model.fit(disp=False, maxiter=maxiter)

        # Extract parameters
        params = results.params
        phi = float(np.clip(params[0], -0.999, 0.999))
        beta = float(params[1])
        q = float(np.exp(params[2]))
        r = float(np.exp(params[3]))

        # Get filtered/smoothed states
        smoothed = results.smoothed_state[0]  # Shape: (n_obs,)
        smoothed_cov = results.smoothed_state_cov[0, 0, :]  # Shape: (n_obs,)

        logger.info(
            f"DSSM fitted: φ={phi:.4f}, β={beta:.4f}, Q={q:.6f}, R={r:.6f}")

        return phi, beta, q, r, smoothed, smoothed_cov

    except Exception as e:
        logger.warning(f"DSSM fitting failed: {e}. Using fallback.")
        # Fallback to simple parameters if fitting fails
        phi = 0.9
        beta = 0.1
        q = 1e-5
        r = 1e-5
        smoothed = np.full(len(y), np.mean(y))
        smoothed_cov = np.full(len(y), np.var(y))
        return phi, beta, q, r, smoothed, smoothed_cov


def kalman_forecast(
    z_last: float,
    p_last: float,
    phi: float,
    beta: float,
    q: float,
    horizons: List[int]
) -> List[Tuple[float, float]]:
    """
    Forecast hidden state using Kalman filter

    Args:
        z_last: Last filtered state estimate
        p_last: Last filtered state variance
        phi: State transition coefficient
        beta: Shock loading (not used in forecast, shocks are unknown)
        q: State noise variance
        horizons: Forecast horizons (in periods)

    Returns:
        List of (state_forecast, state_variance) tuples
    """
    forecasts = []
    z = float(z_last)
    p = float(p_last)

    for _ in horizons:
        # Predict next state (assuming no future shocks)
        z = phi * z
        p = phi * p * phi + q
        forecasts.append((float(z), float(p)))

    return forecasts
