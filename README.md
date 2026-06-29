# CRIS-DSSM: A Dynamic State Space Approach for Cryptocurrency Risk Estimation Using Financial Sentiment Signals

CRIS-DSSM is a research-oriented cryptocurrency risk estimation framework that integrates **Financial Natural Language Processing (FinBERT)** with a **Dynamic State Space Model (DSSM)** and **Kalman Filtering** to estimate and forecast cryptocurrency market risk from financial news and historical market data.

Unlike conventional sentiment analysis systems, CRIS-DSSM models the dynamic influence of financial news on the latent risk state of cryptocurrencies, providing interpretable multi-horizon risk forecasts with confidence estimation.

---

## Key Features

- Financial sentiment analysis using **FinBERT**
- Dynamic Sentiment Shock estimation
- Cryptocurrency detection and impact mapping
- Historical market feature engineering
- Dynamic State Space Model implementation
- Kalman Filter based hidden state estimation
- Multi-horizon risk forecasting (1, 7, and 30 days)
- Confidence interval estimation
- Interactive Streamlit dashboard
- Modular and object-oriented architecture
- Research-oriented implementation following the proposed methodology

---

## Research Workflow

```text
Financial News
      │
      ▼
Text Preprocessing
      │
      ▼
FinBERT Sentiment Analysis
      │
      ▼
Sentiment Shock Estimation
      │
      ▼
Cryptocurrency Detection
      │
      ▼
Market Feature Engineering
      │
      ▼
Dynamic State Space Model
      │
      ▼
Kalman Filter
      │
      ▼
Risk Estimation & Forecasting
      │
      ▼
Interactive Dashboard
```

---

## Repository Structure

```text
crypto_risk_system/
│
├── config.py
├── main.py
├── train.py
├── predict.py
├── pipeline.py
├── dashboard.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── dssm.py
│   ├── feature_engineering.py
│   ├── sentiment_analyzer.py
│   ├── trained/
│   │   ├── *.pkl
│   │   ├── trained_coin_list.json
│   │   ├── validation_metrics.json
│   │   ├── coin_aliases.json
│   │   └── feature_columns.json
│   └── __pycache__/
│
├── utils/
│   ├── data_loader.py
│   ├── metrics.py
│   ├── text_processor.py
│   └── __pycache__/
│
├── data/
│   ├── news/
│   │   └── cryptonews.csv
│   └── market/
│       └── Historical cryptocurrency datasets
│
└── __pycache__/
```

---

## Methodology

The CRIS-DSSM framework follows the complete pipeline below:

1. Load cryptocurrency news articles.
2. Preprocess financial text.
3. Generate financial sentiment using **FinBERT**.
4. Compute **Sentiment Shock** to capture unexpected market sentiment changes.
5. Detect affected cryptocurrencies using aliases and keyword matching.
6. Load historical market datasets.
7. Engineer market features such as returns, volatility, momentum, and moving averages.
8. Train a **Dynamic State Space Model** for each cryptocurrency.
9. Estimate hidden market risk using **Kalman Filtering**.
10. Forecast future cryptocurrency risk and confidence intervals.

---

## Mathematical Model

### State Equation

\[
Z_t = \phi Z_{t-1} + \beta S_t + \eta_t
\]

where:

- \(Z_t\): Hidden market risk state
- \(S_t\): Sentiment shock
- \(\phi\): State persistence
- \(\beta\): Sentiment impact coefficient

### Observation Equation

\[
Y_t = HZ_t + \epsilon_t
\]

where:

- \(Y_t\): Observed market volatility
- \(H\): Observation matrix
- \(\epsilon_t\): Measurement noise

---

## Dataset

### News Dataset

```
data/news/cryptonews.csv
```

Used fields:

- Date
- Title
- Text

The existing sentiment column is ignored and recomputed using FinBERT.

### Market Dataset

```
data/market/
```

Contains historical CSV files for multiple cryptocurrencies used for model training.

Each dataset includes historical market information such as Open, High, Low, Close, Volume, and Market Capitalization.

---

## Installation

Clone the repository

```bash
git clone https://github.com/your-username/CRIS-DSSM.git
cd crypto_risk_system
```

Create a virtual environment

```bash
python -m venv venv
```

Activate the environment

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Training

Train the complete CRIS-DSSM framework using the historical news and market datasets.

```bash
python train.py
```

The training pipeline:

- Loads news and market datasets
- Generates FinBERT sentiment
- Computes sentiment shock
- Creates cryptocurrency impact mappings
- Engineers market features
- Trains Dynamic State Space Models
- Estimates Kalman Filter parameters
- Validates the trained models
- Saves trained model artifacts into `models/trained/`

---

## Prediction

Run inference using the trained models.

```bash
python predict.py
```

Predictions are generated using the saved model parameters without retraining.

---

## Launch Dashboard

```bash
streamlit run dashboard.py
```

or

```bash
python main.py --dashboard
```

---

## Dashboard Outputs

The dashboard provides:

- FinBERT Sentiment Probabilities
- Sentiment Score
- Sentiment Shock
- Detected Cryptocurrencies
- Coin Impact Weights
- Current Risk Score
- Future Risk Forecast (1, 7, and 30 days)
- Risk Category
- Confidence Interval
- Prediction Confidence

---

## Visualizations

The dashboard includes research-oriented visualizations generated from model predictions:

- Hidden Risk State with Forecast
- Sentiment Shock Timeline
- Risk Contribution Analysis
- Coin Impact Network
- Prediction Confidence Forecast

---

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Statsmodels
- PyTorch
- Hugging Face Transformers
- Plotly
- Streamlit

---

## Performance Evaluation

The model is evaluated using:

- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- R² Score (Coefficient of Determination)

---

## Future Work

- Real-time cryptocurrency news integration
- Online model updating
- Cross-market risk propagation
- Multi-language financial news analysis
- Explainable AI for risk attribution

---
