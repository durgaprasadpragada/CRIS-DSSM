"""
CRIS-DSSM Streamlit Dashboard
Cryptocurrency Risk Insight System using Dynamic State Space Modeling
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import logging

from predict import create_predictor
from config import RISK_CATEGORIES

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="CRIS-DSSM",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .risk-high { color: #d32f2f; font-weight: bold; }
    .risk-moderate { color: #f57c00; font-weight: bold; }
    .risk-low { color: #388e3c; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


def display_progress_log(log_messages):
    """Display progress log messages"""
    if log_messages:
        st.subheader("📝 Analysis Log")
        for msg in log_messages:
            st.text(msg)


def create_metric_card(value, label, description):
    """Create a metric card"""
    st.markdown(f"""
    <div class="metric-card">
        <h4 style="color: #667eea; margin-bottom: 5px;">{value}</h4>
        <p style="margin-bottom: 3px; font-weight: bold;">{label}</p>
        <small style="color: #999;">{description}</small>
    </div>
    """, unsafe_allow_html=True)


def get_risk_color(risk_score):
    """Get color class based on risk score"""
    if risk_score >= 0.75:
        return "risk-high"
    elif risk_score >= 0.5:
        return "risk-moderate"
    else:
        return "risk-low"


def create_risk_distribution_chart(results):
    """Create risk distribution bar chart"""
    if not results:
        return go.Figure()
    
    coins = [r['coin'] for r in results]
    risks = [r['current_risk'] for r in results]
    
    fig = go.Figure(data=[
        go.Bar(
            x=coins,
            y=risks,
            marker=dict(
                color=risks,
                colorscale='RdYlGn_r',
                cmin=0,
                cmax=1,
                colorbar=dict(title="Risk Score"),
            )
        )
    ])
    
    fig.update_layout(
        title="Risk Distribution Across Cryptocurrencies",
        xaxis_title="Cryptocurrency",
        yaxis_title="Risk Score",
        hovermode='x unified',
        height=400,
        margin=dict(b=100),
    )
    
    return fig


def create_confidence_chart(results):
    """Create confidence bar chart"""
    if not results:
        return go.Figure()
    
    coins = [r['coin'] for r in results]
    confidence = [r.get('confidence', 0) for r in results]
    
    fig = go.Figure(data=[
        go.Bar(
            x=coins,
            y=confidence,
            marker=dict(color='#667eea', opacity=0.8)
        )
    ])
    
    fig.update_layout(
        title="Prediction Confidence",
        xaxis_title="Cryptocurrency",
        yaxis_title="Confidence (%)",
        hovermode='x unified',
        height=400,
        margin=dict(b=100),
    )
    
    return fig


def create_forecast_chart(coin_result):
    """Create forecast chart for a single coin"""
    if not coin_result:
        return go.Figure()
    
    coin = coin_result['coin']
    current_risk = coin_result['current_risk']
    future_risks = coin_result.get('future_risks', {})
    
    horizons = ['Current'] + [f't+{h}' for h in future_risks.keys()]
    risks = [current_risk] + list(future_risks.values())
    
    fig = go.Figure(data=[
        go.Scatter(
            x=horizons,
            y=risks,
            mode='lines+markers',
            name='Risk Forecast',
            line=dict(color='#667eea', width=3),
            marker=dict(size=10)
        )
    ])
    
    fig.update_layout(
        title=f"Risk Forecast: {coin}",
        xaxis_title="Time Horizon",
        yaxis_title="Risk Score",
        height=300,
    )
    
    return fig


def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="margin-bottom: 5px;">CRIS-DSSM</h1>
        <p style="margin-top: 0;">Cryptocurrency Risk Insight System using Dynamic State Space Modeling</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Input mode selection
    input_mode = st.sidebar.radio(
        "Select Input Mode",
        ["News Articles", "Cryptocurrency Selection"]
    )
    
    # News input
    if input_mode == "News Articles":
        st.subheader("📰 Enter Financial News")
        articles_text = st.text_area(
            "Enter one or more cryptocurrency news articles (one per line or separated by double-break)...",
            height=150,
            placeholder="Example: Bitcoin ETF approval delayed by SEC regulatory concerns..."
        )
    else:
        st.subheader("🪙 Select Cryptocurrency")
        # Load trained coins only
        try:
            predictor = create_predictor()
            coin_names = predictor.trained_coins
            
            if not coin_names:
                st.error("No trained models found. Please run train.py first.")
                return
            
            selected_coin = st.selectbox("Choose a cryptocurrency", coin_names)
            articles_text = f"Latest market analysis for {selected_coin}"
        except Exception as e:
            st.error(f"Error loading trained models: {e}")
            st.error("Please run train.py first to train the models.")
            return
    
    # Analyze button
    if st.button("🚀 Analyze", type="primary"):
        # Parse articles
        if input_mode == "News Articles":
            articles = [a.strip() for a in articles_text.split('\n\n') if a.strip()]
        else:
            articles = [articles_text]
        
        if not articles:
            st.error("Please enter at least one article.")
            return
        
        # Create progress placeholder
        progress_placeholder = st.empty()
        results_placeholder = st.empty()
        
        try:
            # Initialize predictor with trained models
            predictor = create_predictor()
            
            # Display progress
            progress_placeholder.info("Loading trained models...")
            
            # Run prediction
            results = predictor.predict(articles)
            
            # Display progress log
            display_progress_log(results.get('progress_log', []))
            
            # Check if results exist
            if not results.get('results'):
                results_placeholder.error("No coins were successfully analyzed.")
                return
            
            # Display results
            with results_placeholder.container():
                # Key Metrics
                st.subheader("📊 Key Metrics")
                col1, col2, col3, col4 = st.columns(4)
                
                metrics = results.get('validation_metrics', {})
                with col1:
                    create_metric_card(
                        f"{metrics.get('RMSE', {}).get('mean', 'N/A'):.4f}",
                        "RMSE",
                        "Model Validation RMSE"
                    )
                with col2:
                    create_metric_card(
                        f"{metrics.get('R2', {}).get('mean', 'N/A'):.4f}",
                        "R² Score",
                        "Model Validation R²"
                    )
                with col3:
                    create_metric_card(
                        str(results.get('coins_predicted', 0)),
                        "Coins",
                        "Predicted"
                    )
                with col4:
                    create_metric_card(
                        str(results.get('articles_analyzed', 0)),
                        "Articles",
                        "Processed"
                    )
                
                # Risk Rankings
                st.subheader("⚠️ Risk Rankings")
                col1, col2 = st.columns(2)
                
                top_risk = results.get('highest_risk_coins', [])[:10]
                safest = results.get('safest_coins', [])[:10]
                
                with col1:
                    st.markdown("### Highest Risk Coins")
                    for coin in top_risk:
                        risk_score = coin['current_risk']
                        risk_class = get_risk_color(risk_score)
                        st.markdown(f"""
                        <div style="padding: 10px; border-bottom: 1px solid #eee;">
                            <strong>{coin['coin']}</strong> 
                            <span class="{risk_class}">({coin.get('risk_category', 'Unknown')})</span><br/>
                            <small>Risk: {risk_score:.2%} | Confidence: {coin.get('confidence', 0):.0f}%</small>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("### Safest Coins")
                    for coin in safest:
                        risk_score = coin['current_risk']
                        risk_class = get_risk_color(risk_score)
                        st.markdown(f"""
                        <div style="padding: 10px; border-bottom: 1px solid #eee;">
                            <strong>{coin['coin']}</strong> 
                            <span class="{risk_class}">({coin.get('risk_category', 'Unknown')})</span><br/>
                            <small>Risk: {risk_score:.2%} | Confidence: {coin.get('confidence', 0):.0f}%</small>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Charts
                st.subheader("📈 Visualizations")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(
                        create_risk_distribution_chart(results.get('results', [])),
                        use_container_width=True
                    )
                
                with col2:
                    st.plotly_chart(
                        create_confidence_chart(results.get('results', [])),
                        use_container_width=True
                    )
                
                # Detailed Results Table
                st.subheader("📋 Detailed Results")
                results_df = pd.DataFrame(results.get('results', []))
                
                display_columns = ['coin', 'current_risk', 'risk_category', 'confidence', 'RMSE', 'R2']
                available_columns = [col for col in display_columns if col in results_df.columns]
                
                if available_columns:
                    st.dataframe(
                        results_df[available_columns].head(20),
                        use_container_width=True
                    )
                
                # Forecast Charts for Top 5 Coins
                st.subheader("🔮 Risk Forecasts")
                top_5_coins = results.get('highest_risk_coins', [])[:5]
                
                for coin_result in top_5_coins:
                    with st.expander(f"Forecast: {coin_result['coin']}"):
                        st.plotly_chart(
                            create_forecast_chart(coin_result),
                            use_container_width=True
                        )
                
                # Download Options
                st.subheader("💾 Export Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"cris_dssm_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    st.info("PDF report generation coming soon")
        
        except Exception as e:
            logger.exception("Error running pipeline")
            st.error(f"Error: {str(e)}")


if __name__ == '__main__':
    main()
