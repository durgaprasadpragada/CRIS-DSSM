"""
CRIS-DSSM Streamlit Dashboard
Cryptocurrency Risk Insight System using Dynamic State Space Modeling
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
from datetime import datetime
import logging
import numpy as np

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
    initial_sidebar_state="expanded",
    menu_items={}
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
    /* Hide Streamlit footer */
    footer {visibility: hidden;}
    footer:before {
        content: 'CRIS-DSSM Research System';
        visibility: visible;
        display: block;
        position: relative;
        color: #ccc;
        font-size: 0.8em;
        padding: 5px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


def display_progress_log(log_messages):
    """Display progress log messages"""
    if log_messages:
        with st.expander("📝 Analysis Log (Debug)", expanded=False):
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


def create_hidden_risk_forecast_chart(coin_result, historical_states=None):
    """
    Create historical hidden risk + forecast chart
    
    Shows:
    - Historical latent risk (from trained model)
    - Today's estimated risk
    - Tomorrow forecast
    - 7-Day forecast
    - 30-Day forecast
    - Confidence intervals
    """
    if not coin_result:
        return go.Figure()
    
    coin = coin_result['coin']
    current_risk = coin_result['current_risk']
    future_risks = coin_result.get('future_risks', {})
    forecast_variances = coin_result.get('forecast_variances', {})
    
    # Use historical states if available
    if historical_states is not None and len(historical_states) > 0:
        # Take last 30 historical points
        hist_states = historical_states[-30:] if len(historical_states) > 30 else historical_states
        hist_x = list(range(len(hist_states)))
        
        fig = go.Figure()
        
        # Historical states
        fig.add_trace(go.Scatter(
            x=hist_x,
            y=hist_states,
            mode='lines',
            name='Historical Hidden Risk',
            line=dict(color='#ccc', width=2),
            opacity=0.7
        ))
        
        # Current point
        current_x = len(hist_states)
        fig.add_trace(go.Scatter(
            x=[current_x],
            y=[current_risk],
            mode='markers',
            name='Current Risk',
            marker=dict(color='#667eea', size=15, symbol='diamond')
        ))
        
        # Forecasts - safely handle both string and integer keys
        horizon_keys = []
        for k in future_risks.keys():
            if isinstance(k, str) and k.startswith('t+'):
                horizon_keys.append(int(k.replace('t+', '')))
            elif isinstance(k, int):
                horizon_keys.append(k)
            else:
                # Try to convert to int
                try:
                    horizon_keys.append(int(k))
                except:
                    pass
        horizon_keys = sorted(horizon_keys)
        
        forecast_x = [current_x + h for h in horizon_keys]
        forecast_y = [future_risks.get(f't+{h}') if f't+{h}' in future_risks else future_risks.get(h) for h in horizon_keys]
        
        fig.add_trace(go.Scatter(
            x=[current_x] + forecast_x,
            y=[current_risk] + forecast_y,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#d32f2f', width=3, dash='dash'),
            marker=dict(size=10)
        ))
        
        # Confidence intervals
        upper_bounds = [current_risk]
        lower_bounds = [current_risk]
        for i, h in enumerate(horizon_keys):
            var_key = f't+{h}' if f't+{h}' in forecast_variances else h
            var = forecast_variances.get(var_key, 0.01)
            upper_bounds.append(forecast_y[i] + np.sqrt(var) * 1.96)
            lower_bounds.append(forecast_y[i] - np.sqrt(var) * 1.96)
        
        fig.add_trace(go.Scatter(
            x=[current_x] + forecast_x + forecast_x[::-1] + [current_x],
            y=upper_bounds + lower_bounds[::-1],
            fill='toself',
            fillcolor='rgba(211, 47, 47, 0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% Confidence Interval',
            hoverinfo="skip"
        ))
        
        fig.update_layout(
            title=f"Hidden Risk State & Forecast: {coin}",
            xaxis_title="Time (Days)",
            yaxis_title="Hidden State (Latent Risk)",
            height=400,
            hovermode='x unified',
            legend=dict(x=0.7, y=0.95)
        )
    else:
        # Fallback without historical data
        horizon_keys = []
        for k in future_risks.keys():
            if isinstance(k, str) and k.startswith('t+'):
                horizon_keys.append(int(k.replace('t+', '')))
            elif isinstance(k, int):
                horizon_keys.append(k)
            else:
                try:
                    horizon_keys.append(int(k))
                except:
                    pass
        horizon_keys = sorted(horizon_keys)
        
        horizons = ['Today'] + [f'{h} Days' for h in horizon_keys]
        risks = [current_risk] + [future_risks.get(f't+{h}') if f't+{h}' in future_risks else future_risks.get(h) for h in horizon_keys]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=horizons,
            y=risks,
            mode='lines+markers',
            name='Risk Forecast',
            line=dict(color='#667eea', width=3),
            marker=dict(size=10)
        ))
        
        fig.update_layout(
            title=f"Risk Forecast: {coin}",
            xaxis_title="Time Horizon",
            yaxis_title="Risk Score",
            height=400
        )
    
    return fig


def create_sentiment_shock_timeline(shock_value, historical_shocks=None):
    """
    Create sentiment shock timeline
    
    Shows:
    - Historical shock (if available)
    - Current shock
    - Highlights anomalies
    """
    fig = go.Figure()
    
    # If we have historical shocks from training, show them
    if historical_shocks is not None and len(historical_shocks) > 0:
        # Take last 30 historical points
        hist_shocks = historical_shocks[-30:] if len(historical_shocks) > 30 else historical_shocks
        hist_x = list(range(len(hist_shocks)))
        
        fig.add_trace(go.Scatter(
            x=hist_x,
            y=hist_shocks,
            mode='lines',
            name='Historical Shock',
            line=dict(color='#667eea', width=2),
            opacity=0.6
        ))
        
        # Current shock
        current_x = len(hist_shocks)
        fig.add_trace(go.Scatter(
            x=[current_x],
            y=[shock_value],
            mode='markers',
            name='Current Shock',
            marker=dict(color='#d32f2f', size=20, symbol='diamond'),
            text=[f'Shock: {shock_value:.4f}'],
            textposition='top center'
        ))
        
        # Zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        # Highlight anomalies (shocks beyond 2 std dev)
        if len(hist_shocks) > 0:
            mean_shock = np.mean(hist_shocks)
            std_shock = np.std(hist_shocks)
            upper_bound = mean_shock + 2 * std_shock
            lower_bound = mean_shock - 2 * std_shock
            
            fig.add_hline(y=upper_bound, line_dash="dot", line_color="orange", opacity=0.5, annotation_text="+2σ")
            fig.add_hline(y=lower_bound, line_dash="dot", line_color="orange", opacity=0.5, annotation_text="-2σ")
        
        fig.update_layout(
            title="Sentiment Shock Timeline",
            xaxis_title="Time (Days)",
            yaxis_title="Shock Value",
            height=400,
            showlegend=True
        )
    else:
        # Fallback without historical data
        fig.add_trace(go.Scatter(
            x=['Current'],
            y=[shock_value],
            mode='markers',
            name='Current Shock',
            marker=dict(color='#d32f2f', size=30),
            text=[f'Shock: {shock_value:.4f}'],
            textposition='top center'
        ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        fig.update_layout(
            title="Current Sentiment Shock",
            xaxis_title="",
            yaxis_title="Shock Value",
            height=300,
            showlegend=False
        )
    
    return fig


def create_risk_contribution_plot(coin_result):
    """
    Create risk contribution plot
    
    Shows how the final prediction was formed:
    - Previous Hidden Risk
    - Sentiment Contribution
    - Shock Contribution
    - Market Contribution
    = Final Risk
    """
    if not coin_result:
        return go.Figure()
    
    coin = coin_result['coin']
    previous_state = coin_result.get('previous_state', 0)
    current_state = coin_result.get('hidden_state', 0)
    shock_applied = coin_result.get('shock_applied', 0)
    state_change = coin_result.get('state_change', 0)
    
    # Decompose the change
    phi = coin_result['model_params'].get('phi', 0.9)
    beta = coin_result['model_params'].get('beta', 0.1)
    
    # Contributions
    persistence_contribution = phi * previous_state
    shock_contribution = beta * shock_applied
    residual = current_state - persistence_contribution - shock_contribution
    
    categories = ['Previous State (φ×Z)', 'Shock Effect (β×S)', 'Residual', 'Final State']
    values = [persistence_contribution, shock_contribution, residual, current_state]
    colors = ['#667eea', '#f57c00', '#ccc', '#d32f2f']
    
    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=values,
            marker_color=colors,
            text=[f"{v:.4f}" for v in values],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title=f"Risk Decomposition: {coin}",
        xaxis_title="Component",
        yaxis_title="Contribution to Hidden State",
        height=400,
        showlegend=False
    )
    
    return fig


def create_coin_impact_network(results, affected_coins):
    """
    Create coin impact network graph
    
    Shows only affected cryptocurrencies:
    - Nodes: Coins
    - Edges: Impact relationships
    - Edge width: Impact weight
    """
    if not results:
        return go.Figure()
    
    # Create network graph
    G = nx.Graph()
    
    # Add nodes for affected coins
    for coin_result in results:
        coin = coin_result['coin']
        risk = coin_result['current_risk']
        G.add_node(coin, risk=risk)
    
    # Add edges based on impact weights (simplified: connect if both affected)
    if len(results) > 1:
        for i in range(len(results)):
            for j in range(i+1, len(results)):
                coin1 = results[i]['coin']
                coin2 = results[j]['coin']
                # Edge weight based on risk similarity
                risk_diff = abs(results[i]['current_risk'] - results[j]['current_risk'])
                weight = max(0.1, 1.0 - risk_diff)
                G.add_edge(coin1, coin2, weight=weight)
    
    # Get positions
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Create node traces
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Get risk for color
        node_data = next((r for r in results if r['coin'] == node), None)
        risk = node_data['current_risk'] if node_data else 0.5
        node_colors.append(risk)
        
        node_text.append(f"{node}<br>Risk: {risk:.3f}")
    
    # Create edge traces
    edge_x = []
    edge_y = []
    edge_widths = []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_widths.append(edge[2]['weight'] * 5)
    
    fig = go.Figure()
    
    # Edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=2, color='#ccc'),
        hoverinfo='none',
        showlegend=False
    ))
    
    # Nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(
            size=30,
            color=node_colors,
            colorscale='RdYlGn_r',
            cmin=0,
            cmax=1,
            colorbar=dict(title="Risk Score"),
            line=dict(width=2, color='white')
        ),
        text=[node.split('<br>')[0] for node in node_text],
        textposition='middle center',
        hovertext=node_text,
        hoverinfo='text',
        showlegend=False
    ))
    
    fig.update_layout(
        title=f"Coin Impact Network ({len(results)} Affected Coins)",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500,
        margin=dict(b=20, l=20, r=20, t=60)
    )
    
    return fig


def create_prediction_confidence_chart(coin_result):
    """
    Create prediction confidence chart
    
    Displays:
    - Confidence Interval
    - Prediction Variance
    - Kalman Covariance
    - Forecast uncertainty
    """
    if not coin_result:
        return go.Figure()
    
    coin = coin_result['coin']
    variance = coin_result.get('variance', 0)
    confidence = coin_result.get('confidence', 0)
    forecast_variances = coin_result.get('forecast_variances', {})
    
    # Convert to readable time horizons
    horizon_keys = []
    for k in forecast_variances.keys():
        if isinstance(k, str) and k.startswith('t+'):
            horizon_keys.append(int(k.replace('t+', '')))
        elif isinstance(k, int):
            horizon_keys.append(k)
        else:
            try:
                horizon_keys.append(int(k))
            except:
                pass
    horizon_keys = sorted(horizon_keys)
    
    horizon_labels = ['Today'] + [f'{h} Days' for h in horizon_keys]
    variances = [variance] + [forecast_variances.get(f't+{h}') if f't+{h}' in forecast_variances else forecast_variances.get(h) for h in horizon_keys]
    
    # Confidence decreases with time
    confidences = [confidence] * len(variances)
    for i in range(1, len(confidences)):
        confidences[i] = max(0, confidence - i * 0.05)
    
    fig = go.Figure()
    
    # Variance (secondary y-axis)
    fig.add_trace(go.Scatter(
        x=horizon_labels,
        y=variances,
        mode='lines+markers',
        name='Prediction Variance',
        yaxis='y',
        line=dict(color='#667eea', width=3),
        marker=dict(size=10)
    ))
    
    # Confidence (primary y-axis)
    fig.add_trace(go.Scatter(
        x=horizon_labels,
        y=confidences,
        mode='lines+markers',
        name='Prediction Confidence (%)',
        yaxis='y2',
        line=dict(color='#388e3c', width=3),
        marker=dict(size=10)
    ))
    
    fig.update_layout(
        title=f"Prediction Confidence & Uncertainty: {coin}",
        xaxis_title="Time Horizon",
        yaxis=dict(title="Variance", side="left", showgrid=True),
        yaxis2=dict(title="Confidence (%)", side="right", overlaying="y", range=[0, 100], showgrid=False),
        height=400,
        hovermode='x unified',
        legend=dict(x=0.7, y=0.95)
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
            "Enter one or more cryptocurrency news articles...",
            height=150,
            placeholder="Example: Bitcoin ETF approval delayed by SEC regulatory concerns..."
        )
    else:
        st.subheader("🪙 Select Cryptocurrency")
        try:
            predictor = create_predictor()
            coin_names = predictor.trained_coins
            
            if not coin_names:
                st.error("No trained models found. Please run train.py first.")
                return
            
            selected_coin = st.selectbox("Choose a cryptocurrency", coin_names)
            
            # Show that we're using the trained model for this specific coin
            st.info(f"Will use trained DSSM model for {selected_coin} with its specific parameters (φ, β, q, r) and historical states.")
            
            # Create an article that mentions the selected coin using various aliases
            # This ensures the coin detector will find it
            articles_text = f"Breaking news about {selected_coin}. The {selected_coin} cryptocurrency has been experiencing significant market activity."
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
            # For cryptocurrency selection, use the generated article
            articles = [articles_text]
        
        if not articles:
            st.error("Please enter at least one article.")
            return
        
        try:
            # Initialize predictor with trained models
            predictor = create_predictor()
            
            # Run prediction
            with st.spinner("Running prediction..."):
                results = predictor.predict(articles)
            
            # Display progress log
            display_progress_log(results.get('progress_log', []))
            
            # Check if results exist
            if not results.get('results'):
                st.error("No coins were successfully analyzed.")
                return
            
            # Display results
            results_list = results.get('results', [])
            affected_coins = results.get('affected_coins', [])
            
            # Get predictor for historical states (load once)
            predictor = create_predictor()
            
            # Model Validation Metrics (per-coin)
            st.subheader("📊 Model Validation Metrics")
            
            # Show aggregate metrics
            metrics = results.get('validation_metrics', {})
            aggregate = metrics.get('aggregate', {})
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                create_metric_card(
                    f"{aggregate.get('RMSE', {}).get('mean', 'N/A'):.4f}",
                    "RMSE",
                    "Aggregate Model Validation"
                )
            with col2:
                create_metric_card(
                    f"{aggregate.get('MAE', {}).get('mean', 'N/A'):.4f}",
                    "MAE",
                    "Aggregate Model Validation"
                )
            with col3:
                create_metric_card(
                    f"{aggregate.get('R2', {}).get('mean', 'N/A'):.4f}",
                    "R²",
                    "Aggregate Model Validation"
                )
            with col4:
                create_metric_card(
                    f"{aggregate.get('MAPE', {}).get('mean', 'N/A'):.2f}%",
                    "MAPE",
                    "Aggregate Model Validation"
                )
            
            # Affected Coins Summary
            st.subheader("🪙 Affected Cryptocurrencies")
            st.write(f"Detected {len(affected_coins)} affected coins from the article(s):")
            st.write(", ".join(affected_coins))
            
            # Detailed results for each affected coin
            for coin_result in results_list:
                coin = coin_result['coin']
                
                # Load historical states from trained model
                trained_model = predictor.trained_models.get(coin, {})
                historical_states = trained_model.get('smoothed_states', [])
                
                with st.expander(f"📈 {coin} - Detailed Analysis", expanded=True):
                    # Per-coin validation metrics
                    coin_metrics = coin_result.get('validation_metrics', {})
                    if coin_metrics:
                        st.markdown(f"**{coin} Model Validation Metrics:**")
                        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                        with mcol1:
                            st.metric("RMSE", f"{coin_metrics.get('RMSE', 'N/A'):.4f}")
                        with mcol2:
                            st.metric("MAE", f"{coin_metrics.get('MAE', 'N/A'):.4f}")
                        with mcol3:
                            st.metric("R²", f"{coin_metrics.get('R2', 'N/A'):.4f}")
                        with mcol4:
                            st.metric("MAPE", f"{coin_metrics.get('MAPE', 'N/A'):.2f}%")
                    
                    # Key metrics for this coin
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        create_metric_card(
                            f"{coin_result['current_risk']:.4f}",
                            "Current Risk",
                            coin_result['risk_category']
                        )
                    
                    with col2:
                        create_metric_card(
                            f"{coin_result['confidence']:.1f}%",
                            "Confidence",
                            "Prediction"
                        )
                    
                    with col3:
                        create_metric_card(
                            f"{coin_result.get('shock_applied', 0):.4f}",
                            "Shock Applied",
                            "Sentiment"
                        )
                    
                    # Forecast values with readable names
                    st.markdown("### Risk Forecast")
                    future_risks = coin_result.get('future_risks', {})
                    forecast_labels = {
                        't+1': 'Tomorrow',
                        't+7': '7 Days',
                        't+30': '30 Days'
                    }
                    forecast_cols = st.columns(len(future_risks))
                    for i, (horizon, risk) in enumerate(future_risks.items()):
                        with forecast_cols[i]:
                            label = forecast_labels.get(horizon, horizon)
                            st.metric(label, f"{risk:.4f}")
                    
                    # Visualizations for this coin
                    st.markdown("### Visualizations")
                    
                    # 1. Hidden Risk + Forecast (with historical states)
                    st.plotly_chart(
                        create_hidden_risk_forecast_chart(coin_result, historical_states),
                        use_container_width=True
                    )
                    
                    # 2. Sentiment Shock Timeline
                    st.plotly_chart(
                        create_sentiment_shock_timeline(coin_result.get('shock_applied', 0)),
                        use_container_width=True
                    )
                    
                    # 3. Risk Contribution
                    st.plotly_chart(
                        create_risk_contribution_plot(coin_result),
                        use_container_width=True
                    )
                    
                    # 4. Prediction Confidence
                    st.plotly_chart(
                        create_prediction_confidence_chart(coin_result),
                        use_container_width=True
                    )
            
            # 5. Coin Impact Network (all affected coins together)
            if len(results_list) > 1:
                st.subheader("🔗 Coin Impact Network")
                st.plotly_chart(
                    create_coin_impact_network(results_list, affected_coins),
                    use_container_width=True
                )
            
            # Download results
            st.subheader("💾 Export Results")
            results_df = pd.DataFrame(results_list)
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"cris_dssm_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        except Exception as e:
            logger.exception("Error running prediction")
            st.error(f"Error: {str(e)}")


if __name__ == '__main__':
    main()
