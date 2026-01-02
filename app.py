import streamlit as st
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Page Config
st.set_page_config(page_title="MarketIQ", layout="wide")
st.title("MarketIQ - Relative Value Scanner")

# Sidebar
st.sidebar.header("Configuration")

# Market Universes
UNIVERSES = {
    "Custom": "",
    "Tech Giants": "AAPL, MSFT, GOOGL, NVDA, TSLA",
    "Crypto Majors": "BTC-USD, ETH-USD, SOL-USD",
    "Safe Havens": "SPY, GLD, TLT, UUP",
    "Semiconductors": "NVDA, AMD, INTC, TSM"
}

def update_tickers_from_universe():
    selected = st.session_state.universe_selector
    if selected != "Custom":
        st.session_state.tickers_input = UNIVERSES[selected]

# Universe Selector
st.sidebar.selectbox(
    "Market Universe", 
    options=list(UNIVERSES.keys()), 
    key="universe_selector",
    on_change=update_tickers_from_universe
)

default_tickers = "AAPL, MSFT, SPY, BTC-USD"
# Initialize session state for tickers_input if not exists (though widget handles it, setting default helps)
if "tickers_input" not in st.session_state:
    st.session_state.tickers_input = default_tickers

tickers_input = st.sidebar.text_input("Tickers (comma-separated)", key="tickers_input")
timeframe = st.sidebar.select_slider(
    "Timeframe",
    options=["1mo", "3mo", "1y", "5y"],
    value="1y"
)
normalize = st.sidebar.toggle("Normalize Prices (Base 100)", value=True)

# Data Fetching
@st.cache_data(ttl=300)
def fetch_data(tickers, period):
    tickers_list = [t.strip() for t in tickers.split(",")]
    if not tickers_list:
        return pd.DataFrame()
    
    try:
        # Robust fetching: don't group by ticker immediately to handle single/multi consistency better if needed, 
        # but the user specific request implies we should use [Close] or be careful.
        # Let's trust the "Robust Manual Extraction" strategy which works for both structures.
        data = yf.download(tickers_list, period=period, group_by='ticker', auto_adjust=True)
        
        close_data = pd.DataFrame()
        for t in tickers_list:
            # Accessing MultiIndex properly
            if len(tickers_list) > 1:
                # Expecting (Ticker, OHLCV)
                if t in data.columns.levels[0]:
                    close_data[t] = data[t]['Close']
            else:
                # Single ticker: yfinance might return flat columns (Open, High, Low, Close...) 
                # OR (Ticker, Close) depending on version/args.
                # Check for 'Close' directly first (flat)
                if 'Close' in data.columns:
                    close_data[t] = data['Close']
                # Check if it has the ticker as a column (less likely with group_by='ticker' but possible)
                elif t in data.columns and 'Close' in data[t].columns:
                    close_data[t] = data[t]['Close']
                    
        return close_data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def run_monte_carlo(df, n_sims=1000, n_days=252):
    # Calculate log returns for better GBM properties
    returns = np.log(df / df.shift(1)).dropna()
    mean_returns = returns.mean().values
    cov_matrix = returns.cov().values
    
    # Ridge Adjustment for stability
    cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-8
    
    try:
        L = np.linalg.cholesky(cov_matrix)
    except np.linalg.LinAlgError:
        return None 
        
    # Generate Correlated Shocks
    # Z dims: (n_days, n_assets, n_sims)
    Z = np.random.standard_normal((n_days, len(df.columns), n_sims))
    
    simulation_results = {}
    
    for i, ticker in enumerate(df.columns):
        # Initializing paths: (Days+1, Sims)
        paths = np.zeros((n_days + 1, n_sims))
        paths[0] = df[ticker].iloc[-1]
        
        # GBM Parameters
        mu = mean_returns[i]
        sigma = returns[ticker].std()
        
        # Apply Cholesky for correlation: Z_corr = L * Z
        # This is a dot product for each day/sim
        # We simplify for single-ticker path generation in this loop
        for d in range(1, n_days + 1):
            # Correlated random shock for this asset
            # L[i, :] @ Z[d-1, :, s]
            shock = np.dot(L[i, :], Z[d-1, :, :]) 
            # GBM Formula: S_t = S_{t-1} * exp(drift + shock)
            paths[d] = paths[d-1] * np.exp((mu - 0.5 * sigma**2) + shock)
            
        simulation_results[ticker] = paths
    
    return simulation_results

if tickers_input:
    with st.spinner("Fetching market data..."):
        try:
            df = fetch_data(tickers_input, timeframe)
            
            if not df.empty:
                # Handle missing values (e.g. weekends for stocks vs crypto)
                df = df.ffill()

                # Metrics Row
                st.subheader("Market Pulse")
                cols = st.columns(len(df.columns))
                for idx, col in enumerate(df.columns):
                    current_price = df[col].iloc[-1]
                    start_price = df[col].iloc[0]
                    delta = ((current_price - start_price) / start_price) * 100
                    
                    with cols[min(idx, len(cols)-1)]:
                        st.metric(
                            label=col,
                            value=f"${current_price:,.2f}",
                            delta=f"{delta:+.2f}%"
                        )
                
                # Charting
                st.subheader("Performance Chart")
                plot_df = df.copy()
                
                if normalize:
                    # Rebase to 100
                    plot_df = (plot_df / plot_df.iloc[0]) * 100
                    y_axis_title = "Normalized Price (Base 100)"
                else:
                    y_axis_title = "Price ($)"
                
                fig = px.line(plot_df, x=plot_df.index, y=plot_df.columns, title=f"Relative Performance ({timeframe})")
                fig.update_layout(yaxis_title=y_axis_title, xaxis_title="Date", hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)

                # --- Risk Analysis Section ---
                st.subheader("Risk Analysis")
                tab1, tab2, tab3 = st.tabs(["Correlation", "Drawdowns", "Future Projections"])

                with tab1:
                    st.caption("How much do these assets move together?")
                    returns = df.pct_change().dropna()
                    if len(returns.columns) > 1:
                        corr = returns.corr()
                        fig_corr = px.imshow(
                            corr, text_auto=".2f",
                            color_continuous_scale='RdBu_r', 
                            color_continuous_midpoint=0,
                            zmin=-1, zmax=1
                        )
                        st.plotly_chart(fig_corr, use_container_width=True)

                with tab2:
                    st.caption("Underwater Plot (Distance from All-Time High)")
                    rolling_max = df.cummax()
                    drawdown = (df / rolling_max) - 1
                    
                    # Use Line chart instead of Area for clarity
                    fig_dd = px.line(drawdown, title="Asset Drawdowns")
                    
                    # Add a 'Danger Zone' shaded area manually
                    fig_dd.add_hrect(y0=-0.2, y1=0, fillcolor="red", opacity=0.05, line_width=0, annotation_text="Correction Zone")
                    fig_dd.add_hline(y=0, line_dash="dash", line_color="green")
                    
                    fig_dd.update_layout(yaxis_tickformat='.1%', yaxis_title="Drawdown %", hovermode="x unified")
                    st.plotly_chart(fig_dd, use_container_width=True)

                with tab3:
                    st.caption("Monte Carlo GBM Simulation (Correlated)")
                    
                    # Task 1: Multi-Asset Projections
                    selected_ticker = st.selectbox("Select Asset to Project", options=df.columns, index=0)
                    
                    if st.button("Generate 1,000 Paths"):
                        sim_results = run_monte_carlo(df)
                        if sim_results:
                            paths = sim_results[selected_ticker] # (Days, Sims)
                            
                            # Quantitative Metrics
                            start_p = paths[0,0]
                            final_prices = paths[-1, :]
                            exp_return = final_prices.mean()
                            prob_profit = (final_prices > start_p).mean()
                            returns_dist = (final_prices / start_p) - 1
                            var_95 = np.percentile(returns_dist, 5)
                            cvar_95 = returns_dist[returns_dist <= var_95].mean()
                            
                            # Metrics Row
                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric(f"Expected {selected_ticker}", f"${exp_return:,.2f}")
                            m2.metric("Prob. of Profit", f"{prob_profit:.1%}")
                            m3.metric("VaR (95%)", f"{var_95:.1%}")
                            m4.metric("CVaR (Tail Risk)", f"{cvar_95:.1%}")
                            
                            # Task 2: Enhanced Projection Chart
                            # Calculate percentiles across time
                            # paths shape: (n_days, n_sims)
                            median_path = np.median(paths, axis=1)
                            p05_path = np.percentile(paths, 5, axis=1)
                            p95_path = np.percentile(paths, 95, axis=1)
                            x_axis = list(range(len(median_path)))
                            
                            # Main Chart with Ribbon
                            fig_sim = make_subplots(
                                rows=1, cols=2, 
                                column_widths=[0.8, 0.2], 
                                shared_yaxes=True,
                                horizontal_spacing=0.02
                            )
                            
                            # Confidence Ribbon (Low)
                            fig_sim.add_trace(go.Scatter(
                                x=x_axis, y=p05_path, 
                                mode='lines', 
                                line=dict(color='rgba(255,0,0,0)'), # Invisible line
                                showlegend=False,
                                name='5th Percentile'
                            ), row=1, col=1)
                            
                            # Confidence Ribbon (High + Fill)
                            fig_sim.add_trace(go.Scatter(
                                x=x_axis, y=p95_path, 
                                mode='lines', 
                                line=dict(color='rgba(255,0,0,0)'), # Invisible line
                                fill='tonexty', 
                                fillcolor='rgba(0, 100, 255, 0.2)',
                                showlegend=True,
                                name='95% Confidence'
                            ), row=1, col=1)
                            
                            # Median Path
                            fig_sim.add_trace(go.Scatter(
                                x=x_axis, y=median_path, 
                                mode='lines', 
                                line=dict(color='gold', width=3), 
                                name='Median Projection'
                            ), row=1, col=1)
                            
                            # Marginal Histogram
                            fig_sim.add_trace(go.Histogram(
                                y=final_prices, 
                                orientation='h', 
                                showlegend=False,
                                marker_color='rgba(0, 100, 255, 0.5)',
                                name='Final Dist'
                            ), row=1, col=2)

                            fig_sim.update_layout(
                                title=f"Monte Carlo Projections: {selected_ticker} (252 Days)",
                                xaxis_title="Trading Days",
                                yaxis_title="Price ($)",
                                hovermode="x unified",
                                template="plotly_dark",
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                            )
                            st.plotly_chart(fig_sim, use_container_width=True)
                            
                        else:
                            st.error("Simulation failed (Covariance Matrix Error). Input data might be insufficient.")
                
                # Statistics Table
                st.subheader("Key Statistics")
                stats_data = []
                for col in df.columns:
                    # Metric calcs using returns series correctly
                    col_returns = returns[col]
                    
                    # Total Return
                    tot_ret = (df[col].iloc[-1] / df[col].iloc[0]) - 1
                    
                    # Annualized Volatility
                    vol = col_returns.std() * (252 ** 0.5)
                    
                    # Max Drawdown
                    max_dd = drawdown[col].min()
                    
                    # Sharpe Ratio
                    mean_ret = col_returns.mean()
                    std_dev = col_returns.std()
                    sharpe = (mean_ret / std_dev) * (252 ** 0.5) if std_dev != 0 else 0
                    
                    stats_data.append({
                        "Ticker": col,
                        "Total Return": tot_ret,
                        "Ann. Volatility": vol,
                        "Max Drawdown": max_dd,
                        "Sharpe Ratio": sharpe
                    })
                
                stats_df = pd.DataFrame(stats_data).set_index("Ticker")
                
                # Styling with Styler and Column Config
                st.dataframe(
                    stats_df.style
                    .format("{:.2%}", subset=["Total Return", "Ann. Volatility", "Max Drawdown"])
                    .format("{:.2f}", subset=["Sharpe Ratio"])
                    # Return, Sharpe, MaxDD: Higher (closer to +infinity) is Green, Lower (closer to -infinity) is Red.
                    # For MaxDD (-50% vs -5%): -5% is higher (-0.05 > -0.50). So -5% is Green, -50% is Red. 
                    # This matches standard RdYlGn.
                    .background_gradient(cmap="RdYlGn", subset=["Total Return", "Sharpe Ratio", "Max Drawdown"])
                    # Volatility: Lower is better (Green). So we reverse ratio.
                    .background_gradient(cmap="RdYlGn_r", subset=["Ann. Volatility"]),
                    column_config={
                        "Sharpe Ratio": st.column_config.NumberColumn(
                            "Sharpe Ratio",
                            help="Measure of risk-adjusted return. Higher is better.",
                            format="%.2f"
                        )
                    }
                )
                
                # Data Table
                st.subheader("Raw Data")
                with st.expander("View Data"):
                    st.dataframe(df)
                    csv = df.to_csv().encode('utf-8')
                    st.download_button(
                        "Download CSV",
                        csv,
                        "market_iq_data.csv",
                        "text/csv",
                        key='download-csv'
                    )
            else:
                st.error("No data found for the selected tickers. Please check symbol names.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    st.info("Please enter tickers in the sidebar.")
