import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import time
import threading
from datetime import datetime
import websocket  # For real WebSocket integration (commented below)

# Streamlit page config
st.set_page_config(page_title="Live Order Flow Trading Analysis", layout="wide")

# Shared state for live data (using Streamlit session state)
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame(columns=['timestamp', 'price', 'volume_buy', 'volume_sell', 'delta'])
if 'stop_thread' not in st.session_state:
    st.session_state.stop_thread = False

def generate_order_flow_data(num_ticks=100):
    """
    Simulate order flow tick data. Replace with real API for live trading.
    Columns: timestamp, price, volume_buy, volume_sell, delta (buy - sell).
    """
    np.random.seed(42)  # For reproducibility
    timestamps = pd.date_range(start=datetime.now(), periods=num_ticks, freq='S')
    prices = 1.0850 + np.cumsum(np.random.normal(0, 0.0001, num_ticks))  # Random walk for EURUSD-like price
    volumes_buy = np.random.poisson(10, num_ticks)  # Buy volumes
    volumes_sell = np.random.poisson(8, num_ticks)  # Sell volumes (slight buy bias)
    deltas = volumes_buy - volumes_sell
    return pd.DataFrame({
        'timestamp': timestamps,
        'price': prices,
        'volume_buy': volumes_buy,
        'volume_sell': volumes_sell,
        'delta': deltas
    })

# Example for real WebSocket (TraderMade Forex Order Flow - uncomment and configure)
# def on_message(ws, message):
#     data = json.loads(message)  # Parse JSON from TraderMade
#     # Append to st.session_state.df: new_row = {'timestamp': datetime.now(), 'price': data['price'], ...}
#     st.session_state.df = pd.concat([st.session_state.df, pd.DataFrame([new_row])], ignore_index=True)
#     st.rerun()  # Trigger Streamlit rerun for live update
#
# def start_websocket():
#     ws = websocket.WebSocketApp("wss://marketdata.tradermade.com/feedadv",  # Your ORDER_FEED_ADDRESS
#                                 on_message=on_message)
#     ws.run_forever()

def update_data():
    """Background thread to simulate live updates every 2 seconds."""
    while not st.session_state.stop_thread:
        new_data = generate_order_flow_data(num_ticks=1)  # Generate 1 new tick
        st.session_state.df = pd.concat([st.session_state.df, new_data], ignore_index=True).tail(500)  # Keep last 500 ticks
        time.sleep(2)

# Start the update thread
if 'thread' not in st.session_state:
    st.session_state.thread = threading.Thread(target=update_data, daemon=True)
    st.session_state.thread.start()

# UI: Sidebar for controls
st.sidebar.header("Order Flow Controls")
symbol = st.sidebar.selectbox("Symbol", ["EURUSD", "BTCUSD", "SPX500"])
update_interval = st.sidebar.slider("Update Interval (s)", 1, 10, 2)
st.sidebar.button("Stop Live Updates", on_click=lambda: setattr(st.session_state, 'stop_thread', True))

# Main dashboard
st.title("🛡️ Live Trading Order Flow Analysis Dashboard")
st.markdown("Real-time order flow insights: Track buy/sell imbalances and volume profiles for smarter trades.")

# Metrics row
col1, col2, col3, col4 = st.columns(4)
latest = st.session_state.df.iloc[-1] if not st.session_state.df.empty else None
if latest is not None:
    col1.metric("Current Price", f"{latest['price']:.5f}")
    col2.metric("Buy Volume (Last Tick)", latest['volume_buy'])
    col3.metric("Sell Volume (Last Tick)", latest['volume_sell'])
    col4.metric("CVD (Cumulative Delta)", f"{st.session_state.df['delta'].sum():.0f}")

# Charts row
if not st.session_state.df.empty:
    df = st.session_state.df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])  # Ensure datetime for plotting

    # 1. Candlestick + Volume Chart
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Price & Volume Candlesticks")
        fig_candle = go.Figure()
        # Simple OHLC from ticks (group by minute for candles)
        ohlc = df.groupby(df['timestamp'].dt.floor('1min')).agg({
            'price': ['first', 'max', 'min', 'last']
        }).droplevel(0, axis=1)
        ohlc.columns = ['open', 'high', 'low', 'close']
        ohlc = ohlc.reset_index()
        fig_candle.add_trace(go.Candlestick(
            x=ohlc['timestamp'], open=ohlc['open'], high=ohlc['high'],
            low=ohlc['low'], close=ohlc['close'], name="Price"
        ))
        # Add volume bars
        fig_candle.add_trace(go.Bar(x=df['timestamp'], y=df['volume_buy'] + df['volume_sell'],
                                    yaxis='y2', name="Total Volume", opacity=0.3))
        fig_candle.update_layout(yaxis2=dict(overlaying='y', side='right', title="Volume"),
                                 title=f"{symbol} Live Price Action")
        st.plotly_chart(fig_candle, use_container_width=True)

    with col_b:
        st.subheader("Order Flow: Cumulative Volume Delta")
        cvd = df['delta'].cumsum()
        fig_delta = go.Figure()
        fig_delta.add_trace(go.Scatter(x=df['timestamp'], y=cvd, mode='lines', name="CVD",
                                       line=dict(color='green' if cvd.iloc[-1] > 0 else 'red')))
        fig_delta.add_hline(y=0, line_dash="dash", line_color="black")
        fig_delta.update_layout(title="Cumulative Delta (Buy - Sell Imbalance)",
                                yaxis_title="Delta")
        st.plotly_chart(fig_delta, use_container_width=True)

    # 2. Volume Profile (Buy/Sell at Price Levels)
    st.subheader("Volume Profile: Buy vs Sell at Price Levels")
    price_bins = np.linspace(df['price'].min(), df['price'].max(), 20)
    df['price_bin'] = pd.cut(df['price'], bins=price_bins, labels=price_bins[:-1])
    volume_profile = df.groupby('price_bin').agg({
        'volume_buy': 'sum',
        'volume_sell': 'sum'
    }).reset_index()
    volume_profile = volume_profile.dropna()

    fig_profile = go.Figure()
    fig_profile.add_trace(go.Bar(x=volume_profile['price_bin'], y=volume_profile['volume_buy'],
                                 name="Buy Volume", marker_color='green', opacity=0.7))
    fig_profile.add_trace(go.Bar(x=volume_profile['price_bin'], y=-volume_profile['volume_sell'],
                                 name="Sell Volume", marker_color='red', opacity=0.7))
    fig_profile.update_layout(barmode='relative', title="Volume Profile (Positive=Buy, Negative=Sell)",
                              xaxis_title="Price Level", yaxis_title="Volume")
    st.plotly_chart(fig_profile, use_container_width=True)

    # Raw data table (last 50 rows)
    st.subheader("Recent Order Flow Ticks")
    st.dataframe(df.tail(50), use_container_width=True)

else:
    st.info("Starting live data feed... Refresh in a few seconds.")

# Footer
st.markdown("---")
st.caption("Built with Streamlit & Plotly. For production, integrate real-time APIs and deploy to the cloud.")
