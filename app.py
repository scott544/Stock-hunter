import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from io import StringIO
from datetime import datetime

st.set_page_config(page_title="Core Predictor + Stock Hunter", layout="centered")
st.title("📈 Core Predictor + 🕵️ Stock Hunter")
st.caption("Mobile-friendly, price-data-only build. Toggle methods to blend forecasts.")

# -------- Helpers --------
@st.cache_data(ttl=600)
def load_history(ticker, period="2y", interval="1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
        return df.dropna()
    except Exception as e:
        st.error(f"Failed to load data for {ticker}: {e}")
        return pd.DataFrame()

def safe_last(series):
    try:
        return float(series.dropna().iloc[-1])
    except Exception:
        return np.nan

def gbm_forecast(close_series, horizon_days, lookback=60):
    prices = close_series.dropna()
    if len(prices) < 5:
        return np.nan, np.nan, np.nan, 0.0
    window = min(lookback, max(5, len(prices)-1))
    rets = np.log(prices).diff().dropna().tail(window)
    mu = float(rets.mean())
    sigma = float(rets.std())
    last_price = float(prices.iloc[-1])
    pred = last_price * np.exp(mu * horizon_days)
    band = sigma * np.sqrt(horizon_days)
    lo = last_price * np.exp(mu * horizon_days - band)
    hi = last_price * np.exp(mu * horizon_days + band)
    trend_strength = abs(mu) / (sigma + 1e-9)
    conf = float(np.clip(0.5 + 0.25*np.tanh((trend_strength-0.1)*2), 0, 1))
    return pred, lo, hi, conf

def linreg_forecast(close_series, horizon_days):
    y = np.log(close_series.dropna().values)
    if len(y) < 10:
        return np.nan, np.nan
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    future_x = len(y) + horizon_days
    y_hat = intercept + slope * future_x
    last = float(np.exp(y[-1]))
    pred = float(np.exp(y_hat))
    y_fit = intercept + slope * x
    ss_res = np.sum((y - y_fit)**2)
    ss_tot = np.sum((y - np.mean(y))**2) + 1e-9
    r2 = 1 - ss_res/ss_tot
    conf = float(np.clip((r2 + 1)/2, 0, 1))
    return pred, conf

def ema_mean_reversion(close_series, horizon_days):
    close = close_series.dropna()
    if len(close) < 20:
        return np.nan, 0.3
    ema20 = close.ewm(span=20, adjust=False).mean()
    last = float(close.iloc[-1])
    anchor = float(ema20.iloc[-1])
    alpha = min(1.0, 0.15 * np.sqrt(horizon_days))
    pred = last + alpha * (anchor - last)
    conf = float(np.clip(0.4 + 0.2*(abs(last-anchor)/(1e-9 + close.pct_change().std()*last) < 2), 0, 1))
    return pred, conf

def combine_preds(preds, confs, weights):
    w = np.array([weights[i]*confs[i] for i in range(len(preds))], dtype=float)
    preds = np.array(preds, dtype=float)
    if np.all(np.isnan(preds)) or np.sum(w) == 0:
        return np.nan, 0.0
    valid = ~np.isnan(preds)
    w = w[valid]
    preds = preds[valid]
    if len(preds)==0 or np.sum(w)==0:
        return np.nan, 0.0
    combined = float(np.average(preds, weights=w))
    conf = float(np.clip(np.sum(w)/ (np.sum(weights)+1e-9), 0, 1))
    return combined, conf

def pct_change(a, b):
    if np.isnan(a) or np.isnan(b) or b==0:
        return np.nan
    return a/b - 1.0

# Session log
if "log_rows" not in st.session_state:
    st.session_state.log_rows = []

def log_event(kind, payload:dict):
    row = {"timestamp": datetime.utcnow().isoformat()+"Z", "type": kind}
    row.update(payload)
    st.session_state.log_rows.append(row)

def download_log_button():
    if len(st.session_state.log_rows)==0:
        st.caption("No session logs yet.")
        return
    csv = pd.DataFrame(st.session_state.log_rows)
    buf = StringIO()
    csv.to_csv(buf, index=False)
    st.download_button("⬇️ Download Session Log (CSV)", data=buf.getvalue(), file_name="session_log.csv", mime="text/csv")

# -------- UI --------
tab = st.radio("Select mode", ["🔮 Predictor", "🕵️ Stock Hunter"], horizontal=True)

if tab == "🔮 Predictor":
    st.subheader("Single-Ticker Forecasts")
    ticker = st.text_input("Ticker (AAPL, MSFT, CBA.AX, BTC-USD)", value="AAPL")

    with st.expander("⚙️ Settings", expanded=False):
        period = st.selectbox("History window", ["1y","2y","5y"], index=1)
        show_chart = st.checkbox("Show historical chart", value=True)
        st.markdown("**Model Toggles & Weights**")
        use_gbm = st.checkbox("Geometric Brownian Motion", value=True)
        w_gbm = st.slider("GBM weight", 0.0, 1.0, 0.4, 0.05)
        use_lin = st.checkbox("Linear Regression (log price)", value=True)
        w_lin = st.slider("LinReg weight", 0.0, 1.0, 0.4, 0.05)
        use_mr = st.checkbox("EMA Mean-Reversion", value=True)
        w_mr = st.slider("Mean-Reversion weight", 0.0, 1.0, 0.2, 0.05)

    if ticker:
        df = load_history(ticker, period=period, interval="1d")
        if df.empty:
            st.warning("No data returned for that ticker.")
        else:
            last = safe_last(df["Close"])
            if show_chart:
                st.line_chart(df["Close"], height=240, use_container_width=True)

            horizons = {"1 Day":1, "1 Week (5d)":5, "1 Month (~21d)":21, "3 Months (~63d)":63}
            rows = []
            for label, days in horizons.items():
                preds = []; confs = []; weights = []
                if use_gbm:
                    p, lo, hi, c = gbm_forecast(df["Close"], days)
                    preds.append(p); confs.append(c); weights.append(w_gbm)
                if use_lin:
                    p, c = linreg_forecast(df["Close"], days)
                    preds.append(p); confs.append(c); weights.append(w_lin)
                if use_mr:
                    p, c = ema_mean_reversion(df["Close"], days)
                    preds.append(p); confs.append(c); weights.append(w_mr)

                combined, cscore = combine_preds(preds, confs, weights)
                change = pct_change(combined, last)
                rows.append({
                    "Horizon": label,
                    "Predicted Price": combined,
                    "Expected % Move": change,
                    "Model Confidence (0-1)": cscore
                })
            out = pd.DataFrame(rows)
            st.dataframe(out.style.format({
                "Predicted Price":"{:,.4f}",
                "Expected % Move":"{:.2%}",
                "Model Confidence (0-1)":"{:.2f}"
            }), use_container_width=True, height=280)

            log_event("predict", {"ticker": ticker, "period": period,
                                  "use_gbm":use_gbm,"w_gbm":w_gbm,
                                  "use_lin":use_lin,"w_lin":w_lin,
                                  "use_mr":use_mr,"w_mr":w_mr})

    download_log_button()

else:
    st.subheader("Market Scan (Preview)")
    exch = st.selectbox("Exchange / Market", ["US","ASX","CRYPTO"], index=0)
    defaults = {
        "US": "AAPL, MSFT, NVDA, TSLA, AMD, META, GOOGL, AMZN",
        "ASX": "CBA.AX, BHP.AX, WES.AX, CSL.AX, WBC.AX, NAB.AX, ANZ.AX",
        "CRYPTO": "BTC-USD, ETH-USD, SOL-USD, ADA-USD, XRP-USD"
    }
    tickers = st.text_area("Tickers to scan (comma separated)", value=defaults[exch], height=90)

    with st.expander("⚙️ Filters & Scoring", expanded=False):
        pmin = st.number_input("Min price", value=0.0, step=1.0)
        pmax = st.number_input("Max price (0 = no max)", value=0.0, step=1.0)
        cmin = st.number_input("Min market cap (USD)", value=0.0, step=1e6, format="%.0f")
        cmax = st.number_input("Max market cap (USD) (0 = no max)", value=0.0, step=1e6, format="%.0f")
        st.write("**Scoring Toggles & Weights**")
        sh_use_gbm = st.checkbox("GBM component", value=True)
        sh_w_gbm = st.slider("GBM weight ", 0.0, 1.0, 0.4, 0.05, key="w1")
        sh_use_lin = st.checkbox("LinReg component", value=True)
        sh_w_lin = st.slider("LinReg weight ", 0.0, 1.0, 0.4, 0.05, key="w2")
        sh_use_mr = st.checkbox("Mean-Reversion component", value=True)
        sh_w_mr = st.slider("Mean-Reversion weight ", 0.0, 1.0, 0.2, 0.05, key="w3")

    if st.button("Run Stock Hunter"):
        symbols = [s.strip() for s in tickers.split(",") if s.strip()]
        rows = []
        with st.spinner("Scanning..."):
            for sym in symbols:
                data = load_history(sym, period="1y", interval="1d")
                if data.empty:
                    continue
                last = safe_last(data["Close"])
                mcap = np.nan
                try:
                    tk = yf.Ticker(sym)
                    fi = tk.fast_info if hasattr(tk,"fast_info") else {}
                    mcap = float(fi.get("market_cap", np.nan)) if fi else np.nan
                except Exception:
                    pass

                if pmax > 0 and not np.isnan(last) and last > pmax: 
                    continue
                if not np.isnan(last) and last < pmin:
                    continue
                if not np.isnan(mcap):
                    if cmin and mcap < cmin:
                        continue
                    if cmax and cmax > 0 and mcap > cmax:
                        continue

                preds=[]; confs=[]; weights=[]
                if sh_use_gbm:
                    p, _, _, c = gbm_forecast(data["Close"], 21)
                    preds.append(p); confs.append(c); weights.append(sh_w_gbm)
                if sh_use_lin:
                    p, c = linreg_forecast(data["Close"], 21)
                    preds.append(p); confs.append(c); weights.append(sh_w_lin)
                if sh_use_mr:
                    p, c = ema_mean_reversion(data["Close"], 21)
                    preds.append(p); confs.append(c); weights.append(sh_w_mr)

                def momentum_score(df):
                    close = df["Close"].dropna()
                    if len(close) < 63:
                        return np.nan
                    ret_21 = close.iloc[-1] / close.iloc[-21] - 1.0 if len(close) >= 21 else np.nan
                    ret_63 = close.iloc[-1] / close.iloc[-63] - 1.0 if len(close) >= 63 else np.nan
                    sma200 = close.rolling(200).mean()
                    above200 = 1.0 if len(sma200.dropna())>0 and close.iloc[-1] > sma200.iloc[-1] else 0.0
                    vol = np.log(close).diff().std()
                    score = 0.5*np.nan_to_num(ret_21) + 0.4*np.nan_to_num(ret_63) + 0.1*above200 - 0.5*np.nan_to_num(vol)
                    return float(score)

                combined, cscore = combine_preds(preds, confs, weights)
                one_m_pct = pct_change(combined, last)
                mom = momentum_score(data)

                rows.append({
                    "Ticker": sym,
                    "Last Price": last,
                    "Market Cap": mcap,
                    "Momentum Score": mom,
                    "Pred 1M %": one_m_pct,
                    "Model Confidence (0-1)": cscore
                })

        if rows:
            out = pd.DataFrame(rows).sort_values(["Pred 1M %","Momentum Score","Model Confidence (0-1)"], ascending=False)
            st.dataframe(out.style.format({
                "Last Price":"{:,.4f}",
                "Market Cap":"{:,.0f}",
                "Momentum Score":"{:.4f}",
                "Pred 1M %":"{:.2%}",
                "Model Confidence (0-1)":"{:.2f}"
            }), use_container_width=True, height=280)
            log_event("stock_hunter", {
                "exch":exch,"pmin":pmin,"pmax":pmax,"cmin":cmin,"cmax":cmax,
                "use_gbm":sh_use_gbm,"w_gbm":sh_w_gbm,
                "use_lin":sh_use_lin,"w_lin":sh_w_lin,
                "use_mr":sh_use_mr,"w_mr":sh_w_mr,
                "tickers":",".join(symbols)
            })
        else:
            st.info("No symbols matched filters or data unavailable.")

    download_log_button()

st.caption("Tip: Add to Home Screen for app-like use. ASX uses .AX (e.g., CBA.AX). Crypto uses -USD (e.g., BTC-USD).")
