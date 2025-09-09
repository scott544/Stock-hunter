# Core Predictor + Stock Hunter (Mobile-Friendly)

A lightweight Streamlit app for stock & crypto forecasts and a basic Stock Hunter screener.

## Features
- Forecasts: 1D / 1W / 1M / 3M using a blend of GBM, log-linear trend, and EMA mean-reversion.
- Stock Hunter: Filter by market (US/ASX/Crypto), price, and market cap; rank with the same blended model.
- Mobile-friendly UI with toggles and CSV log export.

## Run Locally
```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
streamlit run app.py
