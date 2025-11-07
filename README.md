# Weekly Covered Calls & Put Credit Spreads Scanner (FMP)

A Streamlit app to help you find **weekly covered call** and **put credit spread** ideas sized to an investment budget (default: **$4,000**).  
Data source: **Financial Modeling Prep (FMP)**.

## What it does
- Pulls live quotes and option chains from FMP.
- For **covered calls**: filters stocks affordable for 100-share lots, picks weekly calls (7–14 DTE) near delta 0.20–0.35, and ranks by estimated **annualized ROI** and **POP** (probability-of-profit approximation from delta).
- For **put credit spreads**: finds weekly short puts near delta 0.20–0.30, pairs with long puts 1–2 strikes lower, computes **credit, max risk, return on risk, POP**, and sizes trades to stay within your budget.
- Lets you paste a watchlist or use the preloaded universe.
- Export results to CSV.

> **Note**: POP and greeks are approximations. Real fills vary; always double-check in your broker before trading.

## Quickstart (Local)
1. Install Python 3.10+
2. `pip install -r requirements.txt`
3. Put your FMP API key in `.streamlit/secrets.toml` like:
   ```toml
   [fmp]
   api_key = "REPLACE_WITH_YOUR_FMP_KEY"
   ```
4. Run: `streamlit run app.py`

## Deploy to Streamlit Cloud
1. Push this folder to a **public GitHub repo** (e.g., `weekly-options-scanner`).
2. In Streamlit Cloud, create a new app from your repo.
3. Add a **Secret**:
   ```toml
   [fmp]
   api_key = "REPLACE_WITH_YOUR_FMP_KEY"
   ```

## Configuration tips
- Default budget = $4,000; change in the sidebar.
- For covered calls, the app filters tickers with **price ≤ budget / 100** (affordable 100-share lots).
- For put credit spreads, the app limits position **max risk ≤ budget** (can open multiple spreads if sized).

## Disclaimers
Educational use only. Not financial advice. Markets involve risk.
