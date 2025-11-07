# Weekly Options Scanner — FMP + Yahoo Finance (Free)

This Streamlit app finds **weekly covered calls** and **put credit spreads** sized to your budget.  
It now supports **Yahoo Finance (yfinance)** as a free data source for quotes, historical volatility, and (optionally) option chains.

## Data Sources
- **FMP (Financial Modeling Prep)** — quotes, historical vol, **option chains** (primary).
- **Yahoo Finance** — quotes, historical vol (from price history), and **option chains** (fallback or selectable).

Choose the source in the **sidebar**:
- Quotes/Volatility: **FMP** or **Yahoo**
- Option Chains: **FMP**, **Yahoo**, or **Auto** (try FMP then fallback to Yahoo)

## Local Run
1. `pip install -r requirements.txt`
2. Add your FMP key (optional but recommended for option chains) in `.streamlit/secrets.toml`:
   ```toml
   [fmp]
   api_key = "YOUR_FMP_KEY"
   ```
3. `streamlit run app.py`

## Notes
- Yahoo endpoints are rate-limited and sometimes incomplete. The app will gracefully skip tickers that fail.
- POP and greeks are approximations; verify in your broker platform.
