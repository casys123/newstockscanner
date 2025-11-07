import os, io, math, time, requests, datetime as dt
import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf

# -------------------------
# Config
# -------------------------
FMP_KEY = st.secrets.get("fmp", {}).get("api_key", os.getenv("FMP_API_KEY", ""))
FMP_BASE = "https://financialmodelingprep.com/api/v3"

# -------------------------
# Helpers
# -------------------------
def annualize_return(credit, collateral, dte):
    if collateral <= 0 or dte <= 0:
        return 0.0
    simple = credit / collateral
    return (1 + simple) ** (365.0 / dte) - 1

def pick_nearest_weekly(expirations, min_dte=7, max_dte=14, today=None):
    if today is None:
        today = dt.date.today()
    choices = []
    for e in expirations or []:
        try:
            d = dt.datetime.strptime(e, "%Y-%m-%d").date()
            dte = (d - today).days
            if min_dte <= dte <= max_dte:
                choices.append((e, dte))
        except Exception:
            continue
    if not choices:
        return None, None
    choices.sort(key=lambda x: x[1])
    return choices[0]

# -------------------------
# FMP Access
# -------------------------
def fmp_get(path, params=None):
    if params is None: params = {}
    if not FMP_KEY:
        raise RuntimeError("FMP API key missing. Add it to .streamlit/secrets.toml as [fmp].api_key")
    params["apikey"] = FMP_KEY
    r = requests.get(f"{FMP_BASE}/{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def fmp_get_quotes(tickers):
    data = fmp_get("quote/" + ",".join(tickers))
    if isinstance(data, dict): data = [data]
    df = pd.DataFrame(data)
    if df.empty:
        return pd.DataFrame(columns=["ticker","last","volAvg","beta"])
    ticker_col = "symbol" if "symbol" in df.columns else ("ticker" if "ticker" in df.columns else None)
    price_col = "price" if "price" in df.columns else ("regularMarketPrice" if "regularMarketPrice" in df.columns else None)
    out = pd.DataFrame()
    out["ticker"] = df[ticker_col] if ticker_col else pd.Series(tickers[:len(df)])
    out["last"]   = df[price_col] if price_col else np.nan
    out["volAvg"] = df["volAvg"] if "volAvg" in df.columns else np.nan
    out["beta"]   = df["beta"]   if "beta"   in df.columns else np.nan
    out = out.dropna(subset=["last"]).groupby("ticker", as_index=False).agg({"last":"last","volAvg":"last","beta":"last"})
    return out

def fmp_hist_vol(symbol):
    try:
        js = fmp_get(f"historical-volatility/{symbol}?limit=1")
        if isinstance(js, list) and js:
            return js[0].get("volatility", np.nan)
    except Exception:
        pass
    return np.nan

def fmp_option_chain(symbol):
    try:
        chain = fmp_get(f"option-chain/{symbol}")
        if isinstance(chain, list): return chain
    except Exception:
        pass
    return []

# -------------------------
# Yahoo Finance Access
# -------------------------
def yf_get_quotes(tickers):
    tickers = [t for t in tickers if t]
    if not tickers:
        return pd.DataFrame(columns=["ticker","last","volAvg","beta"])
    yft = yf.Tickers(" ".join(tickers))
    rows=[]
    for t in tickers:
        try:
            info = yft.tickers[t].fast_info  # fast_info is quicker and lighter than .info
            last = info.get("last_price") or info.get("last_price", np.nan)
            if last is None: last = np.nan
            rows.append({"ticker": t, "last": float(last), "volAvg": np.nan, "beta": np.nan})
        except Exception:
            rows.append({"ticker": t, "last": np.nan, "volAvg": np.nan, "beta": np.nan})
    df = pd.DataFrame(rows).dropna(subset=["last"])
    return df

def yf_hist_vol(symbol, lookback_days=30):
    try:
        hist = yf.download(symbol, period="90d", interval="1d", progress=False, auto_adjust=True)
        if hist is None or hist.empty: return np.nan
        close = hist["Close"].tail(lookback_days+1)
        ret = close.pct_change().dropna()
        if ret.empty: return np.nan
        # Daily std * sqrt(252) for annualized volatility
        return float(ret.std() * (252 ** 0.5))
    except Exception:
        return np.nan

def yf_option_chain(symbol):
    try:
        t = yf.Ticker(symbol)
        exps = t.options or []
        out = []
        for exp in exps:
            # Yahoo expiry strings like '2025-11-08' already YYYY-MM-DD
            try:
                calls = t.option_chain(exp).calls
                puts  = t.option_chain(exp).puts
            except Exception:
                continue
            options = []
            def map_rows(df, opt_type):
                for _, r in df.iterrows():
                    bid = float(r.get("bid", np.nan)) if pd.notna(r.get("bid", np.nan)) else np.nan
                    ask = float(r.get("ask", np.nan)) if pd.notna(r.get("ask", np.nan)) else np.nan
                    last = float(r.get("lastPrice", np.nan)) if pd.notna(r.get("lastPrice", np.nan)) else np.nan
                    mid = None
                    if pd.notna(bid) and pd.notna(ask) and bid>0 and ask>0:
                        mid = (bid+ask)/2
                    elif pd.notna(last) and last>0:
                        mid = last
                    options.append({
                        "symbol": r.get("contractSymbol"),
                        "underlyingSymbol": symbol,
                        "optionType": opt_type,
                        "strike": float(r.get("strike", np.nan)),
                        "bid": bid,
                        "ask": ask,
                        "last": last,
                        "impliedVolatility": float(r.get("impliedVolatility", np.nan)),
                        "delta": np.nan,  # Yahoo doesn't provide delta in public endpoint
                        "mid": mid
                    })
            map_rows(calls, "call")
            map_rows(puts, "put")
            out.append({"expirationDate": exp, "options": options})
        return out
    except Exception:
        return []
# -------------------------
# Alpha Vantage Access (quotes + historical vol from prices)
# -------------------------
def av_get_key():
    return st.secrets.get("alphavantage", {}).get("api_key", os.getenv("ALPHAVANTAGE_API_KEY", ""))

def av_get_quote(symbol):
    """GLOBAL_QUOTE endpoint; returns last price or NaN"""
    key = av_get_key()
    if not key:
        return np.nan
    try:
        r = requests.get("https://www.alphavantage.co/query", params={"function":"GLOBAL_QUOTE","symbol":symbol,"apikey":key}, timeout=30)
        r.raise_for_status()
        j = r.json().get("Global Quote", {})
        p = j.get("05. price")
        return float(p) if p is not None else np.nan
    except Exception:
        return np.nan

def av_get_quotes(tickers, throttle_seconds=12):
    """Respect free-tier rate limits (5 req/min)."""
    rows = []
    for i, t in enumerate([t for t in tickers if t]):
        last = av_get_quote(t)
        rows.append({"ticker": t, "last": last, "volAvg": np.nan, "beta": np.nan})
        if i < len(tickers)-1:
            time.sleep(throttle_seconds)  # be gentle with AV limits
    df = pd.DataFrame(rows).dropna(subset=["last"])
    return df

def av_hist_vol(symbol, lookback_days=30):
    """Compute annualized hist vol from TIME_SERIES_DAILY_ADJUSTED."""
    key = av_get_key()
    if not key:
        return np.nan
    try:
        r = requests.get("https://www.alphavantage.co/query",
                         params={"function":"TIME_SERIES_DAILY_ADJUSTED","symbol":symbol,"outputsize":"compact","apikey":key},
                         timeout=30)
        r.raise_for_status()
        j = r.json().get("Time Series (Daily)", {})
        if not j:
            return np.nan
        # j is dict of date->dict; convert to df sorted by date
        df = pd.DataFrame(j).T
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        close = pd.to_numeric(df["4. close"], errors="coerce").dropna()
        close = close.tail(lookback_days+1)
        ret = close.pct_change().dropna()
        if ret.empty: return np.nan
        return float(ret.std() * (252 ** 0.5))
    except Exception:
        return np.nan


# -------------------------
# Generic Chain & Flatten
# -------------------------
def flatten_chain(chain, call_or_put="call", target_exp=None):
    rows = []
    for exp in chain:
        expiration = exp.get("expirationDate")
        if target_exp and expiration != target_exp:
            continue
        for c in exp.get("options", []) or []:
            if str(c.get("optionType","")).lower() != call_or_put:
                continue
            bid = c.get("bid"); ask = c.get("ask"); last = c.get("last"); mid = c.get("mid")
            if mid is None:
                if bid and ask and bid>0 and ask>0: mid=(bid+ask)/2
                elif last and last>0: mid=last
            rows.append({
                "symbol": c.get("symbol"),
                "underlying": c.get("underlyingSymbol") or c.get("underlying", ""),
                "type": call_or_put,
                "expiration": expiration,
                "strike": c.get("strike"),
                "bid": bid,
                "ask": ask,
                "last": last,
                "mid": mid,
                "delta": c.get("delta"),
                "iv": c.get("impliedVolatility") or c.get("iv"),
            })
    return pd.DataFrame(rows)

# -------------------------
# Scanners
# -------------------------
def scan_covered_calls(quotes_df, budget, min_dte, max_dte, d_low, d_high, chain_source):
    results = []
    today = dt.date.today()
    for symbol, last in quotes_df[["ticker","last"]].itertuples(index=False):
        lot_cost = last * 100
        if lot_cost <= 0 or lot_cost > budget: continue

        # get chain
        chain = []
        if chain_source in ("FMP","Auto"):
            try: chain = fmp_option_chain(symbol)
            except Exception: chain = []
        if not chain and chain_source in ("YF","Auto"):
            chain = yf_option_chain(symbol)

        exps = [x.get("expirationDate") for x in chain] if chain else []
        exp, dte = pick_nearest_weekly(exps, min_dte, max_dte, today=today)
        if not exp: continue

        calls = flatten_chain(chain, "call", target_exp=exp).dropna(subset=["mid","strike"])
        if calls.empty: continue
        window = calls.dropna(subset=["delta"]).copy()
        window = window[(window["delta"]>=d_low) & (window["delta"]<=d_high)] if not window.empty else pd.DataFrame()
        if window.empty:
            # fallback: OTM strike just above last
            window = calls[calls["strike"]>last].copy()
        if window.empty: continue

        window["credit"] = window["mid"] * 100
        # POP approximation: if delta available use 1-|delta| else estimate by moneyness heuristic
        if "delta" in window.columns and window["delta"].notna().any():
            window["pop_est"] = (1 - window["delta"].abs()).clip(0,1)
        else:
            # simple heuristic: further OTM = higher POP
            window["pop_est"] = (1 - ((window["strike"]-last).clip(lower=0) / max(last, 1e-6)).clip(0,1))
        window["roi_ann"] = window.apply(lambda r: annualize_return(r["credit"], lot_cost, dte), axis=1)

        best = window.sort_values(["roi_ann","pop_est"], ascending=[False, False]).head(1)
        if best.empty: continue
        b = best.iloc[0].to_dict()
        b.update({"underlying": symbol, "under_price": last, "lot_cost": lot_cost, "dte": dte})
        results.append(b)

    if not results:
        return pd.DataFrame(columns=["underlying","under_price","expiration","dte","strike","mid","credit","pop_est","roi_ann","lot_cost"])
    df = pd.DataFrame(results).sort_values(["roi_ann","pop_est"], ascending=[False, False])
    return df

def scan_put_credit_spreads(quotes_df, budget, min_dte, max_dte, d_low, d_high, widths, chain_source):
    rows = []
    today = dt.date.today()
    for symbol, last in quotes_df[["ticker","last"]].itertuples(index=False):
        chain = []
        if chain_source in ("FMP","Auto"):
            try: chain = fmp_option_chain(symbol)
            except Exception: chain = []
        if not chain and chain_source in ("YF","Auto"):
            chain = yf_option_chain(symbol)

        exps = [x.get("expirationDate") for x in chain] if chain else []
        exp, dte = pick_nearest_weekly(exps, min_dte, max_dte, today=today)
        if not exp: continue

        puts = flatten_chain(chain, "put", target_exp=exp).dropna(subset=["strike","mid"])
        if puts.empty: continue

        # Prefer delta-filter if available
        window = puts.dropna(subset=["delta"]).copy()
        if not window.empty:
            window["abs_delta"] = window["delta"].abs()
            window = window[(window["abs_delta"]>=d_low) & (window["abs_delta"]<=d_high)]
        if window.empty:
            window = puts[puts["strike"] < last].copy()
        if window.empty: continue

        for _, short in window.iterrows():
            for w in widths:
                long_strike = round(float(short["strike"]) - float(w), 2)
                long = puts.iloc[(puts["strike"] - long_strike).abs().argsort()[:1]].copy()
                if long.empty: continue
                long = long.iloc[0]

                short_mid = float(short["mid"]); long_mid=float(long["mid"])
                if np.isnan(short_mid) or np.isnan(long_mid): continue

                credit = max(short_mid - long_mid, 0) * 100
                width_dollars = max(float(short["strike"]) - float(long["strike"]), 0) * 100
                max_risk = max(width_dollars - credit, 0)
                if max_risk <= 0 or max_risk > budget: continue

                pop_est = (1 - abs(short.get("delta", 0))) if pd.notna(short.get("delta", np.nan)) else np.nan
                roi_ann = annualize_return(credit, max_risk, dte)

                rows.append({
                    "underlying": symbol,
                    "under_price": last,
                    "expiration": exp,
                    "dte": dte,
                    "short_strike": float(short["strike"]),
                    "long_strike": float(long["strike"]),
                    "width": width_dollars/100.0,
                    "credit": credit,
                    "max_risk": max_risk,
                    "return_on_risk": credit/max_risk if max_risk>0 else np.nan,
                    "roi_ann": roi_ann,
                    "pop_est": np.nan if pd.isna(pop_est) else max(min(pop_est,1),0)
                })
    if not rows:
        return pd.DataFrame(columns=["underlying","under_price","expiration","dte","short_strike","long_strike","width","credit","max_risk","return_on_risk","roi_ann","pop_est"])
    return pd.DataFrame(rows).sort_values(["roi_ann","pop_est"], ascending=[False, False])

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="Weekly Options Scanner — FMP + Yahoo", page_icon="📈", layout="wide")
st.title("📈 Weekly Covered Calls & Put Credit Spreads — FMP + Yahoo Finance")

with st.sidebar:
    st.header("Settings")
    budget = st.number_input("Investment Budget ($)", min_value=500, max_value=100000, value=4000, step=100)
    min_dte, max_dte = st.slider("DTE Range", 0, 45, (7,14))

    st.subheader("Data Sources")
    limit_to_top = st.checkbox('Only scan Top N most volatile before option chains', value=True)
    top_n = st.slider('N (by annualized historical vol)', 5, 50, 12)
    quote_source = st.selectbox("Quotes/Volatility source", ["FMP", "Yahoo", "AlphaVantage"])
    chain_source = st.selectbox("Option Chains source", ["Auto","FMP","YF"])

    st.subheader("Strategy Params")
    cc_low, cc_high = st.slider("Covered Call delta range", 0.05, 0.8, (0.20, 0.35))
    pc_low, pc_high = st.slider("Put Credit Spread short-put delta", 0.05, 0.8, (0.20, 0.30))
    widths = st.multiselect("PCS width (points)", options=[0.5,1,2,3,5], default=[1,2])
    if not widths: widths=[1,2]

    st.subheader("Universe")
    default_watchlist = "AAPL,MSFT,AMD,SOFI,F,PBR,NU,PLTR,CCL,NIO,T,PARA,UAL,DIS,AI,SNAP,KO,INTC,ENVX,RIVN,TSLA,SMCI,UPST,COIN"
    wl = st.text_area("Tickers (comma-separated)", value=default_watchlist, height=100)
    tickers = [t.strip().upper() for t in wl.split(",") if t.strip()]

    run_btn = st.button("Run Scan")

# Validate FMP when selected
if quote_source == "FMP" or chain_source == "FMP":
    if not FMP_KEY:
        st.warning("FMP selected but no API key found. Add it to `.streamlit/secrets.toml` under [fmp]. Using Yahoo where possible.")

# Validate Alpha Vantage when selected
if quote_source == "AlphaVantage":
    from os import getenv
    if not (st.secrets.get("alphavantage", {}).get("api_key") or getenv("ALPHAVANTAGE_API_KEY")):
        st.warning("Alpha Vantage selected but no API key found. Add it to `.streamlit/secrets.toml` under [alphavantage].")

if run_btn:
    # Get quotes + volatility
    st.info("Fetching quotes & volatility...")
    if quote_source == "FMP" and FMP_KEY:
        quotes = fmp_get_quotes(tickers)
        quotes["hist_vol"] = quotes["ticker"].apply(fmp_hist_vol)
    elif quote_source == "AlphaVantage":
        quotes = av_get_quotes(tickers)
        quotes["hist_vol"] = quotes["ticker"].apply(av_hist_vol)
    else:
        quotes = yf_get_quotes(tickers)
        quotes["hist_vol"] = quotes["ticker"].apply(yf_hist_vol)

    if quotes.empty:
        st.error("No quotes returned. Check tickers or data source.")
        st.stop()

    quotes["vol_rank"] = quotes["hist_vol"].rank(ascending=False, method="min")
    # Optionally restrict to Top N most volatile to cut API calls / speed up
    if limit_to_top:
        quotes = quotes.sort_values("hist_vol", ascending=False).head(top_n).reset_index(drop=True)

    st.subheader("🔝 Top Volatile Stocks (annualized hist vol)")
    st.dataframe(quotes.sort_values("hist_vol", ascending=False).head(15))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Covered Calls (Weekly)")
        affordable = quotes[quotes["last"] * 100 <= budget].copy()
        if affordable.empty:
            st.warning("No tickers affordable for covered calls with the current budget.")
            cc_df = pd.DataFrame()
        else:
            cc_df = scan_covered_calls(
                affordable, budget, min_dte, max_dte, cc_low, cc_high, chain_source
            )
        if not cc_df.empty:
            st.dataframe(cc_df.reset_index(drop=True))
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                cc_df.to_excel(writer, index=False, sheet_name="CoveredCalls")
            st.download_button("⬇️ Download Covered Calls (Excel)", buf.getvalue(), "covered_calls.xlsx")
        else:
            st.write("—")

    with col2:
        st.subheader("Put Credit Spreads (Weekly)")
        pcs_df = scan_put_credit_spreads(
            quotes, budget, min_dte, max_dte, pc_low, pc_high, widths, chain_source
        )
        if not pcs_df.empty:
            st.dataframe(pcs_df.reset_index(drop=True))
            buf2 = io.BytesIO()
            with pd.ExcelWriter(buf2, engine="xlsxwriter") as writer:
                pcs_df.to_excel(writer, index=False, sheet_name="PutSpreads")
            st.download_button("⬇️ Download Put Spreads (Excel)", buf2.getvalue(), "put_credit_spreads.xlsx")
        else:
            st.write("—")

    st.caption("POP ≈ 1 − |delta| (heuristic if delta available). ROI_ANN annualizes return by DTE. Validate in your broker.")
else:
    st.caption("Pick your data sources, set your tickers, and click **Run Scan**.")
