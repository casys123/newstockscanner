import os, math, time, requests, datetime as dt
from dataclasses import dataclass
import pandas as pd
import numpy as np
import streamlit as st

# -------------------------
# Config & Helpers
# -------------------------
FMP_KEY = st.secrets.get("fmp", {}).get("api_key", os.getenv("FMP_API_KEY", ""))
BASE = "https://financialmodelingprep.com/api/v3"

def fmp_get(path, params=None):
    if params is None: params = {}
    if not FMP_KEY:
        raise RuntimeError("FMP API key missing. Add it to .streamlit/secrets.toml as [fmp].api_key")
    params["apikey"] = FMP_KEY
    url = f"{BASE}/{path}"
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def annualize_return(credit, collateral, dte):
    if collateral <= 0 or dte <= 0:
        return 0.0
    simple = credit / collateral
    return (1 + simple) ** (365.0 / dte) - 1

def pick_nearest_weekly(expirations, min_dte=7, max_dte=14, today=None):
    if today is None:
        today = dt.date.today()
    choices = []
    for e in expirations:
        try:
            d = dt.datetime.strptime(e, "%Y-%m-%d").date()
            dte = (d - today).days
            if min_dte <= dte <= max_dte:
                choices.append((e, dte))
        except:
            continue
    if not choices:
        return None, None
    choices.sort(key=lambda x: x[1])
    return choices[0]  # (expiration, dte)

# -------------------------
# Option Chain Fetchers
# -------------------------
def get_quote_batch(tickers):
    # FMP supports comma-separated for "quote"
    data = fmp_get("quote/" + ",".join(tickers))
    df = pd.DataFrame(data)
    if df.empty:
        return pd.DataFrame(columns=["symbol","price"])
    return df[["symbol","price"]].rename(columns={"price":"last"})

def get_option_chain(symbol):
    # FMP endpoints (subject to plan): /option-chain/{symbol}?limit=... returns list with calls/puts grouped by expiration
    # We'll call /option-chain to get available expirations then /option-chain-implied-volatility if needed.
    try:
        chain = fmp_get(f"option-chain/{symbol}")
        return chain
    except Exception as e:
        st.warning(f"Failed to load options for {symbol}: {e}")
        return []

def flatten_chain(chain, call_or_put="call", target_exp=None):
    rows = []
    for exp in chain:
        expiration = exp.get("expirationDate")
        if target_exp and expiration != target_exp:
            continue
        # items inside have 'optionType' = 'call' or 'put'
        for c in exp.get("options", []):
            if str(c.get("optionType","")).lower() != call_or_put:
                continue
            rows.append({
                "symbol": c.get("symbol"),
                "underlying": c.get("underlyingSymbol") or c.get("underlying", ""),
                "type": call_or_put,
                "expiration": expiration,
                "strike": c.get("strike"),
                "bid": c.get("bid"),
                "ask": c.get("ask"),
                "last": c.get("last"),
                "mid": np.nan if c.get("bid") in (None,0) or c.get("ask") in (None,0) else (c.get("bid")+c.get("ask"))/2,
                "delta": c.get("delta"),
                "iv": c.get("impliedVolatility") or c.get("iv"),
            })
    df = pd.DataFrame(rows)
    return df

# -------------------------
# Strategy Scans
# -------------------------
def scan_covered_calls(quotes_df, budget, min_dte=7, max_dte=14, target_delta_low=0.20, target_delta_high=0.35):
    results = []
    today = dt.date.today()
    for symbol, last in quotes_df[["symbol","last"]].itertuples(index=False):
        # require 100-share affordability
        lot_cost = last * 100
        if lot_cost <= 0 or lot_cost > budget:
            continue

        chain = get_option_chain(symbol)
        exps = [x.get("expirationDate") for x in chain]
        exp, dte = pick_nearest_weekly(exps, min_dte, max_dte, today=today)
        if not exp:
            continue

        calls = flatten_chain(chain, "call", target_exp=exp)
        if calls.empty:
            continue

        # Choose calls in target delta window
        window = calls.dropna(subset=["delta"]).copy()
        window = window[(window["delta"]>=target_delta_low) & (window["delta"]<=target_delta_high)]
        if window.empty:
            # fallback: pick OTM by strike just above last
            window = calls[calls["strike"] > last].copy()
        if window.empty:
            continue

        # Pick the best by annualized ROI = credit / (lot_cost) annualized
        window["credit"] = window["mid"].fillna(window["last"]).fillna(0) * 100
        window["roi_ann"] = window["credit"].apply(lambda c: annualize_return(c, lot_cost, dte))
        window["pop_est"] = (1 - window["delta"].abs()).clip(0,1)

        best = window.sort_values(["roi_ann","pop_est"], ascending=[False, False]).head(1)
        if best.empty:
            continue

        b = best.iloc[0].to_dict()
        b.update({
            "under_price": last,
            "lot_cost": lot_cost,
            "dte": dte,
        })
        results.append(b)

    if not results:
        return pd.DataFrame(columns=["underlying","under_price","expiration","dte","strike","mid","credit","pop_est","roi_ann","lot_cost"])
    df = pd.DataFrame(results)
    # Select/rename
    view = df[["underlying","under_price","expiration","dte","strike","mid","credit","pop_est","roi_ann","lot_cost"]].copy()
    view = view.sort_values(["roi_ann","pop_est"], ascending=[False, False])
    return view

def scan_put_credit_spreads(quotes_df, budget, width_choices=(1,2), min_dte=7, max_dte=14, target_delta_low=0.20, target_delta_high=0.30):
    rows = []
    today = dt.date.today()
    for symbol, last in quotes_df[["symbol","last"]].itertuples(index=False):
        chain = get_option_chain(symbol)
        exps = [x.get("expirationDate") for x in chain]
        exp, dte = pick_nearest_weekly(exps, min_dte, max_dte, today=today)
        if not exp:
            continue

        puts = flatten_chain(chain, "put", target_exp=exp).dropna(subset=["strike"])
        if puts.empty:
            continue

        # Candidates around target delta
        window = puts.dropna(subset=["delta"]).copy()
        window["abs_delta"] = window["delta"].abs()
        window = window[(window["abs_delta"]>=target_delta_low) & (window["abs_delta"]<=target_delta_high)]
        if window.empty:
            # fallback: pick OTM below last
            window = puts[puts["strike"] < last].copy()
        if window.empty:
            continue

        for _, short in window.iterrows():
            for w in width_choices:
                long_strike = round(short["strike"] - w, 2)
                long = puts.iloc[(puts["strike"] - long_strike).abs().argsort()[:1]].copy()
                if long.empty:
                    continue
                long = long.iloc[0]

                short_mid = (short["bid"] + short["ask"])/2 if pd.notna(short["bid"]) and pd.notna(short["ask"]) else short.get("last", np.nan)
                long_mid  = (long["bid"] + long["ask"])/2 if pd.notna(long["bid"]) and pd.notna(long["ask"]) else long.get("last", np.nan)
                if pd.isna(short_mid) or pd.isna(long_mid):
                    continue

                credit = max(short_mid - long_mid, 0) * 100
                width_dollars = max(short["strike"] - long["strike"], 0) * 100
                max_risk = max(width_dollars - credit, 0)
                if max_risk <= 0:
                    continue
                if max_risk > budget:
                    # Too large for a single spread given budget; skip (or size down to 0)
                    continue

                # Delta-based POP approximation (very rough): POP ≈ 1 - |short delta|
                pop_est = (1 - abs(short.get("delta", 0))) if pd.notna(short.get("delta", np.nan)) else np.nan
                roi_ann = annualize_return(credit, max_risk, dte)

                rows.append({
                    "underlying": symbol,
                    "under_price": last,
                    "expiration": exp,
                    "dte": dte,
                    "short_strike": short["strike"],
                    "long_strike": long["strike"],
                    "width": width_dollars/100.0,
                    "credit": credit,
                    "max_risk": max_risk,
                    "return_on_risk": credit / max_risk if max_risk>0 else np.nan,
                    "roi_ann": roi_ann,
                    "pop_est": np.nan if pd.isna(pop_est) else max(min(pop_est,1),0)
                })

    if not rows:
        return pd.DataFrame(columns=["underlying","under_price","expiration","dte","short_strike","long_strike","width","credit","max_risk","return_on_risk","roi_ann","pop_est"])

    df = pd.DataFrame(rows).sort_values(["roi_ann","pop_est"], ascending=[False, False])
    return df

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="Weekly Options Scanner", page_icon="📈", layout="wide")
st.title("📈 Weekly Covered Calls & Put Credit Spreads (FMP)")

with st.sidebar:
    st.header("Settings")
    budget = st.number_input("Investment Budget ($)", min_value=500, max_value=100000, value=4000, step=100)
    min_dte = st.slider("Min DTE", 0, 30, 7)
    max_dte = st.slider("Max DTE", 1, 45, 14)
    st.caption("Weekly trades typically use 7–14 DTE.")

    st.subheader("Universe")
    default_watchlist = "AAPL,MSFT,AMD,SOFI,F,PBR,NU,PLTR,CCL,NIO,T,PARA,UAL,DIS,AI,SNAP,KO,INTC,ENVX,RIVN"
    wl = st.text_area("Tickers (comma-separated)", value=default_watchlist, height=100)
    tickers = [t.strip().upper() for t in wl.split(",") if t.strip()]

    st.subheader("Target Deltas")
    cc_low, cc_high = st.slider("Covered Call delta range", 0.05, 0.8, (0.20, 0.35))
    pc_low, pc_high = st.slider("Put Credit Spread short-put delta", 0.05, 0.8, (0.20, 0.30))

    width_choices = st.multiselect("PCS width (points)", options=[0.5,1,2,3,5], default=[1,2])
    if not width_choices:
        width_choices = [1,2]

    run_btn = st.button("Run Scan")

if not FMP_KEY:
    st.error("Add your FMP API key to `.streamlit/secrets.toml` under [fmp].")
    st.stop()

if run_btn:
    st.info("Fetching quotes…")
    quotes = get_quote_batch(tickers)
    if quotes.empty:
        st.error("No quotes returned. Check tickers.")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Covered Calls (Weekly)")
        # Filter affordable names for 100 shares
        affordable = quotes[quotes["last"] * 100 <= budget].copy()
        if affordable.empty:
            st.warning("No tickers affordable for covered calls with the current budget.")
            cc_df = pd.DataFrame()
        else:
            cc_df = scan_covered_calls(
                affordable, budget, min_dte=min_dte, max_dte=max_dte,
                target_delta_low=cc_low, target_delta_high=cc_high
            )
        if not cc_df.empty:
            st.dataframe(cc_df.reset_index(drop=True))
            st.download_button("Download Covered Calls CSV", cc_df.to_csv(index=False), "covered_calls.csv", "text/csv")
        else:
            st.write("—")

    with col2:
        st.subheader("Put Credit Spreads (Weekly)")
        pcs_df = scan_put_credit_spreads(
            quotes, budget, width_choices=width_choices, min_dte=min_dte, max_dte=max_dte,
            target_delta_low=pc_low, target_delta_high=pc_high
        )
        if not pcs_df.empty:
            st.dataframe(pcs_df.reset_index(drop=True))
            st.download_button("Download Put Credit Spreads CSV", pcs_df.to_csv(index=False), "put_credit_spreads.csv", "text/csv")
        else:
            st.write("—")

    st.caption("POP ≈ 1 − |delta| (heuristic). ROI_ANN annualizes return by DTE. Always validate in your broker.")
else:
    st.caption("Set your tickers and press **Run Scan** to begin.")
