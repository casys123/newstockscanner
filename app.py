import os, math, time, requests, datetime as dt, io
import pandas as pd
import numpy as np
import streamlit as st

FMP_KEY = st.secrets.get("fmp", {}).get("api_key", os.getenv("FMP_API_KEY", ""))
BASE = "https://financialmodelingprep.com/api/v3"

def fmp_get(path, params=None):
    if params is None: params = {}
    params["apikey"] = FMP_KEY
    r = requests.get(f"{BASE}/{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def annualize_return(credit, collateral, dte):
    if collateral <= 0 or dte <= 0:
        return 0.0
    return (1 + credit/collateral) ** (365/dte) - 1

def pick_nearest_weekly(exps, min_dte=7, max_dte=14):
    today = dt.date.today()
    best = None
    for e in exps:
        try:
            d = dt.datetime.strptime(e, "%Y-%m-%d").date()
            dte = (d - today).days
            if min_dte <= dte <= max_dte:
                if not best or dte < best[1]:
                    best = (e, dte)
        except:
            continue
    return best if best else (None, None)

def get_quote_batch(tickers):
    data = fmp_get("quote/" + ",".join(tickers))
    df = pd.DataFrame(data)
    if df.empty: return pd.DataFrame(columns=["symbol","price","volAvg","beta"])
    df = df.rename(columns={"symbol":"ticker","price":"last"})
    return df[["ticker","last","volAvg","beta"]]

def get_hist_vol(symbol):
    try:
        js = fmp_get(f"historical-volatility/{symbol}?limit=1")
        if isinstance(js, list) and js:
            return js[0].get("volatility", np.nan)
    except:
        pass
    return np.nan

def get_option_chain(symbol):
    try:
        return fmp_get(f"option-chain/{symbol}")
    except Exception as e:
        st.warning(f"Options missing for {symbol}: {e}")
        return []

def flatten_chain(chain, call_or_put="call", exp=None):
    rows=[]
    for c in chain:
        expiration=c.get("expirationDate")
        if exp and expiration!=exp: continue
        for o in c.get("options",[]):
            if o.get("optionType")!=call_or_put: continue
            rows.append({
                "underlying":o.get("underlyingSymbol"),
                "type":call_or_put,
                "expiration":expiration,
                "strike":o.get("strike"),
                "bid":o.get("bid"),
                "ask":o.get("ask"),
                "last":o.get("last"),
                "delta":o.get("delta"),
                "iv":o.get("impliedVolatility")
            })
    df=pd.DataFrame(rows)
    if df.empty: return df
    df["mid"]=(df["bid"].fillna(0)+df["ask"].fillna(0))/2
    return df

def scan_covered_calls(quotes, budget, min_dte, max_dte, d_low, d_high):
    res=[]
    for t,last,vol,beta in quotes[["ticker","last","volAvg","beta"]].itertuples(index=False):
        if last*100>budget: continue
        ch=get_option_chain(t)
        exps=[x.get("expirationDate") for x in ch]
        exp,dte=pick_nearest_weekly(exps,min_dte,max_dte)
        if not exp: continue
        df=flatten_chain(ch,"call",exp)
        df=df[(df["delta"]>=d_low)&(df["delta"]<=d_high)]
        if df.empty: continue
        df["credit"]=df["mid"]*100
        df["roi_ann"]=df.apply(lambda r:annualize_return(r["credit"],last*100,dte),axis=1)
        df["pop"]=1-abs(df["delta"])
        best=df.sort_values("roi_ann",ascending=False).head(1)
        if not best.empty:
            b=best.iloc[0].to_dict()
            b.update({"under_price":last,"lot_cost":last*100,"dte":dte,"volAvg":vol,"beta":beta})
            res.append(b)
    if not res: return pd.DataFrame()
    return pd.DataFrame(res)

def scan_put_spreads(quotes,budget,min_dte,max_dte,d_low,d_high,widths):
    res=[]
    for t,last,vol,beta in quotes[["ticker","last","volAvg","beta"]].itertuples(index=False):
        ch=get_option_chain(t)
        exps=[x.get("expirationDate") for x in ch]
        exp,dte=pick_nearest_weekly(exps,min_dte,max_dte)
        if not exp: continue
        df=flatten_chain(ch,"put",exp)
        df=df[(df["delta"]<=-d_low)&(df["delta"]>=-d_high)]
        if df.empty: continue
        for _,sp in df.iterrows():
            for w in widths:
                long_strike=sp["strike"]-w
                lp=df.iloc[(df["strike"]-long_strike).abs().argsort()[:1]]
                if lp.empty: continue
                short_mid=sp["mid"]; long_mid=lp["mid"].values[0]
                credit=(short_mid-long_mid)*100
                width=w*100
                risk=max(width-credit,0)
                if risk<=0 or risk>budget: continue
                roi=annualize_return(credit,risk,dte)
                res.append({
                    "underlying":t,"under_price":last,"expiration":exp,"dte":dte,
                    "short_strike":sp["strike"],"long_strike":long_strike,"width":w,
                    "credit":credit,"max_risk":risk,
                    "return_on_risk":credit/risk,"roi_ann":roi,
                    "pop":1-abs(sp["delta"]),"volAvg":vol,"beta":beta
                })
    if not res: return pd.DataFrame()
    return pd.DataFrame(res)

st.set_page_config(page_title="Volatility Options Scanner", page_icon="📈", layout="wide")
st.title("📈 Weekly Covered Calls & Put Credit Spreads — Volatility Focus")

with st.sidebar:
    st.header("Settings")
    budget=st.number_input("Investment Budget ($)",500,100000,4000,100)
    min_dte,max_dte=st.slider("DTE Range",0,30,(7,14))
    d_low,d_high=st.slider("Call Delta Range",0.05,0.8,(0.20,0.35))
    p_low,p_high=st.slider("Put Delta Range",0.05,0.8,(0.20,0.30))
    widths=st.multiselect("Spread Widths",options=[0.5,1,2,3,5],default=[1,2])
    wl=st.text_area("Tickers", "AAPL,AMD,SOFI,PLTR,CCL,NIO,AI,TSLA,RIVN,UPST,SMCI,COIN")
    run=st.button("Run Scan")

if not FMP_KEY:
    st.error("Missing FMP API key in .streamlit/secrets.toml")
    st.stop()

if run:
    tickers=[t.strip().upper() for t in wl.split(",") if t.strip()]
    st.info("Fetching quotes and volatility...")
    quotes=get_quote_batch(tickers)
    if quotes.empty: st.error("No quotes found."); st.stop()
    quotes["hist_vol"]=quotes["ticker"].apply(get_hist_vol)
    quotes["vol_rank"]=quotes["hist_vol"].rank(ascending=False)
    st.subheader("🔝 Top Volatile Stocks")
    st.dataframe(quotes.sort_values("hist_vol",ascending=False).head(10))

    col1,col2=st.columns(2)
    with col1:
        st.subheader("Covered Calls (High Volatility = High Premium)")
        cc=scan_covered_calls(quotes,budget,min_dte,max_dte,d_low,d_high)
        if not cc.empty:
            st.dataframe(cc)
            buffer=io.BytesIO()
            with pd.ExcelWriter(buffer,engine="xlsxwriter") as writer:
                cc.to_excel(writer,index=False,sheet_name="CoveredCalls")
            st.download_button("⬇️ Download Excel",buffer.getvalue(),"covered_calls.xlsx")
        else: st.write("—")

    with col2:
        st.subheader("Put Credit Spreads (Volatility-Adjusted)")
        pcs=scan_put_spreads(quotes,budget,min_dte,max_dte,p_low,p_high,widths)
        if not pcs.empty:
            st.dataframe(pcs)
            buffer2=io.BytesIO()
            with pd.ExcelWriter(buffer2,engine="xlsxwriter") as writer:
                pcs.to_excel(writer,index=False,sheet_name="PutSpreads")
            st.download_button("⬇️ Download Excel",buffer2.getvalue(),"put_spreads.xlsx")
        else: st.write("—")
