#!/usr/bin/env python3
"""
Trading.py (V9 + live monitoring & mode semantics)
See in-file docstring for details.
"""
# [The full code content is below; trimmed intro to avoid tool size issues]
import os, sys, math, json, hashlib, warnings, argparse
from datetime import datetime, timedelta, timezone
from functools import lru_cache
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    import yfinance as yf
except Exception:
    print("Please: pip install yfinance"); sys.exit(1)
try:
    import pandas_ta as ta
except Exception:
    print("Please: pip install pandas_ta"); sys.exit(1)

# ===================== USER CONFIG =====================
INPUT_CSV  = "Ticker Universe.csv"
NOVICE_CSV = "Ticker_Tech_Indicators_V9_Novice.csv"
EXPERT_CSV = "Ticker_Tech_Indicators_V9_Expert.csv"
OUT_XLSX   = "Ticker_Tech_Indicators_V9.xlsx"
HIST_DAYS  = 300
HIT_LOOK_D = 60
CACHE_DIR  = ".cache_ti"
TTL_HIST_H = 12
TTL_INFO_H = 24
INDEX_TICK = "SPY"
VIX_TICK   = "^VIX"
VIX_MAX    = 25.0

K_ZONE_DEFAULT      = 0.4
VOL_ZONE_THRESH_LO  = 1.5
VOL_ZONE_THRESH_HI  = 4.0

def _normalize_text(s: str) -> str:
    if s is None: return ""
    repl = {"—":"-", "–":"-", "•":"-", "’":"'", "“":"\"", "”":"\"", "→":"->", "←":"<-","≥":">=", "≤":"<=", "…":"...", "✔":"[ok]", "✖":"[x]","⋅":"-", "·":"-","€":"EUR", "£":"GBP"}
    out = "".join(repl.get(ch, ch) for ch in str(s))
    return " ".join(out.split())

def safe_float(x):
    try: return float(x)
    except Exception: return np.nan

def pct(a, b):
    if b in (0, None) or pd.isna(a) or pd.isna(b) or b == 0: return np.nan
    return (a / b - 1.0) * 100.0

def pct_dist(base, target):
    if base in (0, None) or pd.isna(base) or pd.isna(target): return np.nan
    return (target - base) / base * 100.0

def _empirical_hit_rate(series_hits, min_n=40, lo=0.35, hi=0.85):
    s = pd.Series(series_hits).dropna()
    if len(s) < min_n: return np.nan
    raw = s.mean()
    return float(max(lo, min(hi, raw)))

def bars_since_cross(e8, e34):
    e8 = pd.Series(e8); e34 = pd.Series(e34)
    diff = e8 - e34; sign = np.sign(diff); ch = sign.diff()
    cross_idx = np.where(ch != 0)[0]
    return int(len(sign) - 1 - cross_idx[-1]) if cross_idx.size > 0 else np.nan

def _fmt_price(x):
    if pd.isna(x): return ""
    try:
        if abs(x) >= 1000: return f"{x:,.2f}"
        if abs(x) >= 100:  return f"{x:,.2f}"
        if abs(x) >= 1:    return f"{x:.2f}"
        return f"{x:.4f}"
    except Exception:
        return str(x)

def _is_leveraged_symbol(tkr: str) -> bool:
    t = tkr.upper()
    lever_keys = ("TQQQ","SQQQ","UPRO","SPXL","SPXS","SOXL","SOXS","FNGU","TSLL","TSLS","LABU","LABD")
    return any(k == t or k in t for k in lever_keys)

def _is_crypto(tkr: str, desc: str) -> bool:
    t = tkr.upper()
    return t.endswith("-USD") or (desc or "").strip().lower() == "crypto"

ETF_WHITELIST = {"FBTC","IBIT","BITB","ARKB","HODL"}
def _is_mutual_fund_or_401k(tkr: str, desc: str) -> bool:
    if tkr.upper() in ETF_WHITELIST: return False
    mf_list = {"AMCPX","FCNTX","JGACX","BBUS"}
    if tkr.upper() in mf_list: return True
    d = (desc or "").strip().lower()
    return d in {"401k","mutual fund"}

def _is_etf(tkr: str, desc: str, yi: dict) -> bool:
    if _is_crypto(tkr, desc): return False
    if _is_mutual_fund_or_401k(tkr, desc): return False
    d = (desc or "").lower()
    if d == "etf": return True
    if tkr.upper() in ETF_WHITELIST: return True
    qt = (yi or {}).get("quoteType", "").lower()
    if qt == "etf": return True
    if tkr.upper() in {"QQQ","VTI","VOO","VUG","VGT","SMH","FDIS","SCHX"}: return True
    return False

def _cache_path(kind: str, ticker: str):
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = f"{kind}_{ticker.upper()}"; import hashlib; h = hashlib.sha1(key.encode()).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"{kind}_{ticker.upper()}_{h}.json" if kind=="info" else f"{kind}_{ticker.upper()}_{h}.csv")

def _is_fresh(path: str, ttl_hours: int) -> bool:
    if not os.path.exists(path): return False
    mtime = datetime.fromtimestamp(os.path.getmtime(path), timezone.utc)
    return (datetime.now(timezone.utc) - mtime) < timedelta(hours=ttl_hours)

def _save_csv(path: str, df: pd.DataFrame):
    try: df.to_csv(path)
    except Exception: pass

def _load_csv(path: str):
    try: return pd.read_csv(path, index_col=0, parse_dates=True)
    except Exception: return None

def _save_json(path: str, obj: dict):
    try:
        with open(path, "w", encoding="utf-8") as f: json.dump(obj, f)
    except Exception: pass

def _load_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f: return json.load(f)
    except Exception: return None

@lru_cache(maxsize=256)
def _yf_hist_daily(ticker: str, period_days: int, adjusted: bool = True, ttl_hours: int = TTL_HIST_H):
    kind = "hist_adj" if adjusted else "hist_raw"
    cache_p = _cache_path(kind, ticker)
    if _is_fresh(cache_p, ttl_hours):
        dfc = _load_csv(cache_p)
        if isinstance(dfc, pd.DataFrame) and not dfc.empty:
            return dfc
    end = datetime.now(timezone.utc); start = end - timedelta(days=period_days + 10)
    try:
        df = yf.download(ticker, start=start.date(), end=end.date(), interval="1d", auto_adjust=adjusted, progress=False)
    except Exception:
        df = pd.DataFrame()
    if isinstance(df, pd.DataFrame) and not df.empty:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [" ".join([c for c in tup if c]).strip() for tup in df.columns.values]
        rename_map = {}
        for c in df.columns:
            base = c.split(" ")[0] if isinstance(c, str) else str(c)
            if base in {"Open","High","Low","Close","Adj","AdjClose"}:
                rename_map[c] = {"Adj":"Adj Close", "AdjClose":"Adj Close"}.get(base, base)
        if rename_map: df = df.rename(columns=rename_map)
        keep = [c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in df.columns]
        df = df[keep].copy()
        df.index = pd.to_datetime(df.index)
        for col in keep: df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(how="any")
        _save_csv(cache_p, df)
    else:
        df = pd.DataFrame()
    return df

@lru_cache(maxsize=256)
def _yf_info(ticker: str):
    cache_p = _cache_path("info", ticker)
    if _is_fresh(cache_p, TTL_INFO_H):
        j = _load_json(cache_p)
        if isinstance(j, dict): return j
    try:
        t = yf.Ticker(ticker); info = t.info or {}
    except Exception:
        info = {}
    _save_json(cache_p, info); return info

def _yf_intraday_snapshot(ticker: str, prepost=True):
    try:
        df = yf.download(ticker, period="2d", interval="1m", prepost=prepost, auto_adjust=False, progress=False)
    except Exception:
        df = pd.DataFrame()
    if df is None or df.empty:
        return {"last": np.nan, "high": np.nan, "low": np.nan, "asof": None, "session": "n/a", "delay_min": np.nan}
    now_utc = datetime.now(timezone.utc)
    try:
        last_ts = df.index[-1]
        last_row = df.iloc[-1]
        dft = df[df.index.date == now_utc.date()]
        day_high = float(dft["High"].max()) if not dft.empty else float(df["High"].iloc[-1])
        day_low  = float(dft["Low"].min()) if not dft.empty else float(df["Low"].iloc[-1])
        last = float(last_row["Close"]) if "Close" in df.columns else np.nan
    except Exception:
        return {"last": np.nan, "high": np.nan, "low": np.nan, "asof": None, "session": "n/a", "delay_min": np.nan}
    try:
        delay_min = (now_utc - last_ts.tz_convert(timezone.utc)).total_seconds() / 60.0
    except Exception:
        delay_min = (now_utc - last_ts.to_pydatetime().replace(tzinfo=timezone.utc)).total_seconds() / 60.0
    try:
        hhmm = int(last_ts.tz_convert("US/Eastern").strftime("%H%M"))
    except Exception:
        hhmm = int(datetime.now().astimezone().strftime("%H%M"))
    session = "pre" if hhmm < 930 else ("post" if hhmm >= 1600 else "regular")
    return {"last": last, "high": day_high, "low": day_low, "asof": last_ts.isoformat(), "session": session, "delay_min": float(delay_min)}

def compute_indicators(df: pd.DataFrame):
    out = {}
    if df is None or df.empty or len(df) < 50:
        return out, df
    df = df.copy()
    df["SMA21"] = ta.sma(df["Close"], length=21)
    df["EMA8"]  = ta.ema(df["Close"], length=8)
    df["EMA34"] = ta.ema(df["Close"], length=34)
    df["EMA21"] = ta.ema(df["Close"], length=21)
    df["RSI14"] = ta.rsi(df["Close"], length=14)
    df["RSI_EMA21"] = ta.ema(df["RSI14"], length=21)
    macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)
    df["MACD_Hist"] = macd["MACDh_12_26_9"] if macd is not None and "MACDh_12_26_9" in macd.columns else np.nan
    adx = ta.adx(df["High"], df["Low"], df["Close"], length=14)
    df["ADX14"] = adx["ADX_14"] if adx is not None and "ADX_14" in adx.columns else np.nan
    df["ATR14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    df["ATRpct"]= (df["ATR14"] / df["Close"]) * 100.0
    if "Volume" in df.columns:
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
        vol20 = ta.sma(df["Volume"], length=20)
        df["VolRatio_20d"] = df["Volume"] / vol20
        chg = df["Close"].diff()
        up_vol   = np.where(chg > 0, df["Volume"], 0.0)
        down_vol = np.where(chg < 0, df["Volume"], 0.0)
        udvr_up = pd.Series(up_vol, index=df.index).rolling(20).sum()
        udvr_dn = pd.Series(down_vol, index=df.index).rolling(20).sum().replace(0, np.nan)
        df["UDVR_20d"] = udvr_up / udvr_dn
        vol20_mean = ta.sma(df["Volume"], length=20)
        vol20_std  = ta.stdev(df["Volume"], length=20)
        df["VolZ_20"] = (df["Volume"] - vol20_mean) / vol20_std
    else:
        df["VolRatio_20d"] = np.nan; df["UDVR_20d"] = np.nan; df["VolZ_20"] = np.nan
    df["HH20"] = df["Close"].rolling(20).max()
    df["LL20"] = df["Close"].rolling(20).min()
    df["Breakout20"]  = df["Close"] > df["HH20"].shift(1)
    df["Breakdown20"] = df["Close"] < df["LL20"].shift(1)
    def _bsc(e8, e34):
        e8 = pd.Series(e8); e34 = pd.Series(e34); diff = e8 - e34; sign = np.sign(diff); ch = sign.diff()
        cross_idx = np.where(ch != 0)[0]; return int(len(sign)-1-cross_idx[-1]) if cross_idx.size>0 else np.nan
    bsc = _bsc(df["EMA8"].values, df["EMA34"].values)
    out["Signal_Age"]  = bsc; out["Fresh_Cross"] = (not pd.isna(bsc)) and (bsc <= 3)
    out["EMA_Regime"] = "Buy" if df["EMA8"].iloc[-1] >= df["EMA34"].iloc[-1] else "Sell"
    w = df.resample("W-FRI").agg({k:("first" if k=="Open" else "max" if k=="High" else "min" if k=="Low" else "last") for k in ["Open","High","Low","Close"]}).dropna()
    if len(w) >= 10:
        w["EMA8w"]  = ta.ema(w["Close"], length=8)
        w["EMA34w"] = ta.ema(w["Close"], length=34)
        out["Weekly_Regime"] = "Buy" if w["EMA8w"].iloc[-1] >= w["EMA34w"].iloc[-1] else "Sell"
    else:
        out["Weekly_Regime"] = "Neutral"
    h_52w = df["High"].rolling(252, min_periods=50).max().iloc[-1]
    l_52w = df["Low"].rolling(252, min_periods=50).min().iloc[-1]
    px    = float(df["Close"].iloc[-1])
    out["Close"] = px; out["Open"] = float(df["Open"].iloc[-1]); out["High"] = float(df["High"].iloc[-1]); out["Low"] = float(df["Low"].iloc[-1])
    out["SMA21"] = float(df["SMA21"].iloc[-1]); out["EMA_Short"]= float(df["EMA8"].iloc[-1]); out["EMA_Long"] = float(df["EMA34"].iloc[-1]); out["EMA21"]= float(df["EMA21"].iloc[-1])
    out["RSI(14)"]= float(df["RSI14"].iloc[-1]); out["MACD_Hist_Pos"]= bool(df["MACD_Hist"].iloc[-1] > 0) if pd.notna(df["MACD_Hist"].iloc[-1]) else False
    out["ADX(14)"]= float(df["ADX14"].iloc[-1]) if pd.notna(df["ADX14"].iloc[-1]) else np.nan
    out["ATR"]= float(df["ATR14"].iloc[-1]); out["ATR%"]= float(df["ATRpct"].iloc[-1])
    out["Volume"]= float(df["Volume"].iloc[-1]) if ("Volume" in df.columns and pd.notna(df["Volume"].iloc[-1])) else np.nan
    out["UDVR(20d)"]= float(df["UDVR_20d"].iloc[-1]) if pd.notna(df["UDVR_20d"].iloc[-1]) else np.nan
    out["VolRatio_20d"]= float(df["VolRatio_20d"].iloc[-1]) if pd.notna(df["VolRatio_20d"].iloc[-1]) else np.nan
    out["VolZ_20"]= float(df["VolZ_20"].iloc[-1]) if pd.notna(df["VolZ_20"].iloc[-1]) else np.nan
    out["Breakout20"]= bool(df["Breakout20"].iloc[-1]) if pd.notna(df["Breakout20"].iloc[-1]) else False
    out["Breakdown20"]= bool(df["Breakdown20"].iloc[-1]) if pd.notna(df["Breakdown20"].iloc[-1]) else False
    out["High_52W"]= float(h_52w) if pd.notna(h_52w) else np.nan
    out["Low_52W"]= float(l_52w) if pd.notna(l_52w) else np.nan
    out["%From_21SMA"]= pct(out["Close"], out["SMA21"])
    if pd.notna(out["ATR"]) and out["ATR"] > 0 and pd.notna(df["HH20"].iloc[-1]):
        out["BreakoutStrength_ATR"] = (out["Close"] - float(df["HH20"].iloc[-1])) / out["ATR"]
    else:
        out["BreakoutStrength_ATR"] = np.nan
    hits_today, hits_week = [], []
    look = min(HIT_LOOK_D, len(df) - 2)
    if look > 10 and pd.notna(out["ATR"]) and out["ATR"] > 0:
        for i in range(look, 1, -1):
            c = float(df["Close"].iloc[-i]); atr = float(df["ATR14"].iloc[-i]); hi = float(df["High"].iloc[-i+1])
            hits_today.append(1.0 if hi >= c + atr else 0.0)
        for i in range(look, 6, -1):
            c = float(df["Close"].iloc[-i]); atr = float(df["ATR14"].iloc[-i])
            window_high = float(df["High"].iloc[-i+1:-i+6].max())
            hits_week.append(1.0 if window_high >= c + atr else 0.0)
    out["HitToday_Up1ATR %"] = 100.0 * _empirical_hit_rate(hits_today, min_n=40)
    out["HitWeek_Up1ATR %"]  = 100.0 * _empirical_hit_rate(hits_week,  min_n=8)
    return out, df

def _pivot_highs(series: pd.Series, lookback=5, tol=1e-6):
    s = series; highs = []
    for i in range(lookback, len(s)-lookback):
        window = s.iloc[i-lookback:i+lookback+1]
        if (window.max() - s.iloc[i]) <= tol:
            highs.append((s.index[i], s.iloc[i]))
    return highs

def _pivot_lows(series: pd.Series, lookback=5, tol=1e-6):
    s = series; lows = []
    for i in range(lookback, len(s)-lookback):
        window = s.iloc[i-lookback:i+lookback+1]
        if (s.iloc[i] - window.min()) <= tol:
            lows.append((s.index[i], s.iloc[i]))
    return lows

def _zone_k_from_atrp(atrp: float) -> float:
    if pd.isna(atrp): return K_ZONE_DEFAULT
    if atrp < VOL_ZONE_THRESH_LO: return 0.30
    if atrp < VOL_ZONE_THRESH_HI: return 0.40
    return 0.50

def compute_resistance_targets(df_full: pd.DataFrame, atr: float, last_close: float, zone_k: float):
    res = {"Breakout_Base": np.nan, "Res1_Pivot": np.nan, "Res1_Zone_Lo": np.nan, "Res1_Zone_Hi": np.nan, "BO_T1": np.nan, "BO_T2": np.nan, "At_Res1_Zone": False}
    if df_full is None or df_full.empty or pd.isna(atr) or atr <= 0: return res
    df = df_full.copy()
    hh20_yest = df["Close"].rolling(20).max().shift(1).iloc[-1]
    last_close = float(df["Close"].iloc[-1]) if pd.isna(last_close) else float(last_close)
    is_fresh_bo = (last_close > hh20_yest) if pd.notna(hh20_yest) else False
    base = hh20_yest if is_fresh_bo else last_close
    res["Breakout_Base"] = base
    piv = _pivot_highs(df["High"].tail(180), lookback=3, tol=1e-6)
    if piv:
        above = [p for (_,p) in piv if p >= base]
        if above:
            nearest = min(above, key=lambda v: abs(v - base))
            res["Res1_Pivot"] = nearest
            res["Res1_Zone_Lo"] = nearest - zone_k*atr
            res["Res1_Zone_Hi"] = nearest + zone_k*atr
    res["BO_T1"] = base + 1.0*atr
    res["BO_T2"] = base + 2.0*atr
    if pd.notna(res["Res1_Zone_Lo"]) and pd.notna(res["Res1_Zone_Hi"]):
        px = last_close
        res["At_Res1_Zone"] = (res["Res1_Zone_Lo"] <= px <= res["Res1_Zone_Hi"])
    return res

def _anchored_vwap_from_timestamp(df: pd.DataFrame, anchor_ts):
    try:
        if anchor_ts is None or ("Volume" not in df.columns):
            return np.nan
        s = df.loc[df.index >= pd.to_datetime(anchor_ts)]
        if s.empty: return np.nan
        vol = pd.to_numeric(s["Volume"], errors="coerce")
        if vol.isna().all() or float(vol.sum()) == 0.0: return np.nan
        num = (s["Close"] * vol).sum(); den = vol.sum()
        x = float(num / den) if den != 0 else np.nan
        return x
    except Exception:
        return np.nan

def ladder_profile(ticker, desc, atr_pct):
    is_crypto = _is_crypto(ticker, desc)
    is_lev    = _is_leveraged_symbol(ticker)
    v = atr_pct if pd.notna(atr_pct) else 2.0
    if v < 1.5:  vol = "low"
    elif v < 4:  vol = "med"
    else:        vol = "high"
    if _is_etf(ticker, desc, {}) and not is_lev:
        prof = {"low": (0.4,0.8,1.2), "med": (0.5,1.0,1.5), "high": (0.6,1.2,1.8)}[vol]
    elif is_crypto or is_lev:
        prof = {"low": (0.6,1.2,1.8), "med": (0.8,1.5,2.25), "high": (1.0,2.0,3.0)}[vol]
    else:
        prof = {"low": (0.5,1.0,1.5), "med": (0.6,1.1,1.8), "high": (0.8,1.4,2.2)}[vol]
    return prof

def get_asset_profile(ticker: str, desc: str, atrp: float) -> dict:
    """Return volatility/asset-aware parameters.
    Keys: trail_k, ladder_scale, gap_atr_th, volratio_th, rsi_gate,
          class_label, class_detail, vol_band
    """
    cls = "stock"
    is_etf = _is_etf(ticker, desc, {})
    is_lev = _is_leveraged_symbol(ticker)
    is_crypto = _is_crypto(ticker, desc)
    if is_crypto:   cls = "crypto"
    elif is_lev:    cls = "levETF"
    elif is_etf:    cls = "etf"

    a = safe_float(atrp); a = 2.0 if pd.isna(a) else a
    params = {"trail_k": 1.0, "ladder_scale": 1.0, "gap_atr_th": 0.30, "volratio_th": 1.20, "rsi_gate": 45}
    band = "med"

    def pack(class_detail=None):
        return {**params, "class_label": cls, "class_detail": class_detail or cls, "vol_band": band}

    # ETFs
    if cls == "etf":
        if a < 1.5: params.update(trail_k=0.9,  ladder_scale=1.00, gap_atr_th=0.25, volratio_th=1.10, rsi_gate=45); band="low"
        elif a < 4: params.update(trail_k=1.0,  ladder_scale=0.95, gap_atr_th=0.30, volratio_th=1.15, rsi_gate=45); band="med"
        else:       params.update(trail_k=1.1,  ladder_scale=0.90, gap_atr_th=0.35, volratio_th=1.20, rsi_gate=47); band="high"
        return pack(f"etf_{band}")

    # Leveraged ETFs
    if cls == "levETF":
        if a < 1.5: params.update(trail_k=1.25, ladder_scale=0.90, gap_atr_th=0.35, volratio_th=1.25, rsi_gate=47); band="low"
        elif a < 4: params.update(trail_k=1.5,  ladder_scale=0.85, gap_atr_th=0.40, volratio_th=1.30, rsi_gate=50); band="med"
        else:       params.update(trail_k=1.75, ladder_scale=0.80, gap_atr_th=0.45, volratio_th=1.35, rsi_gate=52); band="high"
        return pack(f"levETF_{band}")

    # Crypto
    if cls == "crypto":
        if a < 4:   params.update(trail_k=1.5,  ladder_scale=0.90, gap_atr_th=0.40, volratio_th=1.30, rsi_gate=50); band="low"
        elif a < 8: params.update(trail_k=1.75, ladder_scale=0.85, gap_atr_th=0.45, volratio_th=1.35, rsi_gate=52); band="med"
        else:       params.update(trail_k=2.0,  ladder_scale=0.80, gap_atr_th=0.50, volratio_th=1.40, rsi_gate=55); band="high"
        return pack(f"crypto_{band}")

    # Single stocks (default)
    if a < 1.5:
        params.update(trail_k=1.0,  ladder_scale=1.00, gap_atr_th=0.30, volratio_th=1.20, rsi_gate=45); band="low"
        return pack("stock_low")
    elif a < 4:
        params.update(trail_k=1.0,  ladder_scale=0.95, gap_atr_th=0.35, volratio_th=1.25, rsi_gate=47); band="med"
        return pack("stock_med")
    else:
        params.update(trail_k=1.25, ladder_scale=0.85, gap_atr_th=0.40, volratio_th=1.30, rsi_gate=50); band="high"
        return pack("stock_high")


def compute_priority(row, is_etf: bool):
    trend = 0.0
    trend += 30.0 if row.get("EMA_Regime") == "Buy" else 0.0
    rsi = safe_float(row.get("RSI(14)"))
    if 50 <= rsi <= 70: trend += 6.0
    if rsi > 70:        trend += 3.0
    adx = safe_float(row.get("ADX(14)"))
    if adx >= 15: trend += 4.0
    if bool(row.get("MACD_Hist_Pos")): trend += 5.0
    sent = 0.0
    vr   = safe_float(row.get("VolRatio_20d"))
    udvr = safe_float(row.get("UDVR(20d)"))
    if pd.notna(vr):
        if vr >= 1.10: sent += 10.0
        elif vr >= 1.00: sent += 6.0
    if pd.notna(udvr):
        if udvr >= 1.05: sent += 12.0
        elif udvr >= 1.00: sent += 8.0
    if bool(row.get("Breakout20")): sent += 6.0
    if bool(row.get("Breakdown20")): sent -= 6.0
    fund = 0.0
    if not is_etf and not _is_crypto(row.get("Ticker",""), row.get("Description","")):
        pe  = safe_float(row.get("PE"))
        eps = safe_float(row.get("EPS_Growth_Pct"))
        if pd.notna(pe):
            if pe <= 20:   fund += 8.0
            elif pe <= 30: fund += 4.0
        if pd.notna(eps):
            if eps >= 15:  fund += 8.0
            elif eps >= 5: fund += 4.0
    risk = 0.0
    atrp = safe_float(row.get("ATR%"))
    if pd.notna(atrp):
        if atrp <= 2.0:   risk += 10.0
        elif atrp <= 4.0: risk += 6.0
        else:             risk += 2.0
    score = 0.40*trend + 0.30*sent + 0.20*fund + 0.10*risk
    return round(max(0.0, min(100.0, score)), 1)

def compute_guardbias():
    try:
        idx = _yf_hist_daily(INDEX_TICK, 120, adjusted=True)
        vix = _yf_hist_daily(VIX_TICK, 120, adjusted=False)
        if idx is None or idx.empty or vix is None or vix.empty: return 0.0, np.nan
        e21 = ta.ema(idx["Close"], length=21).iloc[-1]
        bias = (0.5 if idx["Close"].iloc[-1] > e21 else -0.5)
        vix_last = float(vix["Close"].iloc[-1])
        if vix_last > VIX_MAX: bias -= 1.0
        return float(bias), vix_last
    except Exception:
        return 0.0, np.nan

def _resample(df: pd.DataFrame, rule: str):
    if df is None or df.empty: return df
    cols = ["Open","High","Low","Close"]
    agg = {"Open":"first","High":"max","Low":"min","Close":"last"}
    got = [c for c in cols if c in df.columns]
    out = df[got].resample(rule).agg({k: agg[k] for k in got}).dropna()
    if "Volume" in df.columns:
        v = df["Volume"].resample(rule).sum().dropna()
        out["Volume"] = v.reindex(out.index).fillna(0.0)
    return out

def _monthly_regime_and_ema21(df_full: pd.DataFrame):
    """Return (monthly_regime_label, EMA21m) from monthly resample of df_full."""
    try:
        df_m = _resample(df_full, "M")
        if df_m is None or df_m.empty or len(df_m) < 10:
            return "Neutral", np.nan
        ema21m = float(ta.ema(df_m["Close"], length=21).iloc[-1]) if len(df_m) >= 21 else np.nan
        if len(df_m) >= 34:
            r = "Buy" if ta.ema(df_m["Close"], length=8).iloc[-1] >= ta.ema(df_m["Close"], length=34).iloc[-1] else "Sell"
        else:
            r = "Neutral"
        return r, ema21m
    except Exception:
        return "Neutral", np.nan

def _earnings_proximity_days(ticker: str):
    try:
        t = yf.Ticker(ticker)
        cal = t.calendar
        if isinstance(cal, pd.DataFrame) and not cal.empty:
            if "Earnings Date" in cal.index:
                ed = pd.to_datetime(cal.loc["Earnings Date"].values[0])
                if not pd.isna(ed):
                    return int((ed.tz_localize("UTC").date() - datetime.now(timezone.utc).date()).days)
        edf = t.get_earnings_dates(limit=1)
        if isinstance(edf, pd.DataFrame) and not edf.empty:
            ed = pd.to_datetime(edf.index[-1])
            return int((ed.tz_localize("UTC").date() - datetime.now(timezone.utc).date()).days)
    except Exception:
        pass
    return None

def _novice_long_term_text(weekly_regime: str, EMA21w: float, EMA21d: float, px: float, atrp: float, res_lo: float, res_hi: float) -> str:
    band_txt = ""
    if pd.notna(res_lo) and pd.notna(res_hi):
        band_txt = f" Resistance band: ${_fmt_price(res_lo)}–${_fmt_price(res_hi)}."
    band_hint = ""
    if pd.notna(res_lo) and pd.notna(res_hi) and pd.notna(px):
        if px < res_lo: band_hint = " Price is below the band; prefer entries after reclaiming the lower band."
        elif res_lo <= px <= res_hi: band_hint = " Price is inside the band; prefer pullbacks within the band or a daily close above it."
        else: band_hint = " Price is above the band; let it pull back toward the band before adding."
    if pd.isna(EMA21w):
        return f"Long-term: insufficient weekly history; use the daily value-line at ${_fmt_price(EMA21d)} as a guide; avoid chasing." + band_txt + band_hint
    wk_gap = pct(px, EMA21w)
    stretch_th = max(5.0, atrp if pd.notna(atrp) else 0.0)
    stretched = (pd.notna(wk_gap) and wk_gap >= stretch_th)
    emaw_str = f"${_fmt_price(EMA21w)}"
    wr = (weekly_regime or "").lower()
    if wr == "buy":
        if stretched:
            return f"Long-term: trend up; add slowly on weakness while above weekly value-line {emaw_str}. Currently {wk_gap:+.1f}% above it—avoid adding when extended." + band_txt + band_hint
        return f"Long-term: average in while price is above weekly value-line {emaw_str}; prioritize dips; avoid chasing." + band_txt + band_hint
    elif wr == "neutral":
        return f"Long-term: wait for a weekly Buy or a reclaim of {emaw_str}; keep sizes small until then." + band_txt + band_hint
    else:
        return f"Long-term: avoid averaging down while weekly trend is down; prefer strength after reclaiming {emaw_str} or a weekly flip to Buy." + band_txt + band_hint

def main():
    ap = argparse.ArgumentParser(description="Vol-Adaptive Stock Screener V9 with live monitoring")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("-A","--after", action="store_true", help="Night-before review (prior close)")
    mode.add_argument("-P","--pre",   action="store_true", help="Pre-market review (prior close + annotations)")
    mode.add_argument("-L","--live",  action="store_true", help="Live monitor (intraday hits; no indicator recalcs)")
    ap.add_argument("--guard", choices=["OFF","SAFE","HARD"], default="OFF", help="Market guardrails via SPY/VIX")
    ap.add_argument("--no-xlsx", action="store_true", help="Do not create XLSX")
    ap.add_argument("--trade-session", choices=["RTH","EXT"], default="RTH", help="How to treat pre/post hits")
    args = ap.parse_args()
    DataMode = "AFTER"
    if args.pre: DataMode = "PRE"
    if args.live: DataMode = "LIVE"
    guard_mode = args.guard
    guard_bias, vix_last = compute_guardbias() if guard_mode in {"SAFE","HARD"} else (0.0, np.nan)
    print(f"[Guardrails] mode={guard_mode}, index={INDEX_TICK}, vix={VIX_TICK}, bias={guard_bias:.2f}, VIX={vix_last}")
    if not os.path.exists(INPUT_CSV):
        print(f"ERROR: '{INPUT_CSV}' not found."); sys.exit(1)
    uni = pd.read_csv(INPUT_CSV, usecols=["Ticker","Description"]).fillna("")
    uni["Ticker"] = uni["Ticker"].astype(str).str.strip()
    uni["Description"] = uni["Description"].astype(str).str.strip()

    novice_rows, expert_rows = [], []

    for _, r in uni.iterrows():
        t = r["Ticker"].upper(); d = r["Description"]
        if not t: continue
        print(f"Processing {t} ...")
        if _is_mutual_fund_or_401k(t, d):
            print(f"{t}: skipping (mutual fund/401K)"); continue
        df = _yf_hist_daily(t, HIST_DAYS, adjusted=True, ttl_hours=TTL_HIST_H)
        info = _yf_info(t)
        if df is None or df.empty or len(df) < 60:
            print(f"{t}: skipping (short/invalid data)"); continue
        ind, df_full = compute_indicators(df)
        if not ind:
            print(f"{t}: skipping (indicator error)"); continue
        snap = {"last": np.nan, "high": np.nan, "low": np.nan, "asof": None, "session": "n/a", "delay_min": np.nan}
        if DataMode in {"PRE","LIVE"}:
            snap = _yf_intraday_snapshot(t, prepost=True)

        px = ind["Close"]; emaD = ind.get("EMA_Regime","Sell")
        df_w = _resample(df_full, "W-FRI")
        if df_w is not None and not df_w.empty and len(df_w) >= 10:
            EMA21w = float(ta.ema(df_w["Close"], length=21).iloc[-1]) if len(df_w) >= 21 else np.nan
            weekly_regime = "Buy" if (
                len(df_w) >= 34
                and ta.ema(df_w["Close"], length=8).iloc[-1] >= ta.ema(df_w["Close"], length=34).iloc[-1]
            ) else ind.get("Weekly_Regime","Neutral")
        else:
            EMA21w = np.nan
            weekly_regime = ind.get("Weekly_Regime","Neutral")
        
        monthly_regime, EMA21m = _monthly_regime_and_ema21(df_full)
        ema21 = ind.get("EMA21"); atr = ind.get("ATR"); atrp = ind.get("ATR%")
        volratio = ind.get("VolRatio_20d"); udvr = ind.get("UDVR(20d)")
        # --- Earnings proximity guard (±N days to next earnings) ---
        edays = ind.get("EarningsInDays")
        ewin = get_asset_profile(t, d, atrp).get("earnings_window_days", 5) if 'get_asset_profile' in globals() else 5
        eflag = bool(pd.notna(edays) and edays <= ewin)
        ind.update({
            "Earnings_Window_Flag": eflag,
            "EarningsDays": edays,
            "Earnings_Commentary": (f"Earnings in {int(edays)}d — reduced weight" if (eflag and pd.notna(edays)) else "Earnings: clear"),
        })
        # --- Stochastic (14,3,3) ---
        prof = get_asset_profile(t, d, atrp) if 'get_asset_profile' in globals() else {}
        st_gate = prof.get("stoch_long_gate", 20)
        st = _stoch_features(df_full, k=14, d=3, smooth_k=3, gate_long=st_gate)
        ind.update({
            "StochK": st["StochK"],
            "StochD": st["StochD"],
            "StochCross": st["StochCross"],
            "Stoch_OS_Up": st["Stoch_OS_Up"],
            "Stoch_Long_OK": st["Stoch_Long_OK"],
            "Stoch_Commentary": st["Stoch_Commentary"],
        })
        # --- BB Squeeze & Band Position ---
        bb_th = get_asset_profile(t, d, atrp).get("bb_squeeze_ratio_th", 0.85) if 'get_asset_profile' in globals() else 0.85
        bb = _bb_features(df_full, length=20, std=2.0, squeeze_ratio_th=bb_th)
        ind.update({
            "BB_Width": bb["BB_Width"],
            "BB_Width_MA20": bb["BB_Width_MA20"],
            "BB_Width_Ratio": bb["BB_Width_Ratio"],
            "BB_Pos": bb["BB_Pos"],
            "Squeeze": bb["Squeeze"],
            "Priority_Squeeze": bb["Priority_Squeeze"],
            "BB_Commentary": bb["BB_Commentary"],
        })
        is_etf = _is_etf(t, d, info); is_crypto = _is_crypto(t, d)
        pe   = info.get("trailingPE", np.nan) if (not is_etf and not is_crypto) else np.nan
        epsg = info.get("earningsGrowth", np.nan); epsg = epsg * 100.0 if pd.notna(epsg) else np.nan

        pivH_d = _pivot_highs(df_full["High"].tail(180), lookback=3, tol=1e-6)
        dl1 = np.nan
        if pivH_d:
            above = [p for (_,p) in pivH_d if p >= px]
            if above: dl1 = min(above, key=lambda v: abs(v-px))
        zone_k = _zone_k_from_atrp(atrp)
        dlo = dl1 - zone_k*atr if pd.notna(dl1) and pd.notna(atr) else np.nan
        dhi = dl1 + zone_k*atr if pd.notna(dl1) and pd.notna(atr) else np.nan

        pivH_w = _pivot_highs(df_w["High"].tail(120), lookback=2, tol=1e-6) if df_w is not None and not df_w.empty else []
        wl1 = np.nan
        if pivH_w:
            above_w = [p for (_,p) in pivH_w if p >= px]
            if above_w: wl1 = min(above_w, key=lambda v: abs(v-px))
        try:
            atr_w = float(ta.atr(df_w["High"], df_w["Low"], df_w["Close"], length=14).iloc[-1]) if (df_w is not None and not df_w.empty and len(df_w) >= 15) else np.nan
        except Exception:
            atr_w = np.nan
        w_zone_k = _zone_k_from_atrp(atrp)
        wlo = wl1 - w_zone_k*atr_w if pd.notna(wl1) and pd.notna(atr_w) else np.nan
        whi = wl1 + w_zone_k*atr_w if pd.notna(wl1) and pd.notna(atr_w) else np.nan

        pivL_d = _pivot_lows(df_full["Low"].tail(180), lookback=3, tol=1e-6)
        tsH_d = pivH_d[-1][0] if pivH_d else None
        tsL_d = pivL_d[-1][0] if pivL_d else None
        D_AVWAP_H = _anchored_vwap_from_timestamp(df_full, tsH_d)
        D_AVWAP_L = _anchored_vwap_from_timestamp(df_full, tsL_d)
        tsH_w = (pivH_w[-1][0] if pivH_w else None)
        pivL_w = _pivot_lows(df_w["Low"].tail(120), lookback=2, tol=1e-6) if df_w is not None and not df_w.empty else []
        tsL_w = (pivL_w[-1][0] if pivL_w else None)
        W_AVWAP_H = _anchored_vwap_from_timestamp(df_w if df_w is not None else df_full, tsH_w)
        W_AVWAP_L = _anchored_vwap_from_timestamp(df_w if df_w is not None else df_full, tsL_w)
        ladder = ladder_profile(t, d, atrp)
        # Volatility/asset-aware parameterization
        asset_prof = get_asset_profile(t, d, atrp)
        ladder = tuple(x * asset_prof["ladder_scale"] for x in ladder)
        long_bias = (emaD == "Buy" and weekly_regime in {"Buy","Neutral"})
        if long_bias:
            entry1 = px - ladder[0]*atr; entry2 = px - ladder[1]*atr; entry3 = px - ladder[2]*atr
            exit1  = px + 1.0*atr;       exit2  = px + 2.5*atr;       stop   = px - 1.5*atr
            action_core = "Strong Buy" if (emaD=="Buy" and weekly_regime=="Buy" and 50<=ind["RSI(14)"]<=70) else "Buy / Pullbacks"
            verb = "dips"
        else:
            entry1 = px + ladder[0]*atr; entry2 = px + ladder[1]*atr; entry3 = px + ladder[2]*atr
            exit1  = px - 1.0*atr;       exit2  = px - 2.5*atr;       stop   = px + 1.5*atr
            action_core = "Sell Rallies" if (emaD=="Sell" and weekly_regime in {"Sell","Neutral"}) else "Wait"
            verb = "rallies"

        ladder_note = "OK"
        if long_bias:
            if pd.notna(entry3) and entry3 <= stop: entry3 = np.nan; ladder_note = "E3 suppressed (<= stop)"
            if pd.notna(entry2) and entry2 <= stop: entry2 = np.nan; ladder_note = (ladder_note+"; " if ladder_note!="OK" else "") + "E2 suppressed (<= stop)"
            if pd.notna(entry1) and entry1 <= stop: entry1 = np.nan; ladder_note = (ladder_note+"; " if ladder_note!="OK" else "") + "E1 suppressed (<= stop)"
        else:
            if pd.notna(entry3) and entry3 >= stop: entry3 = np.nan; ladder_note = "E3 suppressed (>= stop)"
            if pd.notna(entry2) and entry2 >= stop: entry2 = np.nan; ladder_note = (ladder_note+"; " if ladder_note!="OK" else "") + "E2 suppressed (>= stop)"
            if pd.notna(entry1) and entry1 >= stop: entry1 = np.nan; ladder_note = (ladder_note+"; " if ladder_note!="OK" else "") + "E1 suppressed (>= stop)"

        dist_e1 = pct_dist(px, entry1)
        dist_st = pct_dist(entry2 if pd.notna(entry2) else px, stop)
        rr_e2   = (exit1 - entry2) / abs(entry2 - stop) if all(pd.notna(x) for x in (exit1, entry2, stop)) and (entry2 - stop) != 0 else np.nan

        res = compute_resistance_targets(df_full, atr, px, _zone_k_from_atrp(atrp))
        lo = res.get("Res1_Zone_Lo"); hi = res.get("Res1_Zone_Hi")
        if not pd.isna(lo) and not pd.isna(hi):
            band_text = " Inside resistance band; prefer dips into band or a close above upper band." if (lo <= px <= hi) else (" Above resistance band; look for pullback toward upper band (old resistance becomes support)." if px > hi else " Below resistance band; better odds after reclaiming lower band.")
        else:
            band_text = ""

        row_tmp = {"EMA_Regime": emaD, "Weekly_Regime": weekly_regime, "RSI(14)": ind.get("RSI(14)"), "ADX(14)": ind.get("ADX(14)"), "MACD_Hist_Pos": ind.get("MACD_Hist_Pos"), "VolRatio_20d": volratio, "UDVR(20d)": udvr, "Breakout20": ind.get("Breakout20"), "Ticker": t, "Description": d, "ATR%": atrp}
        priority = compute_priority(row_tmp, is_etf=is_etf)
        if args.guard == "SAFE":
            priority = max(0.0, min(100.0, priority + (5.0 if guard_bias > 0 else -10.0)))
        elif args.guard == "HARD":
            priority = max(0.0, min(100.0, priority + (8.0 if guard_bias > 0 else -20.0)))

        asof = snap.get("asof"); session = snap.get("session"); delay = snap.get("delay_min")
        px_live = snap.get("last") if (DataMode in {"LIVE","PRE"} and pd.notna(snap.get("last"))) else px
        day_high = snap.get("high"); day_low = snap.get("low")

        T1_Hit = bool(pd.notna(day_high) and day_high >= exit1) if DataMode in {"LIVE","PRE"} else False
        T2_Hit = bool(pd.notna(day_high) and day_high >= exit2) if DataMode in {"LIVE","PRE"} else False
        Stop_Hit = bool(pd.notna(day_low) and day_low <= stop) if DataMode in {"LIVE","PRE"} else False
        if DataMode in {"LIVE","PRE"}:
            if long_bias:
                E1_Hit = bool(pd.notna(day_low) and pd.notna(entry1) and day_low <= entry1)
                E2_Hit = bool(pd.notna(day_low) and pd.notna(entry2) and day_low <= entry2)
                E3_Hit = bool(pd.notna(day_low) and pd.notna(entry3) and day_low <= entry3)
            else:
                E1_Hit = bool(pd.notna(day_high) and pd.notna(entry1) and day_high >= entry1)
                E2_Hit = bool(pd.notna(day_high) and pd.notna(entry2) and day_high >= entry2)
                E3_Hit = bool(pd.notna(day_high) and pd.notna(entry3) and day_high >= entry3)
        else:
            E1_Hit = E2_Hit = E3_Hit = False
        At_Res1_Zone_LIVE = bool(pd.notna(lo) and pd.notna(hi) and pd.notna(px_live) and (lo <= px_live <= hi)) if DataMode in {"LIVE","PRE"} else False
        Live_Delta_pct = pct(px_live, px)

        PremarketGapPct = np.nan; EarningsInDays = None
        if DataMode == "PRE":
            PremarketGapPct = pct(px_live, px) if session == "pre" and pd.notna(px_live) else np.nan
            EarningsInDays = _earnings_proximity_days(t)

        mtf_tip = (f"Value-line (21D/21W): {ind.get('EMA21'):.2f}/{EMA21w:.2f}" if pd.notna(EMA21w) else f"Value-line (21D): {ind.get('EMA21'):.2f}")
        long_term = _novice_long_term_text(weekly_regime, EMA21w, ind.get('EMA21'), px, atrp, lo, hi)
        novice_short = f"Short-term: consider adds on {verb} near {entry2:.2f} if trend stays {emaD}; first trim near {exit1:.2f}. {mtf_tip}."
        next_ceiling_str = (f"${_fmt_price(lo)}-{_fmt_price(hi)}" if (pd.notna(lo) and pd.notna(hi)) else "n/a (no pivot above—ATH)")

        novice_rows.append({
            "Ticker": t,
            "Close": px,
            "Action": f"{action_core}. Plan: {'Buy pullbacks near' if long_bias else 'Sell rallies near'} "
                      f"{('$' + _fmt_price(entry1)) if pd.notna(entry1) else 'n/a'}/"
                      f"{('$' + _fmt_price(entry2)) if pd.notna(entry2) else 'n/a'}/"
                      f"{('$' + _fmt_price(entry3)) if pd.notna(entry3) else 'n/a'}. "
                      f"Stop ${stop:.2f}. Targets ${exit1:.2f}/${exit2:.2f}. "
                      f"Note: base ${_fmt_price(res.get('Breakout_Base'))} ; next ceiling {next_ceiling_str} ; "
                      f"BO T1 ${_fmt_price(res.get('BO_T1'))} ; BO T2 ${_fmt_price(res.get('BO_T2'))}{band_text}"
                      f"{' [Ladder check: ' + ladder_note + ']' if ladder_note != 'OK' else ''}",
            "Novice_Action_Short": novice_short,
            "Novice_Action_Long": long_term,
            "BB_Commentary": ind.get("BB_Commentary"),
            "Stoch_Commentary": ind.get("Stoch_Commentary"),
            "Earnings_Commentary": ind.get("Earnings_Commentary"),
            "W_Commentary": f"W: {weekly_regime}. Px vs EMA21w: " + (f"{((px/(EMA21w)-1.0)*100.0):+.1f}%" if pd.notna(EMA21w) else "n/a") + ".",
            "M_Commentary": "M: " + str(monthly_regime) + ". Px vs EMA21m: " + (f"{pct(px, EMA21m):+.1f}%" if pd.notna(EMA21m) else "n/a") + ".",  
            "MTF_Compare": f"D/W value-line gaps: {pct(px, ema21):+.1f}% / " + (f"{pct(px, EMA21w):+.1f}%" if pd.notna(EMA21w) else "n/a") + f" - Regimes {emaD}/{weekly_regime}",
            "Priority": round(priority,1),
            "Entry1": entry1, "Entry2": entry2, "Entry3": entry3, "T1": exit1, "T2": exit2, "Stop": stop,
            "D_Layer": dl1, "D_Zone_Lo": dlo, "D_Zone_Hi": dhi,
            "W_Layer": wl1, "W_Zone_Lo": wlo, "W_Zone_Hi": whi,
            "DataMode": DataMode, "AsOf": asof, "Session": session, "DelayMin": delay,
            "AdjustedHistory": True, "Live_Delta%": Live_Delta_pct,
            "At_Res1_Zone_LIVE": At_Res1_Zone_LIVE,
            "T1_Hit": T1_Hit, "T2_Hit": T2_Hit, "Stop_Hit": Stop_Hit,
            "E1_Hit": E1_Hit, "E2_Hit": E2_Hit, "E3_Hit": E3_Hit,
            "PremarketGapPct": PremarketGapPct, "EarningsInDays": EarningsInDays,
            "Ladder_Sanity": ladder_note,
        })

        today_lbl = "n/a"; week_lbl = "n/a"
        ht = ind.get("HitToday_Up1ATR %"); hw = ind.get("HitWeek_Up1ATR %")
        if pd.notna(ht): today_lbl = f"{ht:.0f}%"
        if pd.notna(hw): week_lbl  = f"{hw:.0f}%"
        expert_comment = (
            f"Trend {emaD}/{weekly_regime}; RSI {safe_float(ind.get('RSI(14)')):.0f}, ADX {safe_float(ind.get('ADX(14)')):.0f}; "
            f"MACD hist {'+' if bool(ind.get('MACD_Hist_Pos')) else '-'}; VolRatio {safe_float(volratio):.2f}, UDVR {safe_float(udvr):.2f}; "
            f"D LILO {(_fmt_price(dl1) if pd.notna(dl1) else 'n/a')} zone {(_fmt_price(dlo))}-{(_fmt_price(dhi))}; "
            f"W LILO {(_fmt_price(wl1) if pd.notna(wl1) else 'n/a')} zone {(_fmt_price(wlo))}-{(_fmt_price(whi))}; "
            f"AVWAP(H/L) D {(_fmt_price(D_AVWAP_H))}/{(_fmt_price(D_AVWAP_L))}, W {(_fmt_price(W_AVWAP_H))}/{(_fmt_price(W_AVWAP_L))}; "
            f"Hit +1ATR today {today_lbl}, week {week_lbl}."
        )
        try:
            avgvol20 = float(ta.sma(df_full["Volume"], length=20).iloc[-1]) if "Volume" in df_full.columns else np.nan
        except Exception:
            avgvol20 = np.nan

        expert_rows.append({
            "Ticker": t, "Description": d, "Priority": round(priority,1),
            "EMA_Regime": emaD, "Weekly_Regime": weekly_regime,
            "Signal_Age": ind.get("Signal_Age"), "Fresh_Cross": ind.get("Fresh_Cross"),
            "Open": ind.get("Open"), "High": ind.get("High"), "Low": ind.get("Low"), "Close": px,
            "SMA21": ind.get("SMA21"), "EMA_Short": ind.get("EMA_Short"), "EMA_Long": ind.get("EMA_Long"),
            "EMA21": ema21, "EMA21w": EMA21w,
            "RSI(14)": ind.get("RSI(14)"), "ADX(14)": ind.get("ADX(14)"),
            "MACD_Hist_Pos": ind.get("MACD_Hist_Pos"), "MACD_Hist": (float(df_full["MACD_Hist"].iloc[-1]) if "MACD_Hist" in df_full.columns and not pd.isna(df_full["MACD_Hist"].iloc[-1]) else np.nan),
            "ATR": atr, "ATR%": atrp, "BreakoutStrength_ATR": ind.get("BreakoutStrength_ATR"),
            "VolRatio_20d": ind.get("VolRatio_20d"), "UDVR(20d)": ind.get("UDVR(20d)"), "VolZ_20": ind.get("VolZ_20"),
            "Breakout20": ind.get("Breakout20"), "Breakdown20": ind.get("Breakdown20"),
            "High_52W": ind.get("High_52W"), "Low_52W": ind.get("Low_52W"), "%From_21SMA": ind.get("%From_21SMA"),
            "HitToday_Up1ATR %": ind.get("HitToday_Up1ATR %"), "HitWeek_Up1ATR %": ind.get("HitWeek_Up1ATR %"),
            "Entry1": entry1, "Entry2": entry2, "Entry3": entry3, "Exit1": exit1, "Exit2": exit2, "Stop": stop,
            "Dist_to_Entry1_pct": dist_e1, "Dist_to_Stop_pct": dist_st, "RR_Entry2_to_Exit1": rr_e2,
            "Breakout_Base": res.get("Breakout_Base"), "Res1_Pivot": res.get("Res1_Pivot"),
            "Res1_Zone_Lo": res.get("Res1_Zone_Lo"), "Res1_Zone_Hi": res.get("Res1_Zone_Hi"),
            "BO_T1": res.get("BO_T1"), "BO_T2": res.get("BO_T2"), "At_Res1_Zone": res.get("At_Res1_Zone"),
            "D_Layer": dl1, "D_Zone_Lo": dlo, "D_Zone_Hi": dhi,
            "W_Layer": wl1, "W_Zone_Lo": wlo, "W_Zone_Hi": whi,
            "D_AVWAP_H": D_AVWAP_H, "D_AVWAP_L": D_AVWAP_L, "W_AVWAP_H": W_AVWAP_H, "W_AVWAP_L": W_AVWAP_L,
            "PE": pe, "EPS_Growth_Pct": epsg, "BB_Width": ind.get("BB_Width"), "BB_Width_MA20": ind.get("BB_Width_MA20"), "BB_Width_Ratio": ind.get("BB_Width_Ratio"), "BB_Pos": ind.get("BB_Pos"), "Squeeze": ind.get("Squeeze"), "Priority_Squeeze": ind.get("Priority_Squeeze"), "StochK": ind.get("StochK"), "StochD": ind.get("StochD"), "StochCross": ind.get("StochCross"), "Stoch_OS_Up": ind.get("Stoch_OS_Up"), "Stoch_Long_OK": ind.get("Stoch_Long_OK"), "Earnings_Window_Flag": ind.get("Earnings_Window_Flag"), "EarningsDays": ind.get("EarningsDays"), "Expert_Commentary": _normalize_text(expert_comment),
            "DataMode": DataMode, "AsOf": asof, "Session": session, "DelayMin": delay,
            "AdjustedHistory": True, "Live_Delta%": Live_Delta_pct, "At_Res1_Zone_LIVE": At_Res1_Zone_LIVE,
            "T1_Hit": T1_Hit, "T2_Hit": T2_Hit, "Stop_Hit": Stop_Hit, "E1_Hit": E1_Hit, "E2_Hit": E2_Hit, "E3_Hit": E3_Hit,
            "PremarketGapPct": PremarketGapPct, "EarningsInDays": EarningsInDays, "Ladder_Sanity": ladder_note, "AvgVol20": avgvol20,
        })

    if not novice_rows:
        print("No rows produced."); return
    df_novice = pd.DataFrame(novice_rows)
    df_expert = pd.DataFrame(expert_rows)
    for df_out in (df_novice, df_expert):
        if "Priority" in df_out.columns:
            df_out.sort_values(["Priority","Ticker"], ascending=[False, True], inplace=True, kind="mergesort")
            df_out["Rank"] = range(1, len(df_out)+1)
    df_novice.to_csv(NOVICE_CSV, index=False); df_expert.to_csv(EXPERT_CSV, index=False)
    print(f"Saved: {NOVICE_CSV}"); print(f"Saved: {EXPERT_CSV}")
    if not args.no_xlsx:
        try:
            import xlsxwriter
            with pd.ExcelWriter(OUT_XLSX, engine="xlsxwriter") as writer:
                df_novice.to_excel(writer, sheet_name="Novice", index=False)
                df_expert.to_excel(writer, sheet_name="Expert", index=False)
                legend_text = ("README_Legend\\n- Modes: AFTER (plan), PRE (gap/earnings), LIVE (hits).\\n"
                               "- Novice text is value-line aware; numeric references included.\\n"
                               "- Hits: E1/E2/E3/T1/T2/Stop booleans based on intraday extremes.\\n"
                               "- AdjustedHistory=True: indicators on adjusted data.")
                pd.DataFrame({"README":[legend_text]}).to_excel(writer, sheet_name="README", index=False, header=False)
            print(f"Saved: {OUT_XLSX}")
        except Exception as e:
            print(f"XLSX skipped ({e}). Install xlsxwriter to enable Excel export.")
    print("Done.")

# --- BB Squeeze + Band Position helper ---
from typing import Dict, Any
def _bb_features(df: "pd.DataFrame", length: int = 20, std: float = 2.0, squeeze_ratio_th: float = 0.85) -> Dict[str, Any]:
    out = {"BB_Upper": np.nan, "BB_Lower": np.nan, "BB_Middle": np.nan, "BB_Width": np.nan,
           "BB_Width_MA20": np.nan, "BB_Width_Ratio": np.nan, "BB_Pos": np.nan,
           "Squeeze": False, "Priority_Squeeze": np.nan, "BB_Commentary": "BB: n/a"}
    try:
        if df is None or df.empty or len(df) < 25:
            return out
        bb = ta.bbands(df["Close"], length=length, std=std)
        if bb is None or bb.empty:
            return out
        upper = bb.filter(like="BBU_").iloc[:, -1]
        middle = bb.filter(like="BBM_").iloc[:, -1]
        lower  = bb.filter(like="BBL_").iloc[:, -1]
        width_series = bb.filter(like="BBB_").iloc[:, -1]
        pos_series   = bb.filter(like="BBP_").iloc[:, -1]
        width_ma20 = ta.sma(width_series, length=20)

        def f(x): 
            try: return float(x)
            except Exception: return np.nan

        u = f(upper.iloc[-1]); m = f(middle.iloc[-1]); l = f(lower.iloc[-1])
        w = f(width_series.iloc[-1]); wma = f(width_ma20.iloc[-1]) if width_ma20 is not None else np.nan
        pos = f(pos_series.iloc[-1])

        ratio = (w / wma) if (pd.notna(w) and pd.notna(wma) and wma != 0) else np.nan
        squeeze = bool(pd.notna(ratio) and ratio <= squeeze_ratio_th)
        pr = float(np.clip((1.0 - (ratio - 0.5) / 1.0), 0.0, 1.0)) if pd.notna(ratio) else np.nan

        hint_bits = []
        if pd.notna(ratio): hint_bits.append(f"width {ratio:0.2f}× 20dma")
        if pd.notna(pos):   hint_bits.append(f"pos {pos:0.2f}")
        hint = f"BB: {'SQUEEZE' if squeeze else 'no squeeze'} (" + ", ".join(hint_bits) + ")" if hint_bits else "BB: n/a"

        out.update({"BB_Upper": u, "BB_Lower": l, "BB_Middle": m, "BB_Width": w,
                    "BB_Width_MA20": wma, "BB_Width_Ratio": ratio, "BB_Pos": pos,
                    "Squeeze": squeeze, "Priority_Squeeze": pr, "BB_Commentary": hint})
        return out
    except Exception:
        return out
# --- END helper ---
# --- Stochastic (14,3,3) helper ---
from typing import Dict, Any
def _stoch_features(df: "pd.DataFrame", k: int = 14, d: int = 3, smooth_k: int = 3, gate_long: int = 20) -> Dict[str, Any]:
    out = {"StochK": np.nan, "StochD": np.nan, "StochCross": "Flat",
           "Stoch_OS_Up": False, "Stoch_Long_OK": False, "Stoch_Commentary": "Stoch: n/a"}
    try:
        if df is None or df.empty or len(df) < max(k + smooth_k + d, 30):
            return out
        st = ta.stoch(df["High"], df["Low"], df["Close"], k=k, d=d, smooth_k=smooth_k)
        if st is None or st.empty:
            return out
        kser = st.filter(like="STOCHk_").iloc[:, -1]
        dser = st.filter(like="STOCHd_").iloc[:, -1]
        if len(kser) < 2 or len(dser) < 2:
            return out
        k0, d0 = float(kser.iloc[-1]), float(dser.iloc[-1])
        k1, d1 = float(kser.iloc[-2]), float(dser.iloc[-2])

        cross = "Flat"
        if np.isfinite(k0) and np.isfinite(d0) and np.isfinite(k1) and np.isfinite(d1):
            if k1 <= d1 and k0 > d0:
                cross = "Up"
            elif k1 >= d1 and k0 < d0:
                cross = "Down"

        os_up = bool(np.isfinite(k1) and np.isfinite(k0) and k1 < gate_long and k0 >= gate_long and k0 > d0)
        long_ok = bool(np.isfinite(k0) and np.isfinite(d0) and k0 > d0 and k0 >= gate_long)

        comm = f"Stoch: {cross}{' OS-up' if os_up else ''} (K {k0:.0f}, D {d0:.0f})" if np.isfinite(k0) and np.isfinite(d0) else "Stoch: n/a"
        out.update({"StochK": k0, "StochD": d0, "StochCross": cross,
                    "Stoch_OS_Up": os_up, "Stoch_Long_OK": long_ok, "Stoch_Commentary": comm})
        return out
    except Exception:
        return out
# --- END Stochastic helper ---

if __name__ == "__main__":
    main()

