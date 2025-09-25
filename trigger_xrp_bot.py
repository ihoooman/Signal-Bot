#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, requests, numpy as np, pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

# ====== تنظیمات از .env ======
load_dotenv(os.path.expanduser("~/xrpbot/.env"))
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")
CRYPTOCOMPARE_API_KEY = os.getenv("CRYPTOCOMPARE_API_KEY")

# اگر True فقط موقع سیگنال "خرید/فروش" پیام می‌دهد (وگرنه همیشه گزارش می‌دهد)
SEND_ONLY_ON_TRIGGER = False

# ===== Thresholds (قابل تغییر) =====
BUY_RSI_D_MIN   = 55   # Daily RSI برای خرید
SELL_RSI_D_MAX  = 45   # Daily RSI برای فروش

H4_RSI_BUY_MIN  = 35   # 4h RSI برای خرید
H4_RSI_BUY_MAX  = 45

H4_RSI_SELL_MIN = 55   # 4h RSI برای فروش
H4_RSI_SELL_MAX = 65

# بک‌تست: افق زمانی و آستانه سود/ضرر
DAILY_LOOKAHEAD_BARS = 3     # 3 کندل روزانه بعدی
H4_LOOKAHEAD_BARS    = 6     # 6 کندل 4ساعته (~24 ساعت)
PROFIT_THRESHOLD     = 0.01  # 1% به عنوان برد برای Buy (و -1% برای Sell)
NEAR_PCT             = 0.01  # 1% نزدیکی ساپورت/مقاومت

# دارایی‌ها: XRP + BTC + ETH + SOL
SYMBOLS = [
    ("XRP", "XRP"),
    ("BTC", "BTC"),
    ("ETH", "ETH"),
    ("SOL", "SOL"),
]

CRYPTOCOMPARE_BASE = "https://min-api.cryptocompare.com/data/v2"

# ===================== ابزار داده و اندیکاتورها =====================

def fetch_cc(fsym: str, tsym: str, timeframe: str, limit=300):
    """Fetch OHLCV from CryptoCompare. timeframe: '1d' or '4h' (aggregated)."""
    if timeframe == "1d":
        url = f"{CRYPTOCOMPARE_BASE}/histoday?fsym={fsym}&tsym={tsym}&limit={limit}"
    elif timeframe == "4h":
        url = f"{CRYPTOCOMPARE_BASE}/histohour?fsym={fsym}&tsym={tsym}&aggregate=4&limit={limit}"
    else:
        raise ValueError("Unsupported timeframe")
    headers = {"Authorization": f"Apikey {CRYPTOCOMPARE_API_KEY}"} if CRYPTOCOMPARE_API_KEY else {}
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    j = r.json()
    if j.get("Response") != "Success":
        raise RuntimeError(f"CryptoCompare error: {j.get('Message')}")
    d = j["Data"]["Data"]
    df = pd.DataFrame(d)
    # Columns: time, high, low, open, close, volumefrom, volumeto
    df.rename(columns={"volumeto": "volume"}, inplace=True)
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    return df

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    dn = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).rolling(period).mean()
    roll_dn = pd.Series(dn, index=series.index).rolling(period).mean()
    rs = roll_up / (roll_dn + 1e-12)
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def last_cross_up(line, signal):
    if len(line) < 2: return False
    return (line.iloc[-2] <= signal.iloc[-2]) and (line.iloc[-1] > signal.iloc[-1])

def last_cross_down(line, signal):
    if len(line) < 2: return False
    return (line.iloc[-2] >= signal.iloc[-2]) and (line.iloc[-1] < signal.iloc[-1])

def near(value, target, pct=NEAR_PCT):
    return abs(value - target) <= abs(target) * pct

# === Divergence helpers ===

def local_minima(series):
    mins = []
    for i in range(1, len(series)-1):
        if series.iloc[i] < series.iloc[i-1] and series.iloc[i] <= series.iloc[i+1]:
            mins.append(i)
    return mins

def local_maxima(series):
    maxs = []
    for i in range(1, len(series)-1):
        if series.iloc[i] > series.iloc[i-1] and series.iloc[i] >= series.iloc[i+1]:
            maxs.append(i)
    return maxs

def bullish_divergence(price, rsi_series):
    mp = local_minima(price)
    mr = local_minima(rsi_series)
    if len(mp) < 2 or len(mr) < 2: return False
    i2, i1 = mp[-1], mp[-2]
    r1 = min(mr, key=lambda j: abs(j - i1))
    r2 = min(mr, key=lambda j: abs(j - i2))
    return (price.iloc[i2] < price.iloc[i1]) and (rsi_series.iloc[r2] > rsi_series.iloc[r1])

def bearish_divergence(price, rsi_series):
    mp = local_maxima(price)
    mr = local_maxima(rsi_series)
    if len(mp) < 2 or len(mr) < 2: return False
    i2, i1 = mp[-1], mp[-2]
    r1 = min(mr, key=lambda j: abs(j - i1))
    r2 = min(mr, key=lambda j: abs(j - i2))
    return (price.iloc[i2] > price.iloc[i1]) and (rsi_series.iloc[r2] < rsi_series.iloc[r1])

# ===================== بک‌تست و احتمال =====================

def prob_and_return_from_flags(close: pd.Series, flags: pd.Series, lookahead: int, direction: str):
    """
    close: سری قیمت بسته‌شدن
    flags: سری بولی با True در نقاط سیگنال گذشته
    lookahead: تعداد کندل‌های جلو برای ارزیابی
    direction: "BUY" یا "SELL"
    خروجی: (probability, avg_forward_return)
    """
    idx = np.where(flags.values)[0]
    wins = 0
    rets = []
    for t in idx:
        if t + lookahead >= len(close):
            continue
        r = (close.iloc[t + lookahead] / close.iloc[t]) - 1.0
        rets.append(r)
        if direction == "BUY":
            if r > PROFIT_THRESHOLD:
                wins += 1
        else:  # SELL
            if r < -PROFIT_THRESHOLD:
                wins += 1
    n = len(rets)
    if n == 0:
        return None, None
    # Laplace smoothing برای جلوگیری از 0%/100%
    prob = (wins + 1) / (n + 2)
    avg_ret = float(np.mean(rets))
    return prob, avg_ret

def historical_flags_daily(close_d: pd.Series, rsi_d: pd.Series, macd_d: pd.Series, sig_d: pd.Series, sma50_d: pd.Series):
    # primary BUY: cross up & RSI>55 & close>sma50
    bu_flags = (macd_d.shift(1) <= sig_d.shift(1)) & (macd_d > sig_d) & (rsi_d > BUY_RSI_D_MIN) & (close_d > sma50_d)
    # primary SELL: cross down & RSI<45 & close<sma50
    se_flags = (macd_d.shift(1) >= sig_d.shift(1)) & (macd_d < sig_d) & (rsi_d < SELL_RSI_D_MAX) & (close_d < sma50_d)
    return bu_flags.fillna(False), se_flags.fillna(False)

def historical_flags_4h(close_h: pd.Series, rsi_h: pd.Series, hist_h: pd.Series, for_buy=True):
    # ثانویه BUY: نزدیک ساپورت (لو 30 کندل) + RSI در باند + هیستوگرام مثبت + واگرایی مثبت
    # ثانویه SELL: نزدیک مقاومت (های 30 کندل) + RSI در باند + هیستوگرام منفی + واگرایی منفی
    flags = []
    for t in range(30, len(close_h)-1):
        price_t = close_h.iloc[t]
        recent_low  = close_h.iloc[t-29:t+1].min()
        recent_high = close_h.iloc[t-29:t+1].max()
        if for_buy:
            support_ok = near(price_t, recent_low, pct=NEAR_PCT)
            rsi_ok = (H4_RSI_BUY_MIN <= float(rsi_h.iloc[t]) <= H4_RSI_BUY_MAX)
            div_ok = bullish_divergence(close_h.iloc[:t+1], rsi_h.iloc[:t+1])
            macd_ok = float(hist_h.iloc[t]) > 0
            flags.append(support_ok and rsi_ok and div_ok and macd_ok)
        else:
            near_res = near(price_t, recent_high, pct=NEAR_PCT)
            rsi_ok = (H4_RSI_SELL_MIN <= float(rsi_h.iloc[t]) <= H4_RSI_SELL_MAX)
            div_ok = bearish_divergence(close_h.iloc[:t+1], rsi_h.iloc[:t+1])
            macd_ok = float(hist_h.iloc[t]) < 0
            flags.append(near_res and rsi_ok and div_ok and macd_ok)
    s = pd.Series(flags, index=close_h.index[30:len(close_h)-1])
    return s

# ===================== پیام و ابزار =====================

def tehran_now():
    return datetime.now(ZoneInfo("Asia/Tehran")).strftime("%Y-%m-%d %H:%M:%S")

def send_telegram(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}
    requests.post(url, json=payload, timeout=20)

def build_message_block(symbol_name, signal_type, timeframe, ctx, prob=None, avg_ret=None, horizon_label=""):
    dt = tehran_now()
    if signal_type == "BUY":
        head = f"<b>{symbol_name}</b> — <b>BUY SIGNAL ✅</b>  |  <b>{timeframe}</b>  |  {dt}"
    elif signal_type == "SELL":
        head = f"<b>{symbol_name}</b> — <b>SELL SIGNAL ⛔️</b>  |  <b>{timeframe}</b>  |  {dt}"
    else:
        head = f"<b>{symbol_name}</b> — <b>No signal ❌</b>  |  {dt}"

    lines = [
        head,
        f"Price: <b>{ctx['price']:.4f}$</b>",
        f"Daily — RSI14: {ctx['rsi_d']:.2f} | MACD-Hist: {ctx['hist_d']:.4f} | SMA50: {ctx['sma50_d']:.4f}",
        f"4h — RSI14: {ctx['rsi_h4']:.2f} | MACD-Hist: {ctx['hist_h4']:.4f}",
    ]
    if prob is not None and avg_ret is not None and horizon_label:
        lines.append(f"Backtest({horizon_label}) — Prob. profit: <b>{prob*100:.1f}%</b> | Avg fwd return: <b>{avg_ret*100:.2f}%</b>")

    extra = []
    if ctx.get('support_ok'):
        extra.append("near support")
    if ctx.get('near_res'):
        extra.append("near resistance")
    if extra:
        lines.append("Context: " + ", ".join(extra))

    return "\n".join(lines)

# ===================== منطق هر نماد =====================

def evaluate_symbol(symbol: str, name: str):
    # Daily (CryptoCompare)
    d1 = fetch_cc(symbol, "USD", "1d", limit=220)
    close_d = d1["close"]
    rsi_d = rsi(close_d, 14)
    macd_d, sig_d, hist_d = macd(close_d, 12, 26, 9)
    sma50_d = close_d.rolling(50).mean()

    # سیگنال‌های روزانه جاری
    primary = last_cross_up(macd_d, sig_d) and (rsi_d.iloc[-1] > BUY_RSI_D_MIN) and (close_d.iloc[-1] > sma50_d.iloc[-1])
    sell_primary = last_cross_down(macd_d, sig_d) and (rsi_d.iloc[-1] < SELL_RSI_D_MAX) and (close_d.iloc[-1] < sma50_d.iloc[-1])

    # 4h (CryptoCompare)
    h4 = fetch_cc(symbol, "USD", "4h", limit=320)
    close_h4 = h4["close"]
    rsi_h4 = rsi(close_h4, 14)
    macd_h4, sig_h4, hist_h4 = macd(close_h4, 12, 26, 9)
    price_now = float(close_h4.iloc[-1])

    # ساپورت/مقاومت (برای XRP از لنگر 2.75 استفاده می‌کنیم؛ برای بقیه فقط لو/های اخیر)
    recent_low = close_h4.tail(30).min()
    recent_high = close_h4.tail(30).max()

    if name == "XRP":
        near_anchor = near(price_now, 2.75, pct=NEAR_PCT)
        support_ok = near_anchor or near(price_now, recent_low, pct=NEAR_PCT)
    else:
        support_ok = near(price_now, recent_low, pct=NEAR_PCT)

    near_res = near(price_now, recent_high, pct=NEAR_PCT)

    rsi_buy_band  = H4_RSI_BUY_MIN  <= float(rsi_h4.iloc[-1]) <= H4_RSI_BUY_MAX
    rsi_sell_band = H4_RSI_SELL_MIN <= float(rsi_h4.iloc[-1]) <= H4_RSI_SELL_MAX

    div_ok  = bullish_divergence(close_h4, rsi_h4)
    div_neg = bearish_divergence(close_h4, rsi_h4)

    macd_hist_ok  = float(hist_h4.iloc[-1]) > 0
    macd_hist_neg = float(hist_h4.iloc[-1]) < 0

    secondary       = support_ok and rsi_buy_band  and div_ok  and macd_hist_ok
    sell_secondary  = near_res   and rsi_sell_band and div_neg and macd_hist_neg

    # انتخاب نهایی
    signal_type = "NONE"; timeframe = ""
    if primary:
        signal_type, timeframe = "BUY", "Daily"
    elif secondary:
        signal_type, timeframe = "BUY", "4h"
    elif sell_primary:
        signal_type, timeframe = "SELL", "Daily"
    elif sell_secondary:
        signal_type, timeframe = "SELL", "4h"

    # --- برآورد احتمال بر اساس بک‌تست تجربی ---
    prob = avg_ret = None; horizon_label = ""
    if signal_type != "NONE":
        # DAILY flags
        bu_flags_d, se_flags_d = historical_flags_daily(close_d, rsi_d, macd_d, sig_d, sma50_d)
        # 4H flags
        bu_flags_h = historical_flags_4h(close_h4, rsi_h4, hist_h4, for_buy=True)
        se_flags_h = historical_flags_4h(close_h4, rsi_h4, hist_h4, for_buy=False)

        if timeframe == "Daily":
            if signal_type == "BUY":
                prob, avg_ret = prob_and_return_from_flags(close_d, bu_flags_d, DAILY_LOOKAHEAD_BARS, "BUY")
            else:
                prob, avg_ret = prob_and_return_from_flags(close_d, se_flags_d, DAILY_LOOKAHEAD_BARS, "SELL")
            horizon_label = f"next {DAILY_LOOKAHEAD_BARS}D"
        else:  # 4h
            if signal_type == "BUY":
                prob, avg_ret = prob_and_return_from_flags(close_h4, bu_flags_h, H4_LOOKAHEAD_BARS, "BUY")
            else:
                prob, avg_ret = prob_and_return_from_flags(close_h4, se_flags_h, H4_LOOKAHEAD_BARS, "SELL")
            horizon_label = f"next {H4_LOOKAHEAD_BARS}x4h"

    ctx = dict(
        price=price_now,
        rsi_d=float(rsi_d.iloc[-1]),
        rsi_h4=float(rsi_h4.iloc[-1]),
        hist_d=float((macd_d - sig_d).iloc[-1]),  # برای نمایش مختصر
        hist_h4=float((macd_h4 - sig_h4).iloc[-1]),
        sma50_d=float(sma50_d.iloc[-1]),
        support_ok=support_ok,
        div_ok=div_ok,
        near_res=near_res,
        div_neg=div_neg,
    )

    return name, signal_type, timeframe, ctx, prob, avg_ret, horizon_label

# ===================== اجرای اصلی =====================

def main():
    blocks = []
    any_signal = False

    for sym, name in SYMBOLS:
        try:
            name, sig, tf, ctx, prob, avg_ret, hz = evaluate_symbol(sym, name)
            if sig in ("BUY", "SELL"): any_signal = True
            block = build_message_block(name, sig, tf, ctx, prob, avg_ret, hz)
            blocks.append(block)
        except Exception as e:
            blocks.append(f"<b>{name}</b> — error: {e}")

    text = "\n\n".join(blocks) + "\n\nSources: TradingView (USD pairs), CryptoCompare, CoinMarketCap"

    if SEND_ONLY_ON_TRIGGER:
        if any_signal:
            send_telegram(text)
    else:
        send_telegram(text)

if __name__ == "__main__":
    main()
