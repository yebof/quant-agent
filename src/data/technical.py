import pandas as pd
import ta

from src.models import OHLCV, TechnicalIndicators


def compute_indicators(symbol: str, bars: list[OHLCV]) -> TechnicalIndicators:
    if not bars:
        return TechnicalIndicators(symbol=symbol)

    df = pd.DataFrame([b.model_dump() for b in bars])
    df = df.set_index("date").sort_index()

    result = TechnicalIndicators(symbol=symbol)

    # Moving averages
    if len(df) >= 20:
        result.ma_20 = round(float(df["close"].rolling(20).mean().iloc[-1]), 2)
    if len(df) >= 50:
        result.ma_50 = round(float(df["close"].rolling(50).mean().iloc[-1]), 2)
    if len(df) >= 200:
        result.ma_200 = round(float(df["close"].rolling(200).mean().iloc[-1]), 2)

    # RSI
    if len(df) >= 15:
        rsi = ta.momentum.RSIIndicator(df["close"], window=14)
        rsi_val = rsi.rsi().iloc[-1]
        if pd.notna(rsi_val):
            result.rsi_14 = round(float(rsi_val), 2)

    # MACD
    if len(df) >= 26:
        macd_ind = ta.trend.MACD(df["close"])
        macd_val = macd_ind.macd().iloc[-1]
        signal_val = macd_ind.macd_signal().iloc[-1]
        hist_val = macd_ind.macd_diff().iloc[-1]
        if pd.notna(macd_val):
            result.macd = round(float(macd_val), 4)
        if pd.notna(signal_val):
            result.macd_signal = round(float(signal_val), 4)
        if pd.notna(hist_val):
            result.macd_hist = round(float(hist_val), 4)

    # Bollinger Bands
    if len(df) >= 20:
        bb = ta.volatility.BollingerBands(df["close"], window=20)
        bb_h = bb.bollinger_hband().iloc[-1]
        bb_m = bb.bollinger_mavg().iloc[-1]
        bb_l = bb.bollinger_lband().iloc[-1]
        if pd.notna(bb_h):
            result.bb_upper = round(float(bb_h), 2)
        if pd.notna(bb_m):
            result.bb_middle = round(float(bb_m), 2)
        if pd.notna(bb_l):
            result.bb_lower = round(float(bb_l), 2)

    # ATR
    if len(df) >= 15:
        atr = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14)
        atr_val = atr.average_true_range().iloc[-1]
        if pd.notna(atr_val):
            result.atr_14 = round(float(atr_val), 2)

    # Volume change %
    if len(df) >= 6:
        recent_vol = df["volume"].tail(5).mean()
        prev_vol = df["volume"].iloc[-10:-5].mean() if len(df) >= 10 else df["volume"].iloc[:-5].mean()
        if prev_vol > 0:
            result.volume_change_pct = round(float((recent_vol - prev_vol) / prev_vol * 100), 2)

    return result
