"""テクニカル指標の計算"""
from __future__ import annotations

import math

import pandas as pd

TRADING_DAYS = 260   # FXの年間営業日数 (平日24時間市場)
VALUE_WINDOW = 1300  # 5年平均乖離の計算窓 (約5年分)
VALUE_MIN_DAYS = 750 # 5年平均乖離を計算する最低日数 (約3年)


def compute_technical(prices: pd.DataFrame) -> dict | None:
    """1ペアのレート履歴からテクニカル指標を計算する。

    prices: index=date, columns=[open, high, low, close]
    戻り値: 指標のdict。データ不足なら None
    """
    if prices is None or len(prices) < 200:
        return None

    close = prices["close"]
    last = close.iloc[-1]

    sma50 = close.rolling(50).mean().iloc[-1]
    sma200 = close.rolling(200).mean().iloc[-1]

    # RSI(14) - Wilder方式
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, pd.NA)
    rsi = (100 - 100 / (1 + rs)).iloc[-1]

    # MACD(12,26,9) ヒストグラム。レート水準に依存しないよう終値で正規化する
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist_norm = float((macd - signal).iloc[-1] / last)

    ret_63d = float(last / close.iloc[-63] - 1) if len(close) >= 63 else None

    # 年率ボラティリティ (直近60日の日次リターン標準偏差を年率換算)
    vol_60d = close.pct_change().rolling(60).std().iloc[-1]
    vol_60d = float(vol_60d * math.sqrt(TRADING_DAYS)) if pd.notna(vol_60d) else None

    # 5年平均からの乖離 (マイナス = 長期平均より安い)
    value_dev = None
    if len(close) >= VALUE_MIN_DAYS:
        value_dev = float(last / close.tail(VALUE_WINDOW).mean() - 1)

    return {
        "price": float(last),
        "sma50_ratio": float(last / sma50 - 1) if sma50 else None,
        "sma200_ratio": float(last / sma200 - 1) if sma200 else None,
        "rsi": float(rsi) if pd.notna(rsi) else None,
        "macd_hist": macd_hist_norm,
        "ret_63d": ret_63d,
        "vol_60d": vol_60d,
        "value_dev": value_dev,
    }
