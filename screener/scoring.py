"""複合スコアリング: 各ペアを買い(ロング)・売り(ショート)の両方向で評価する

各ペアを2行(買い/売り)に展開し、方向依存の指標は売り行で符号を反転してから
全行まとめて偏差値化 → 重み付き合算する。ボラティリティは両方向共通。
"""
from __future__ import annotations

import logging

import pandas as pd

from .indicators import compute_technical
from .store import Store

logger = logging.getLogger(__name__)

# カテゴリ → (指標列, 符号)。符号 -1 は「小さいほど良い」指標
CATEGORY_METRICS: dict[str, list[tuple[str, int]]] = {
    "carry": [("carry", 1)],
    "value": [("value_dev", -1)],
    "trend": [("sma50_ratio", 1), ("sma200_ratio", 1)],
    "momentum": [("ret_63d", 1), ("macd_hist", 1)],
    "stability": [("vol_60d", -1)],
}

# 売り行で符号を反転する方向依存の指標 (ボラティリティは方向に依らない)
DIRECTIONAL_COLS = {"carry", "value_dev", "sma50_ratio", "sma200_ratio",
                    "ret_63d", "macd_hist"}


def _zscore(series: pd.Series) -> pd.Series:
    """外れ値の影響を抑えるため上下1%でウィンザライズしてからzスコア化。欠損は0(中立)扱い"""
    s = series.astype(float)
    valid = s.dropna()
    if len(valid) < 5 or valid.std(ddof=0) == 0:
        return pd.Series(0.0, index=series.index)
    lo, hi = valid.quantile(0.01), valid.quantile(0.99)
    s = s.clip(lo, hi)
    z = (s - s.mean()) / s.std(ddof=0)
    return z.fillna(0.0).clip(-3, 3)


def build_features(store: Store, universe: pd.DataFrame,
                   policy_rates: dict[str, float]) -> pd.DataFrame:
    """テクニカル指標と金利差(キャリー)を結合した特徴量テーブルを作る"""
    tech_rows = []
    for t in universe["ticker"]:
        tech = compute_technical(store.load_prices(t))
        tech_rows.append({"ticker": t, **(tech or {})})
    df = universe.merge(pd.DataFrame(tech_rows), on="ticker", how="left")

    # 金利差 (%): 基軸通貨の政策金利 − 決済通貨の政策金利。ロングの概算スワップ方向
    df["carry"] = df.apply(
        lambda r: policy_rates.get(r["base"], 0.0) - policy_rates.get(r["quote"], 0.0),
        axis=1)

    # 価格データが無いペアは選定対象外
    before = len(df)
    df = df.dropna(subset=["price"]).reset_index(drop=True)
    if before - len(df):
        logger.info("価格データ不足のため %dペアを除外", before - len(df))
    return df


def score(features: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """各ペアを買い/売りの2行に展開してスコアリングし、ランキングを返す。

    表示列 (5年乖離・3ヶ月・RSI等) はペアの生の値のまま。方向の反転は
    スコア計算時とスワップ表示用の carry_dir にのみ適用する。
    """
    weights: dict[str, float] = cfg["weights"]

    longs = features.copy()
    longs["direction"], longs["dir"] = "買い", 1
    shorts = features.copy()
    shorts["direction"], shorts["dir"] = "売り", -1
    df = pd.concat([longs, shorts], ignore_index=True)

    # その方向で受け取る金利差 (プラス = スワップ受取の概算)。+0.0 で -0.0 表示を防ぐ
    df["carry_dir"] = df["carry"] * df["dir"] + 0.0

    for cat, metrics in CATEGORY_METRICS.items():
        zs = []
        for col, sign in metrics:
            eff = df[col] * df["dir"] if col in DIRECTIONAL_COLS else df[col]
            zs.append(_zscore(eff) * sign)
        df[f"z_{cat}"] = pd.concat(zs, axis=1).mean(axis=1)

    df["composite"] = sum(df[f"z_{cat}"] * w for cat, w in weights.items())

    # RSI過熱ペナルティ: 買いは買われすぎ、売りは売られすぎで減点
    over = (df["dir"] == 1) & (df["rsi"] > cfg["rsi_overbought"])
    under = (df["dir"] == -1) & (df["rsi"] < 100 - cfg["rsi_overbought"])
    df.loc[over | under, "composite"] -= cfg["rsi_penalty"] * sum(weights.values())

    # 表示用に偏差値 (50 + 10z) へ変換
    comp = df["composite"]
    df["score"] = 50 + 10 * (comp - comp.mean()) / comp.std(ddof=0)

    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    # 順位は方向内で 1 から振る
    df["rank"] = df.groupby("direction")["score"] \
                   .rank(ascending=False, method="first").astype(int)
    return df
