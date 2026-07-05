"""通貨ペアユニバース (主要18ペアの固定リスト)"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# (yfinanceティッカー, 表示ペア, 日本語名, 基軸通貨, 決済通貨, 区分)
PAIRS: list[tuple[str, str, str, str, str, str]] = [
    # 円クロス
    ("USDJPY=X", "USD/JPY", "米ドル/円", "USD", "JPY", "円クロス"),
    ("EURJPY=X", "EUR/JPY", "ユーロ/円", "EUR", "JPY", "円クロス"),
    ("GBPJPY=X", "GBP/JPY", "英ポンド/円", "GBP", "JPY", "円クロス"),
    ("AUDJPY=X", "AUD/JPY", "豪ドル/円", "AUD", "JPY", "円クロス"),
    ("NZDJPY=X", "NZD/JPY", "NZドル/円", "NZD", "JPY", "円クロス"),
    ("CADJPY=X", "CAD/JPY", "カナダドル/円", "CAD", "JPY", "円クロス"),
    ("CHFJPY=X", "CHF/JPY", "スイスフラン/円", "CHF", "JPY", "円クロス"),
    # ドルストレート
    ("EURUSD=X", "EUR/USD", "ユーロ/米ドル", "EUR", "USD", "ドルスト"),
    ("GBPUSD=X", "GBP/USD", "英ポンド/米ドル", "GBP", "USD", "ドルスト"),
    ("AUDUSD=X", "AUD/USD", "豪ドル/米ドル", "AUD", "USD", "ドルスト"),
    ("NZDUSD=X", "NZD/USD", "NZドル/米ドル", "NZD", "USD", "ドルスト"),
    ("USDCAD=X", "USD/CAD", "米ドル/カナダドル", "USD", "CAD", "ドルスト"),
    ("USDCHF=X", "USD/CHF", "米ドル/スイスフラン", "USD", "CHF", "ドルスト"),
    # その他クロス
    ("EURGBP=X", "EUR/GBP", "ユーロ/英ポンド", "EUR", "GBP", "クロス"),
    ("EURAUD=X", "EUR/AUD", "ユーロ/豪ドル", "EUR", "AUD", "クロス"),
    ("GBPAUD=X", "GBP/AUD", "英ポンド/豪ドル", "GBP", "AUD", "クロス"),
    ("AUDNZD=X", "AUD/NZD", "豪ドル/NZドル", "AUD", "NZD", "クロス"),
    ("EURCHF=X", "EUR/CHF", "ユーロ/スイスフラン", "EUR", "CHF", "クロス"),
]


def get_universe(limit: int | None = None) -> pd.DataFrame:
    """戻り値: columns=[ticker, pair, name, base, quote, market]

    スコアは「基軸通貨の買い(ロング)」の魅力度として評価する。
    """
    df = pd.DataFrame(PAIRS, columns=["ticker", "pair", "name", "base", "quote", "market"])
    if limit:
        df = df.head(limit)
    logger.info("ユニバース: %dペア", len(df))
    return df.reset_index(drop=True)
