"""yfinance による為替レート取得 (SQLiteキャッシュ付き)"""
from __future__ import annotations

import logging

import yfinance as yf

from .store import Store

logger = logging.getLogger(__name__)

MIN_DAYS = 250  # 保存する最低データ日数 (200日SMAの計算に必要)


def fetch_prices(store: Store, tickers: list[str], period: str,
                 cache_days: float) -> None:
    """為替レート履歴を一括取得してキャッシュに保存する"""
    fresh = store.fresh_price_tickers(cache_days)
    targets = [t for t in tickers if t not in fresh]
    logger.info("価格取得: %dペア (キャッシュ済 %dペア)", len(targets), len(tickers) - len(targets))
    if not targets:
        return

    try:
        data = yf.download(targets, period=period, group_by="ticker",
                           auto_adjust=True, threads=True, progress=False)
    except Exception as e:  # noqa: BLE001
        logger.warning("価格取得失敗: %s", e)
        return

    for t in targets:
        try:
            df = data[t] if len(targets) > 1 else data
            df = df.dropna(subset=["Close"])
            if len(df) >= MIN_DAYS:
                store.save_prices(t, df)
            else:
                logger.warning("%s: データ不足 (%d日分)", t, len(df))
        except (KeyError, TypeError):
            logger.warning("%s: 価格データなし", t)
