"""Discord Webhook 通知"""
from __future__ import annotations

import logging
import os

import pandas as pd
import requests

logger = logging.getLogger(__name__)

WEBHOOK_ENV = "DISCORD_WEBHOOK_URL"


def send_discord(ranked: pd.DataFrame, run_date: str, top_n: int,
                 new_tickers: set[str]) -> bool:
    url = os.environ.get(WEBHOOK_ENV)
    if not url:
        logger.info("環境変数 %s が未設定のため通知をスキップします", WEBHOOK_ENV)
        return False

    def fmt(v, suffix="", pct=False, digits=1):
        if v is None or pd.isna(v):
            return "-"
        return f"{v * 100:+.{digits}f}%" if pct else f"{v:+.{digits}f}{suffix}"

    lines = []
    for _, r in ranked.head(top_n).iterrows():
        mark = " 🆕" if r["ticker"] in new_tickers else ""
        lines.append(f"**{r['rank']}. {r['pair']}** {r['name']} "
                     f"({r['market']}) スコア {r['score']:.1f}{mark}")
        rsi = f"{r['rsi']:.0f}" if pd.notna(r["rsi"]) else "-"
        lines.append(f"┗ 金利差 {fmt(r['carry'], suffix='%', digits=2)}"
                     f" ／ 5年乖離 {fmt(r['value_dev'], pct=True)}"
                     f" ／ 3ヶ月 {fmt(r['ret_63d'], pct=True)}"
                     f" ／ RSI {rsi}")

    payload = {
        "embeds": [{
            "title": f"💱 FXスクリーニング結果 {run_date}",
            "description": "\n".join(lines)[:4000],
            "color": 0x1E8449,
            "footer": {"text": "スコアはロング(買い)の魅力度。機械的スクリーニングです。投資判断は自己責任で。"},
        }]
    }
    resp = requests.post(url, json=payload, timeout=30)
    if resp.status_code >= 300:
        logger.error("Discord通知失敗: %d %s", resp.status_code, resp.text[:200])
        return False
    logger.info("Discord通知を送信しました")
    return True
