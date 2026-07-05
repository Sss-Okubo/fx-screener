"""Discord Webhook 通知 (買い/売り 両方向)"""
from __future__ import annotations

import logging
import os

import pandas as pd
import requests

logger = logging.getLogger(__name__)

WEBHOOK_ENV = "DISCORD_WEBHOOK_URL"


def _lines(df: pd.DataFrame, new_keys: set[tuple[str, str]]) -> list[str]:
    out = []
    for _, r in df.iterrows():
        mark = " 🆕" if (r["ticker"], r["direction"]) in new_keys else ""
        rsi = f"{r['rsi']:.0f}" if pd.notna(r["rsi"]) else "-"
        swap = "受取" if r["carry_dir"] > 0 else ("支払" if r["carry_dir"] < 0 else "±0")
        ret = (f"{r['ret_63d'] * 100:+.1f}%" if pd.notna(r["ret_63d"]) else "-")
        out.append(f"**{r['rank']}. {r['pair']}** {r['name']} スコア {r['score']:.1f}{mark}")
        out.append(f"┗ スワップ{swap} {r['carry_dir']:+.2f}%"
                   f" ／ 3ヶ月 {ret} ／ RSI {rsi}")
    return out


def send_discord(ranked: pd.DataFrame, run_date: str, top_n: int,
                 new_keys: set[tuple[str, str]]) -> bool:
    url = os.environ.get(WEBHOOK_ENV)
    if not url:
        logger.info("環境変数 %s が未設定のため通知をスキップします", WEBHOOK_ENV)
        return False

    buy = ranked[ranked["direction"] == "買い"].sort_values("rank").head(top_n)
    sell = ranked[ranked["direction"] == "売り"].sort_values("rank").head(top_n)

    lines = ["__📈 買い(ロング)候補__"]
    lines += _lines(buy, new_keys)
    lines.append("")
    lines.append("__📉 売り(ショート)候補__")
    lines += _lines(sell, new_keys)

    payload = {
        "embeds": [{
            "title": f"💱 FXスクリーニング結果 {run_date}",
            "description": "\n".join(lines)[:4000],
            "color": 0x1E8449,
            "footer": {"text": "スワップは政策金利差の概算。機械的スクリーニングです。投資判断は自己責任で。"},
        }]
    }
    resp = requests.post(url, json=payload, timeout=30)
    if resp.status_code >= 300:
        logger.error("Discord通知失敗: %d %s", resp.status_code, resp.text[:200])
        return False
    logger.info("Discord通知を送信しました")
    return True
