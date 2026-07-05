"""マルチタイムフレーム (15分/1時間/4時間) のトレンドそろい判定

各時間足のトレンドを EMA で判定し、3つの足が同方向のペアを抽出する:
- 上昇: 終値 > EMA20 > EMA50
- 下降: 終値 < EMA20 < EMA50
- それ以外は中立

4時間足は yfinance の1時間足をリサンプルして作る (UTC 0時起点)。
出力: Discord通知 + reports/mtf.html (コミットせず Pages デプロイのみ)
"""
from __future__ import annotations

import argparse
import html
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
import yaml
import yfinance as yf

from . import universe
from .report_html import STYLE

logger = logging.getLogger(__name__)

JST = timezone(timedelta(hours=9))
WEBHOOK_ENV = "DISCORD_WEBHOOK_URL"

TF_LABELS = [("m15", "15分"), ("h1", "1時間"), ("h4", "4時間")]
ARROW = {"up": "↑", "down": "↓", "flat": "→"}
MIN_BARS = 60  # EMA50の計算に必要な最低バー数


def _direction(close: pd.Series | None) -> str:
    if close is None or len(close) < MIN_BARS:
        return "flat"
    ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
    last = close.iloc[-1]
    if last > ema20 > ema50:
        return "up"
    if last < ema20 < ema50:
        return "down"
    return "flat"


def _close(data: pd.DataFrame, ticker: str, single: bool) -> pd.Series | None:
    try:
        df = data if single else data[ticker]
        return df["Close"].dropna()
    except (KeyError, TypeError):
        return None


def analyze(uni: pd.DataFrame) -> pd.DataFrame:
    tickers = uni["ticker"].tolist()
    single = len(tickers) == 1
    logger.info("15分足を取得中 (30日分)")
    d15 = yf.download(tickers, period="30d", interval="15m", group_by="ticker",
                      auto_adjust=True, threads=True, progress=False)
    logger.info("1時間足を取得中 (90日分)")
    d1h = yf.download(tickers, period="90d", interval="1h", group_by="ticker",
                      auto_adjust=True, threads=True, progress=False)

    rows = []
    for _, u in uni.iterrows():
        c15 = _close(d15, u["ticker"], single)
        c1h = _close(d1h, u["ticker"], single)
        c4h = c1h.resample("4h").last().dropna() if c1h is not None else None

        dirs = {"m15": _direction(c15), "h1": _direction(c1h), "h4": _direction(c4h)}
        values = set(dirs.values())
        aligned = "買い" if values == {"up"} else ("売り" if values == {"down"} else "")
        price = float(c15.iloc[-1]) if c15 is not None and len(c15) else None
        rows.append({**u.to_dict(), **dirs, "aligned": aligned, "price": price})
    return pd.DataFrame(rows)


def _fmt_price(price: float | None, quote: str) -> str:
    if price is None:
        return "-"
    return f"{price:.3f}" if quote == "JPY" else f"{price:.5f}"


def _arrows(r: pd.Series) -> str:
    return "".join(ARROW[r[tf]] for tf, _ in TF_LABELS)


def send_discord(result: pd.DataFrame, stamp: str) -> bool:
    url = os.environ.get(WEBHOOK_ENV)
    if not url:
        logger.info("環境変数 %s が未設定のため通知をスキップします", WEBHOOK_ENV)
        return False

    buys = result[result["aligned"] == "買い"]["pair"].tolist()
    sells = result[result["aligned"] == "売り"]["pair"].tolist()
    lines = [
        "🟢 買いそろい: " + ("、".join(buys) if buys else "なし"),
        "🔴 売りそろい: " + ("、".join(sells) if sells else "なし"),
        "",
        "```",
        "ペア      15分 1時間 4時間",
    ]
    for _, r in result.iterrows():
        mark = " ◀" if r["aligned"] else ""
        lines.append(f"{r['pair']:<9} {ARROW[r['m15']]}    {ARROW[r['h1']]}     "
                     f"{ARROW[r['h4']]}{mark}")
    lines.append("```")

    payload = {
        "embeds": [{
            "title": f"⏱ MTFトレンドそろい {stamp}",
            "description": "\n".join(lines)[:4000],
            "color": 0x8E44AD,
            "footer": {"text": "判定: 終値>EMA20>EMA50=↑ ／ 終値<EMA20<EMA50=↓ ／ その他=→"},
        }]
    }
    resp = requests.post(url, json=payload, timeout=30)
    if resp.status_code >= 300:
        logger.error("Discord通知失敗: %d %s", resp.status_code, resp.text[:200])
        return False
    logger.info("Discord通知を送信しました")
    return True


def save_html(result: pd.DataFrame, stamp: str, output_dir: Path) -> Path:
    def row_html(r):
        judge = (f'<span class="cpos">買いそろい</span>' if r["aligned"] == "買い"
                 else f'<span class="cneg">売りそろい</span>' if r["aligned"] == "売り"
                 else "-")
        cells = "".join(
            f"<td>{ARROW[r[tf]]}</td>" for tf, _ in TF_LABELS)
        return (f'<tr><td class="l"><strong>{html.escape(r["pair"])}</strong></td>'
                f'<td class="l name">{html.escape(str(r["name"]))}</td>'
                f"<td>{_fmt_price(r['price'], r['quote'])}</td>"
                f"{cells}<td>{judge}</td></tr>")

    head = ('<tr><th class="l">ペア</th><th class="l">名称</th><th>レート</th>'
            + "".join(f"<th>{label}</th>" for _, label in TF_LABELS)
            + "<th>判定</th></tr>")
    rows = "".join(row_html(r) for _, r in result.iterrows())
    buys = len(result[result["aligned"] == "買い"])
    sells = len(result[result["aligned"] == "売り"])

    body = f"""<style>{STYLE}</style>
<div class="scr"><div class="wrap">
<header>
  <h1>マルチタイムフレーム トレンドそろい</h1>
  <div class="date">{stamp} 更新 (平日4時間ごと)</div>
  <div class="stats">
    <div class="stat"><div class="k">買いそろい</div><div class="v">{buys}</div></div>
    <div class="stat"><div class="k">売りそろい</div><div class="v">{sells}</div></div>
    <div class="stat"><div class="k">対象ペア</div><div class="v">{len(result)}</div></div>
  </div>
</header>
<section>
  <h2>15分足・1時間足・4時間足のトレンド方向</h2>
  <p class="note">↑ = 終値 &gt; EMA20 &gt; EMA50 ／ ↓ = 終値 &lt; EMA20 &lt; EMA50 ／ → = 中立。3つそろったペアが売買候補</p>
  <div class="tblwrap"><table><thead>{head}</thead><tbody>{rows}</tbody></table></div>
</section>
<footer>週次スクリーニングは<a href="index.html">こちら</a>。本ページは機械的な判定であり投資助言ではありません。データ: Yahoo Finance (yfinance)</footer>
</div></div>"""

    page = (
        "<!doctype html>\n<html lang=\"ja\">\n<head>\n"
        "<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"<title>MTFトレンドそろい {stamp}</title>\n"
        "</head>\n<body style=\"margin:0\">\n"
        f"{body}\n</body>\n</html>\n"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "mtf.html"
    path.write_text(page, encoding="utf-8")
    return path


def run(args: argparse.Namespace) -> None:
    base = Path(args.config).resolve().parent
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    stamp = datetime.now(JST).strftime("%Y-%m-%d %H:%M JST")

    uni = universe.get_universe(args.limit)
    result = analyze(uni)
    if result.empty:
        logger.error("判定対象のペアがありません")
        return

    path = save_html(result, stamp, base / cfg["report"]["output_dir"])
    logger.info("MTFページを保存: %s", path)

    if cfg["notify"]["enabled"] and not args.no_notify:
        send_discord(result, stamp)

    cols = ["pair", "m15", "h1", "h4", "aligned"]
    print(result[cols].to_string(index=False))
