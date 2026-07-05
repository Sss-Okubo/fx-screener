"""マルチタイムフレーム (1時間/4時間/日足) のトレンドそろい判定 + エントリー候補判定

各時間足のトレンドを EMA で判定し、3つの足が同方向のペアを抽出する:
- 上昇: 終値 > EMA20 > EMA50 ／ 下降: 終値 < EMA20 < EMA50 ／ それ以外は中立

そろったペアには裁量チェックリスト5項目を自動判定し、全通過なら
「🎯 エントリー候補」として通知する:
① そろいたて (4時間前の時点では同方向にそろっていなかった)
② 押し目圏 (1時間足終値がEMA20から0.5×ATR14以内)
③ 過熱なし (日足BB±2σ以内 かつ 5年平均乖離が方向に対して15%以内)
④ スワップ受取方向 (config.yaml の政策金利差)
⑤ 48時間以内に重要指標なし (config.yaml の events + 毎月第1金曜=米雇用統計)

2026-07-05に15分/1時間/4時間から上位足版へ変更 (バックテストで下位足は
ノイズによる往復が多いと判明したため)。売買ルールではなく入り口発見用。
出力: Discord通知 + reports/mtf.html (コミットせず Pages デプロイのみ)
"""
from __future__ import annotations

import argparse
import html
import logging
import os
from datetime import date, datetime, timedelta, timezone
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

TF_LABELS = [("h1", "1時間"), ("h4", "4時間"), ("d1", "日足")]
ARROW = {"up": "↑", "down": "↓", "flat": "→"}
MIN_BARS = 60          # EMA50の計算に必要な最低バー数
PULLBACK_ATR = 0.5     # ②: EMA20からの許容乖離 (ATR倍率)
STRETCH_SIGMA = 2.0    # ③: 日足BBの許容σ
STRETCH_DEV = 0.15     # ③: 5年平均乖離の許容幅 (方向に対して)
EVENT_HOURS = 48       # ⑤: 指標前の回避時間
CHECK_LABELS = [
    ("fresh", "①そろいたて"), ("pullback", "②押し目圏"),
    ("calm", "③過熱なし"), ("swap", "④スワップ受取"), ("no_event", "⑤指標なし"),
]


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


def _df(data: pd.DataFrame, ticker: str, single: bool) -> pd.DataFrame | None:
    try:
        df = data if single else data[ticker]
        return df.dropna(subset=["Close"])
    except (KeyError, TypeError):
        return None


def _aligned(dirs: dict) -> str:
    values = set(dirs.values())
    return "買い" if values == {"up"} else ("売り" if values == {"down"} else "")


def _atr14(df: pd.DataFrame) -> float:
    h, l, c = df["High"], df["Low"], df["Close"]
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return float(tr.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1])


def _upcoming_events(cfg: dict) -> list[str]:
    """48時間以内の重要指標 (config の events + 毎月第1金曜=米雇用統計)"""
    today = datetime.now(JST).date()
    events = []
    for e in cfg.get("events", []) or []:
        d = date.fromisoformat(str(e["date"]))
        if 0 <= (d - today).days * 24 <= EVENT_HOURS:
            events.append(f"{d:%m/%d} {e['label']}")
    # 米雇用統計 = 当月/翌月の第1金曜
    for month_offset in (0, 1):
        y, m = today.year, today.month + month_offset
        if m > 12:
            y, m = y + 1, m - 12
        d = date(y, m, 1)
        d += timedelta(days=(4 - d.weekday()) % 7)  # 第1金曜
        if 0 <= (d - today).days * 24 <= EVENT_HOURS:
            events.append(f"{d:%m/%d} 米雇用統計")
    return events


def _checks(u: pd.Series, direction: str, df1h: pd.DataFrame, c1d: pd.Series,
            rates: dict[str, float], events: list[str]) -> dict:
    """そろい済みペアの裁量チェックリスト①〜⑤を自動判定"""
    d = 1 if direction == "買い" else -1
    c1h = df1h["Close"]

    # ① そろいたて: 4時間前 (1時間足4本前) の時点では同方向にそろっていなかった
    c1h_prev = c1h.iloc[:-4]
    prev_dirs = {
        "h1": _direction(c1h_prev),
        "h4": _direction(c1h_prev.resample("4h").last().dropna()),
        "d1": _direction(c1d),  # 日足は4時間では実質変わらないため近似
    }
    fresh = _aligned(prev_dirs) != direction

    # ② 押し目圏: 1時間足終値が EMA20 から 0.5×ATR14 以内
    ema20 = float(c1h.ewm(span=20, adjust=False).mean().iloc[-1])
    atr = _atr14(df1h)
    ema_dist_atr = abs(float(c1h.iloc[-1]) - ema20) / atr if atr > 0 else 99.0
    pullback = ema_dist_atr <= PULLBACK_ATR

    # ③ 過熱なし: 日足BB±2σ以内 かつ 5年平均乖離が方向に対して15%以内
    daily_z, value_dev = None, None
    calm = True
    if len(c1d) >= MIN_BARS:
        sma = c1d.rolling(20).mean().iloc[-1]
        sd = c1d.rolling(20).std(ddof=0).iloc[-1]
        if pd.notna(sd) and sd > 0:
            daily_z = float((c1d.iloc[-1] - sma) / sd)
            calm = calm and (daily_z * d <= STRETCH_SIGMA)
    if len(c1d) >= 750:
        value_dev = float(c1d.iloc[-1] / c1d.tail(1300).mean() - 1)
        calm = calm and (value_dev * d <= STRETCH_DEV)

    # ④ スワップ受取方向
    carry_dir = (rates.get(u["base"], 0.0) - rates.get(u["quote"], 0.0)) * d
    swap = carry_dir > 0

    # ⑤ 48時間以内に重要指標なし (全ペア共通)
    no_event = not events

    checks = {"fresh": fresh, "pullback": pullback, "calm": calm,
              "swap": swap, "no_event": no_event}
    return {
        **checks,
        "candidate": all(checks.values()),
        "carry_dir": carry_dir,
        "ema_dist_atr": ema_dist_atr,
        "daily_z": daily_z,
        "value_dev": value_dev,
    }


def analyze(uni: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    tickers = uni["ticker"].tolist()
    single = len(tickers) == 1
    logger.info("1時間足を取得中 (365日分)")
    d1h = yf.download(tickers, period="365d", interval="1h", group_by="ticker",
                      auto_adjust=True, threads=True, progress=False)
    logger.info("日足を取得中 (6年分)")
    d1d = yf.download(tickers, period="6y", interval="1d", group_by="ticker",
                      auto_adjust=True, threads=True, progress=False)

    rates: dict[str, float] = cfg["rates"]["policy"]
    events = _upcoming_events(cfg)
    if events:
        logger.info("48時間以内の重要指標: %s", " / ".join(events))

    rows = []
    for _, u in uni.iterrows():
        df1h = _df(d1h, u["ticker"], single)
        df1d = _df(d1d, u["ticker"], single)
        c1h = df1h["Close"] if df1h is not None else None
        c1d = df1d["Close"] if df1d is not None else None
        c4h = c1h.resample("4h").last().dropna() if c1h is not None else None

        dirs = {"h1": _direction(c1h), "h4": _direction(c4h), "d1": _direction(c1d)}
        aligned = _aligned(dirs)
        price = float(c1h.iloc[-1]) if c1h is not None and len(c1h) else None
        row = {**u.to_dict(), **dirs, "aligned": aligned, "price": price,
               "candidate": False}
        if aligned and df1h is not None and c1d is not None:
            row.update(_checks(u, aligned, df1h, c1d, rates, events))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.attrs["events"] = events
    return df


def _fmt_price(price: float | None, quote: str) -> str:
    if price is None:
        return "-"
    return f"{price:.3f}" if quote == "JPY" else f"{price:.5f}"


def send_discord(result: pd.DataFrame, stamp: str) -> bool:
    url = os.environ.get(WEBHOOK_ENV)
    if not url:
        logger.info("環境変数 %s が未設定のため通知をスキップします", WEBHOOK_ENV)
        return False

    embeds = []

    # 🎯 エントリー候補 (チェックリスト全通過)
    candidates = result[result["candidate"] == True]  # noqa: E712
    if not candidates.empty:
        lines = []
        for _, r in candidates.iterrows():
            z = f"{r['daily_z']:+.1f}σ" if pd.notna(r.get("daily_z")) else "-"
            dev = (f"{r['value_dev'] * 100:+.1f}%"
                   if pd.notna(r.get("value_dev")) else "-")
            lines.append(f"**{r['pair']} {r['aligned']}** {r['name']}")
            lines.append(f"┗ スワップ {r['carry_dir']:+.2f}%受取"
                         f" ／ EMA20乖離 {r['ema_dist_atr']:.1f}ATR"
                         f" ／ 日足 {z} ／ 5年乖離 {dev}")
        lines.append("")
        lines.append("損切り目安: 建値−1.5×ATR(1時間足) ／ ロットは損失が資金の1%以下になるよう調整")
        embeds.append({
            "title": f"🎯 エントリー候補 (チェックリスト全通過) {stamp}",
            "description": "\n".join(lines)[:4000],
            "color": 0xF1C40F,
            "footer": {"text": "①そろいたて ②押し目圏 ③過熱なし ④スワップ受取 ⑤指標なし を全て満たしたペア。最終判断はご自身で。"},
        })

    # 通常のマトリクス
    buys = result[result["aligned"] == "買い"]["pair"].tolist()
    sells = result[result["aligned"] == "売り"]["pair"].tolist()
    lines = [
        "🟢 買いそろい: " + ("、".join(buys) if buys else "なし"),
        "🔴 売りそろい: " + ("、".join(sells) if sells else "なし"),
        "",
        "```",
        "ペア      1時間 4時間 日足",
    ]
    for _, r in result.iterrows():
        mark = " ◀" if r["aligned"] else ""
        if r.get("candidate"):
            mark = " 🎯"
        lines.append(f"{r['pair']:<9} {ARROW[r['h1']]}     {ARROW[r['h4']]}     "
                     f"{ARROW[r['d1']]}{mark}")
    lines.append("```")
    events = result.attrs.get("events", [])
    if events:
        lines.append("⚠ 48時間以内の指標: " + " / ".join(events))
    embeds.append({
        "title": f"⏱ MTFトレンドそろい {stamp}",
        "description": "\n".join(lines)[:4000],
        "color": 0x8E44AD,
        "footer": {"text": "判定: 終値>EMA20>EMA50=↑ ／ 終値<EMA20<EMA50=↓ ／ その他=→"},
    })

    resp = requests.post(url, json={"embeds": embeds}, timeout=30)
    if resp.status_code >= 300:
        logger.error("Discord通知失敗: %d %s", resp.status_code, resp.text[:200])
        return False
    logger.info("Discord通知を送信しました (候補 %d件)", len(candidates))
    return True


def save_html(result: pd.DataFrame, stamp: str, output_dir: Path) -> Path:
    def row_html(r):
        judge = (f'<span class="cpos">買いそろい</span>' if r["aligned"] == "買い"
                 else f'<span class="cneg">売りそろい</span>' if r["aligned"] == "売り"
                 else "-")
        if r.get("candidate"):
            judge += " 🎯"
        cells = "".join(f"<td>{ARROW[r[tf]]}</td>" for tf, _ in TF_LABELS)
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
    n_cand = int(result["candidate"].sum())

    # そろい済みペアのチェックリスト表
    aligned_df = result[result["aligned"] != ""]
    check_section = ""
    if not aligned_df.empty:
        chead = ('<tr><th class="l">ペア</th><th class="l">方向</th>'
                 + "".join(f"<th>{label}</th>" for _, label in CHECK_LABELS)
                 + "<th>判定</th></tr>")
        crows = []
        for _, r in aligned_df.iterrows():
            marks = "".join(
                f"<td>{'○' if r.get(key) else '×'}</td>" for key, _ in CHECK_LABELS)
            verdict = "🎯 候補" if r.get("candidate") else "見送り"
            crows.append(
                f'<tr><td class="l"><strong>{html.escape(r["pair"])}</strong></td>'
                f'<td class="l">{r["aligned"]}</td>{marks}<td>{verdict}</td></tr>')
        check_section = f"""
<section>
  <h2>エントリー前チェックリスト (そろい済みペア)</h2>
  <p class="note">①4時間前は未そろい ／ ②1時間足EMA20から{PULLBACK_ATR}ATR以内 ／ ③日足±{STRETCH_SIGMA:.0f}σ以内かつ5年乖離{STRETCH_DEV * 100:.0f}%以内 ／ ④政策金利差が受取方向 ／ ⑤48時間以内に重要指標なし</p>
  <div class="tblwrap"><table><thead>{chead}</thead><tbody>{''.join(crows)}</tbody></table></div>
</section>"""

    events = result.attrs.get("events", [])
    event_note = ("⚠ 48時間以内の指標: " + " / ".join(events)) if events else ""

    body = f"""<style>{STYLE}</style>
<div class="scr"><div class="wrap">
<header>
  <h1>マルチタイムフレーム トレンドそろい</h1>
  <div class="date">{stamp} 更新 (平日4時間ごと) {event_note}</div>
  <div class="stats">
    <div class="stat"><div class="k">🎯 候補</div><div class="v">{n_cand}</div></div>
    <div class="stat"><div class="k">買いそろい</div><div class="v">{buys}</div></div>
    <div class="stat"><div class="k">売りそろい</div><div class="v">{sells}</div></div>
    <div class="stat"><div class="k">対象ペア</div><div class="v">{len(result)}</div></div>
  </div>
</header>
<section>
  <h2>1時間足・4時間足・日足のトレンド方向</h2>
  <p class="note">↑ = 終値 &gt; EMA20 &gt; EMA50 ／ ↓ = 終値 &lt; EMA20 &lt; EMA50 ／ → = 中立。3つそろい、かつチェックリスト全通過で 🎯</p>
  <div class="tblwrap"><table><thead>{head}</thead><tbody>{rows}</tbody></table></div>
</section>
{check_section}
<footer>週次スクリーニングは<a href="index.html">こちら</a>。本ページは機械的な判定であり投資助言ではありません。バックテストでは機械的売買の優位性は確認できていません(入り口発見用)。データ: Yahoo Finance (yfinance)</footer>
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
    result = analyze(uni, cfg)
    if result.empty:
        logger.error("判定対象のペアがありません")
        return

    path = save_html(result, stamp, base / cfg["report"]["output_dir"])
    logger.info("MTFページを保存: %s", path)

    if cfg["notify"]["enabled"] and not args.no_notify:
        send_discord(result, stamp)

    cols = ["pair", "h1", "h4", "d1", "aligned", "candidate"]
    print(result[cols].to_string(index=False))
