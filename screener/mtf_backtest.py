"""MTFそろいシグナルの売買検証

ルール: 15分/1時間/4時間足のトレンドが3つそろったバーの終値で建て、
そろいが崩れたバーの終値で決済 (買い→売りへ直接反転した場合はドテン)。

制約: yfinance の15分足は直近60日しか取得できないため、検証期間は約2ヶ月。
1時間足は365日分を取得して 1時間/4時間足のEMAを十分にウォームアップさせる。
進行中の1時間/4時間バーは、最新の15分終値でEMAを1ステップ仮更新して判定する
(ライブ判定と同等の扱い)。
"""
from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import yfinance as yf

from . import universe

logger = logging.getLogger(__name__)


def _dirs_15m(c15: pd.Series) -> np.ndarray:
    e20 = c15.ewm(span=20, adjust=False).mean().to_numpy(float)
    e50 = c15.ewm(span=50, adjust=False).mean().to_numpy(float)
    p = c15.to_numpy(float)
    return np.where((p > e20) & (e20 > e50), 1,
                    np.where((p < e20) & (e20 < e50), -1, 0))


def _dirs_higher(c15: pd.Series, c1h: pd.Series, freq: str) -> np.ndarray:
    """上位足の方向。完了バーのEMAに、進行中バーの15分終値で1ステップ仮更新"""
    bar_close = c1h.resample(freq).last().dropna() if freq != "1h" else c1h
    bars = c15.index.floor(freq)
    p = c15.to_numpy(float)
    emas = []
    for span in (20, 50):
        ema = bar_close.ewm(span=span, adjust=False).mean()
        prev = ema.shift(1).reindex(bars, method="ffill").to_numpy(float)
        a = 2 / (span + 1)
        emas.append(a * p + (1 - a) * prev)
    e20, e50 = emas
    return np.where((p > e20) & (e20 > e50), 1,
                    np.where((p < e20) & (e20 < e50), -1, 0))


def _simulate(c15: pd.Series, sig: np.ndarray, cost: float) -> list[dict]:
    """そろいで建玉、崩れで決済。cost は往復コスト (小数)"""
    trades: list[dict] = []
    prices = c15.to_numpy(float)
    times = c15.index
    pos, p_in, t_in = 0, 0.0, None

    for i in range(len(prices)):
        s = int(sig[i])
        if pos != 0 and s != pos:
            trades.append({
                "entry": t_in, "exit": times[i], "side": pos,
                "ret": (prices[i] / p_in - 1) * pos - cost,
                "hours": (times[i] - t_in).total_seconds() / 3600,
                "forced": False,
            })
            pos = 0
        if pos == 0 and s != 0:
            pos, p_in, t_in = s, prices[i], times[i]
    if pos != 0:  # 期間末で強制決済
        trades.append({
            "entry": t_in, "exit": times[-1], "side": pos,
            "ret": (prices[-1] / p_in - 1) * pos - cost,
            "hours": (times[-1] - t_in).total_seconds() / 3600,
            "forced": True,
        })
    return trades


def _close(data: pd.DataFrame, ticker: str, single: bool) -> pd.Series | None:
    try:
        df = data if single else data[ticker]
        s = df["Close"].dropna()
        return s.tz_convert("UTC") if s.index.tz is not None else s
    except (KeyError, TypeError):
        return None


def run(args: argparse.Namespace) -> None:
    base = Path(args.config).resolve().parent
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run_date = date.today().isoformat()
    cost = args.cost_bp / 10000

    uni = universe.get_universe(args.limit)
    tickers = uni["ticker"].tolist()
    single = len(tickers) == 1
    logger.info("15分足を取得中 (60日分)")
    d15 = yf.download(tickers, period="60d", interval="15m", group_by="ticker",
                      auto_adjust=True, threads=True, progress=False)
    logger.info("1時間足を取得中 (365日分)")
    d1h = yf.download(tickers, period="365d", interval="1h", group_by="ticker",
                      auto_adjust=True, threads=True, progress=False)

    all_trades = []
    per_pair = []
    period_start, period_end = None, None
    for _, u in uni.iterrows():
        c15 = _close(d15, u["ticker"], single)
        c1h = _close(d1h, u["ticker"], single)
        if c15 is None or c1h is None or len(c15) < 200:
            logger.warning("%s: データ不足のためスキップ", u["pair"])
            continue
        period_start = min(period_start or c15.index[0], c15.index[0])
        period_end = max(period_end or c15.index[-1], c15.index[-1])

        d_15 = _dirs_15m(c15)
        d_1h = _dirs_higher(c15, c1h, "1h")
        d_4h = _dirs_higher(c15, c1h, "4h")
        sig = np.where((d_15 == 1) & (d_1h == 1) & (d_4h == 1), 1,
                       np.where((d_15 == -1) & (d_1h == -1) & (d_4h == -1), -1, 0))

        trades = _simulate(c15, sig, cost)
        for t in trades:
            t["pair"] = u["pair"]
        all_trades += trades

        if trades:
            rets = np.array([t["ret"] for t in trades])
            per_pair.append({
                "pair": u["pair"], "trades": len(trades),
                "win": float((rets > 0).mean()),
                "avg": float(rets.mean()), "total": float(rets.sum()),
                "hours": float(np.mean([t["hours"] for t in trades])),
                "buys": sum(1 for t in trades if t["side"] == 1),
            })

    if not all_trades:
        logger.error("トレードが1件も発生しませんでした")
        return

    rets = np.array([t["ret"] for t in all_trades])
    gross = rets + cost
    n = len(rets)
    summary = {
        "trades": n,
        "win": float((rets > 0).mean()),
        "avg_bp": float(rets.mean() * 10000),
        "total": float(rets.sum()),
        "total_gross": float(gross.sum()),
        "hours": float(np.mean([t["hours"] for t in all_trades])),
    }

    # レポート生成
    out_dir = base / cfg["report"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    tdf = pd.DataFrame(all_trades)
    tdf["side"] = tdf["side"].map({1: "買い", -1: "売り"})
    csv_path = out_dir / f"mtf-backtest-trades-{run_date}.csv"
    tdf.to_csv(csv_path, index=False, encoding="utf-8")

    lines = [
        f"# MTFそろい売買検証 {run_date}",
        "",
        "ルール: 15分/1時間/4時間足のEMAトレンドが3つそろったバーの終値で建て、崩れたバーの終値で決済。",
        f"検証期間: {period_start:%Y-%m-%d} 〜 {period_end:%Y-%m-%d} "
        f"(約{(period_end - period_start).days}日間 ／ yfinanceの15分足の上限)",
        f"取引コスト: 往復 {args.cost_bp:.1f}bp (スプレッド想定) ／ 1トレード=固定ロット、リターンは単純合算",
        "",
        "## 全体",
        "",
        "| 項目 | 値 |",
        "|---|---:|",
        f"| トレード数 | {summary['trades']} |",
        f"| 勝率 | {summary['win'] * 100:.1f}% |",
        f"| 平均リターン/トレード | {summary['avg_bp']:+.1f}bp |",
        f"| 合計リターン (コスト後) | {summary['total'] * 100:+.2f}% |",
        f"| 合計リターン (コスト前) | {summary['total_gross'] * 100:+.2f}% |",
        f"| 平均保有時間 | {summary['hours']:.1f}時間 |",
        "",
        "## ペア別 (コスト後)",
        "",
        "| ペア | 回数 | 買/売 | 勝率 | 平均 | 合計 | 平均保有 |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for p in sorted(per_pair, key=lambda x: -x["total"]):
        lines.append(
            f"| {p['pair']} | {p['trades']} | {p['buys']}/{p['trades'] - p['buys']} "
            f"| {p['win'] * 100:.0f}% | {p['avg'] * 10000:+.1f}bp "
            f"| {p['total'] * 100:+.2f}% | {p['hours']:.1f}h |")
    lines += [
        "",
        "## 注意事項",
        "",
        "- **検証期間は約2ヶ月のみ** (yfinanceの15分足は直近60日が上限)。この期間の相場環境に強く依存し、統計的な確度は低い",
        "- 終値ベースの約定でスリッページ未考慮。スワップ損益も未考慮",
        "- 上位足の進行中バーは15分終値でEMAを仮更新して判定 (ライブ判定に近い扱いだが完全一致ではない)",
        "- 期間末の未決済ポジションは最終バーで強制決済として集計",
        "- 過去の成績は将来の成果を保証しない。投資判断は自己責任で",
    ]
    md_path = out_dir / f"mtf-backtest-{run_date}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("検証レポートを保存: %s", md_path)
    print(md_path)
