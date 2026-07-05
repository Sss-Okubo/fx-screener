"""MTFそろいシグナルの売買検証 (ルール変種の比較)

エントリーは全変種共通: 15分/1時間/4時間足のトレンドが3つそろったバーの終値。
比較する変種:
- ① 元ルール: そろいが崩れたら即決済 (中立でも決済)
- ② 出口=15分逆転: 15分足がポジションと逆方向になるまで保有 (中立では保有継続)
- ③ 出口=1時間崩れ: 1時間足がポジション方向でなくなったら決済 (15分足は無視)
- ④ ② + 入口フレッシュ: 直前2時間(8バー)以上そろいが無かった場合のみ新規エントリー
- ⑤ ③ + 入口フレッシュ: 同上

制約: yfinance の15分足は直近60日しか取得できないため、検証期間は約2ヶ月。
1時間足は365日分を取得して 1時間/4時間足のEMAを十分にウォームアップさせる。
進行中の1時間/4時間バーは、最新の15分終値でEMAを1ステップ仮更新して判定する
(ライブ判定と同等の扱い)。建玉中は各変種の決済条件のみを監視する。
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

# (名称, 決済モード, 入口フレッシュ条件の最低非そろいバー数)
VARIANTS: list[tuple[str, str, int]] = [
    ("① 元ルール(崩れで即決済)", "align", 0),
    ("② 出口=15分逆転", "m15", 0),
    ("③ 出口=1時間崩れ", "h1", 0),
    ("④ ②+入口フレッシュ(2h)", "m15", 8),
    ("⑤ ③+入口フレッシュ(2h)", "h1", 8),
]


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


def _simulate(c15: pd.Series, sig: np.ndarray, d15: np.ndarray, d1h: np.ndarray,
              cost: float, exit_mode: str, fresh_bars: int) -> list[dict]:
    prices = c15.to_numpy(float)
    times = c15.index
    trades: list[dict] = []
    pos, p_in, t_in = 0, 0.0, None
    last_nz = -10 ** 9  # 直近で sig != 0 だったバーの位置

    for i in range(len(prices)):
        s = int(sig[i])
        if pos != 0:
            if exit_mode == "align":
                ex = s != pos
            elif exit_mode == "m15":
                ex = d15[i] == -pos
            else:  # h1
                ex = d1h[i] != pos
            if ex:
                trades.append({
                    "entry": t_in, "exit": times[i], "side": pos,
                    "ret": (prices[i] / p_in - 1) * pos - cost,
                    "hours": (times[i] - t_in).total_seconds() / 3600,
                    "forced": False,
                })
                pos = 0
        if pos == 0 and s != 0:
            # 新規はそろいの初回バーのみ。フレッシュ条件は直前の非そろい期間で判定
            run_start = i == 0 or int(sig[i - 1]) != s
            if run_start and (i - last_nz - 1) >= fresh_bars:
                pos, p_in, t_in = s, prices[i], times[i]
        if s != 0:
            last_nz = i
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


def _summary(trades: list[dict], cost: float) -> dict:
    rets = np.array([t["ret"] for t in trades])
    return {
        "trades": len(trades),
        "win": float((rets > 0).mean()) if len(rets) else float("nan"),
        "avg_bp": float(rets.mean() * 10000) if len(rets) else float("nan"),
        "total": float(rets.sum()),
        "total_gross": float((rets + cost).sum()),
        "hours": float(np.mean([t["hours"] for t in trades])) if trades else 0.0,
    }


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

    all_trades: dict[str, list[dict]] = {name: [] for name, _, _ in VARIANTS}
    period_start, period_end = None, None
    for _, u in uni.iterrows():
        c15 = _close(d15, u["ticker"], single)
        c1h = _close(d1h, u["ticker"], single)
        if c15 is None or c1h is None or len(c15) < 200:
            logger.warning("%s: データ不足のためスキップ", u["pair"])
            continue
        period_start = min(period_start or c15.index[0], c15.index[0])
        period_end = max(period_end or c15.index[-1], c15.index[-1])

        dir15 = _dirs_15m(c15)
        dir1h = _dirs_higher(c15, c1h, "1h")
        dir4h = _dirs_higher(c15, c1h, "4h")
        sig = np.where((dir15 == 1) & (dir1h == 1) & (dir4h == 1), 1,
                       np.where((dir15 == -1) & (dir1h == -1) & (dir4h == -1), -1, 0))

        for name, exit_mode, fresh in VARIANTS:
            trades = _simulate(c15, sig, dir15, dir1h, cost, exit_mode, fresh)
            for t in trades:
                t["pair"] = u["pair"]
            all_trades[name] += trades

    if not any(all_trades.values()):
        logger.error("トレードが1件も発生しませんでした")
        return

    # 変種比較表
    lines = [
        f"# MTFそろい売買検証 (ルール変種比較) {run_date}",
        "",
        "エントリーは全変種共通: 3つの足がそろったバーの終値 (④⑤は直前2時間以上の非そろいが条件)。",
        f"検証期間: {period_start:%Y-%m-%d} 〜 {period_end:%Y-%m-%d} "
        f"(約{(period_end - period_start).days}日間 ／ yfinanceの15分足の上限)",
        f"取引コスト: 往復 {args.cost_bp:.1f}bp ／ 1トレード=固定ロット、リターンは単純合算",
        "",
        "## 変種比較 (全10ペア合算)",
        "",
        "| 変種 | 回数 | 勝率 | 平均 | 合計(コスト後) | 合計(コスト前) | 平均保有 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    best_name, best_total = None, -1e9
    for name, _, _ in VARIANTS:
        s = _summary(all_trades[name], cost)
        if s["total"] > best_total:
            best_name, best_total = name, s["total"]
        lines.append(
            f"| {name} | {s['trades']} | {s['win'] * 100:.1f}% | {s['avg_bp']:+.1f}bp "
            f"| {s['total'] * 100:+.2f}% | {s['total_gross'] * 100:+.2f}% "
            f"| {s['hours']:.1f}h |")

    # 最良変種のペア別内訳
    best_trades = all_trades[best_name]
    lines += ["", f"## ペア別内訳 (最良変種: {best_name} / コスト後)", "",
              "| ペア | 回数 | 買/売 | 勝率 | 平均 | 合計 | 平均保有 |",
              "|---|---:|---|---:|---:|---:|---:|"]
    bdf = pd.DataFrame(best_trades)
    for pair, g in bdf.groupby("pair"):
        rets = g["ret"].to_numpy()
        buys = int((g["side"] == 1).sum())
        lines.append(
            f"| {pair} | {len(g)} | {buys}/{len(g) - buys} "
            f"| {(rets > 0).mean() * 100:.0f}% | {rets.mean() * 10000:+.1f}bp "
            f"| {rets.sum() * 100:+.2f}% | {g['hours'].mean():.1f}h |")

    lines += [
        "",
        "## 注意事項",
        "",
        "- **検証期間は約2ヶ月のみ** (yfinanceの15分足は直近60日が上限)。この期間の相場環境に強く依存し、統計的な確度は低い",
        "- 終値ベースの約定でスリッページ未考慮。スワップ損益も未考慮",
        "- 上位足の進行中バーは15分終値でEMAを仮更新して判定 (ライブ判定に近い扱いだが完全一致ではない)",
        "- 建玉中は各変種の決済条件のみを監視し、逆方向のそろいが出ても決済条件を満たすまで保有",
        "- 期間末の未決済ポジションは最終バーで強制決済として集計",
        "- 過去の成績は将来の成果を保証しない。投資判断は自己責任で",
    ]
    out_dir = base / cfg["report"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    bdf_out = bdf.copy()
    bdf_out["side"] = bdf_out["side"].map({1: "買い", -1: "売り"})
    bdf_out.to_csv(out_dir / f"mtf-backtest-trades-{run_date}.csv",
                   index=False, encoding="utf-8")
    md_path = out_dir / f"mtf-backtest-{run_date}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("検証レポートを保存: %s", md_path)
    print(md_path)
