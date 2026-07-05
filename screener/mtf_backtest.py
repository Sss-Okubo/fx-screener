"""MTFそろいシグナルの売買検証 (ルール変種の比較)

エントリーは全変種共通: 15分/1時間/4時間足のトレンドが3つそろったバーの終値。
比較する変種:
- ① 元ルール: そろいが崩れたら即決済 (中立でも決済)
- ② 出口=15分逆転: 15分足がポジションと逆方向になるまで保有 (中立では保有継続)
- ③ 出口=1時間崩れ: 1時間足がポジション方向でなくなったら決済 (15分足は無視)
- ④ ② + 入口フレッシュ: 直前2時間(8バー)以上そろいが無かった場合のみ新規エントリー
- ⑤ ③ + 入口フレッシュ: 同上
- ⑥ 利確=BB3σ / 損切=SMA20−30pips: 15分足のボリンジャーバンド(20期間)+3σタッチで利確、
  中心線(SMA20)から30pips逆行で損切 (トレーリング)。高値/安値でタッチ判定、同一バーは損切優先

制約: yfinance の15分足は直近60日しか取得できないため、検証期間は約2ヶ月。
1時間足は365日分を取得して 1時間/4時間足のEMAを十分にウォームアップさせる。
進行中の1時間/4時間バーは、最新の15分終値でEMAを1ステップ仮更新して判定する
(ライブ判定と同等の扱い)。建玉中は各変種の決済条件のみを監視する。
"""
from __future__ import annotations

import argparse
import logging
import math
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
BB_NAME = "⑥ 利確BB3σ/損切SMA-30p"
BB_SPAN = 20      # ボリンジャーバンドの期間 (15分足)
BB_SIGMA = 3.0    # 利確バンドのシグマ
SL_PIPS = 30.0    # 損切: SMA20 からの逆行幅 (pips)


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
                    "reason": "ルール",
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
            "reason": "強制",
        })
    return trades


def _simulate_bb(df15: pd.DataFrame, sig: np.ndarray, pip: float,
                 cost: float) -> list[dict]:
    """⑥: 利確=BB+3σタッチ、損切=SMA20から30pips逆行 (トレーリング)。

    高値/安値でタッチ判定し、約定はレベル価格 (バーの寄付がレベルを飛び越えて
    いた場合は寄付価格)。同一バーで利確・損切の両方に届いた場合は損切を優先。
    """
    c = df15["Close"].to_numpy(float)
    o = df15["Open"].to_numpy(float)
    h = df15["High"].to_numpy(float)
    lo_ = df15["Low"].to_numpy(float)
    times = df15.index
    close_s = df15["Close"]
    sma = close_s.rolling(BB_SPAN).mean().to_numpy(float)
    sd = close_s.rolling(BB_SPAN).std(ddof=0).to_numpy(float)
    upper = sma + BB_SIGMA * sd
    lower = sma - BB_SIGMA * sd
    sl_off = SL_PIPS * pip

    trades: list[dict] = []
    pos, p_in, t_in, i_in = 0, 0.0, None, -1

    def _close_trade(i: int, price: float, reason: str) -> None:
        nonlocal pos
        trades.append({
            "entry": t_in, "exit": times[i], "side": pos,
            "ret": (price / p_in - 1) * pos - cost,
            "hours": (times[i] - t_in).total_seconds() / 3600,
            "reason": reason,
        })
        pos = 0

    for i in range(len(c)):
        if pos != 0 and i > i_in and not math.isnan(sma[i]):
            if pos == 1:
                stop, tp = sma[i] - sl_off, upper[i]
                if o[i] <= stop:
                    _close_trade(i, o[i], "損切")
                elif lo_[i] <= stop:
                    _close_trade(i, stop, "損切")
                elif o[i] >= tp:
                    _close_trade(i, o[i], "利確")
                elif h[i] >= tp:
                    _close_trade(i, tp, "利確")
            else:
                stop, tp = sma[i] + sl_off, lower[i]
                if o[i] >= stop:
                    _close_trade(i, o[i], "損切")
                elif h[i] >= stop:
                    _close_trade(i, stop, "損切")
                elif o[i] <= tp:
                    _close_trade(i, o[i], "利確")
                elif lo_[i] <= tp:
                    _close_trade(i, tp, "利確")
        if pos == 0:
            s = int(sig[i])
            if s != 0 and (i == 0 or int(sig[i - 1]) != s):
                pos, p_in, t_in, i_in = s, c[i], times[i], i
    if pos != 0:
        _close_trade(len(c) - 1, c[-1], "強制")
    return trades


def _ohlc(data: pd.DataFrame, ticker: str, single: bool) -> pd.DataFrame | None:
    try:
        df = data if single else data[ticker]
        df = df.dropna(subset=["Close"])
        return df.tz_convert("UTC") if df.index.tz is not None else df
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

    names = [name for name, _, _ in VARIANTS] + [BB_NAME]
    all_trades: dict[str, list[dict]] = {name: [] for name in names}
    period_start, period_end = None, None
    for _, u in uni.iterrows():
        df15 = _ohlc(d15, u["ticker"], single)
        df1h = _ohlc(d1h, u["ticker"], single)
        if df15 is None or df1h is None or len(df15) < 200:
            logger.warning("%s: データ不足のためスキップ", u["pair"])
            continue
        c15, c1h = df15["Close"], df1h["Close"]
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

        pip = 0.01 if u["quote"] == "JPY" else 0.0001
        trades = _simulate_bb(df15, sig, pip, cost)
        for t in trades:
            t["pair"] = u["pair"]
        all_trades[BB_NAME] += trades

    if not any(all_trades.values()):
        logger.error("トレードが1件も発生しませんでした")
        return

    # 変種比較表
    lines = [
        f"# MTFそろい売買検証 (ルール変種比較) {run_date}",
        "",
        "エントリーは全変種共通: 3つの足がそろったバーの終値 (④⑤は直前2時間以上の非そろいが条件)。",
        f"⑥は利確=15分足BB({BB_SPAN}期間)+{BB_SIGMA:.0f}σタッチ、"
        f"損切=SMA{BB_SPAN}から{SL_PIPS:.0f}pips逆行 (トレーリング、同一バーは損切優先)。",
        f"検証期間: {period_start:%Y-%m-%d} 〜 {period_end:%Y-%m-%d} "
        f"(約{(period_end - period_start).days}日間 ／ yfinanceの15分足の上限)",
        f"取引コスト: 往復 {args.cost_bp:.1f}bp ／ 1トレード=固定ロット、リターンは単純合算",
        "",
        "## 変種比較 (全10ペア合算)",
        "",
        "| 変種 | 回数 | 勝率 | 平均 | 合計(コスト後) | 合計(コスト前) | 平均保有 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        s = _summary(all_trades[name], cost)
        lines.append(
            f"| {name} | {s['trades']} | {s['win'] * 100:.1f}% | {s['avg_bp']:+.1f}bp "
            f"| {s['total'] * 100:+.2f}% | {s['total_gross'] * 100:+.2f}% "
            f"| {s['hours']:.1f}h |")

    # ⑥の決済内訳とペア別
    bdf = pd.DataFrame(all_trades[BB_NAME])
    reason_counts = bdf["reason"].value_counts()
    reason_ret = bdf.groupby("reason")["ret"].mean() * 10000
    lines += ["", f"## {BB_NAME} の決済内訳", "",
              "| 決済理由 | 回数 | 割合 | 平均リターン |",
              "|---|---:|---:|---:|"]
    for reason in ["利確", "損切", "強制"]:
        if reason in reason_counts:
            n = int(reason_counts[reason])
            lines.append(f"| {reason} | {n} | {n / len(bdf) * 100:.1f}% "
                         f"| {reason_ret[reason]:+.1f}bp |")

    lines += ["", f"## ペア別内訳 ({BB_NAME} / コスト後)", "",
              "| ペア | 回数 | 買/売 | 勝率 | 利確率 | 平均 | 合計 | 平均保有 |",
              "|---|---:|---|---:|---:|---:|---:|---:|"]
    for pair, g in bdf.groupby("pair"):
        rets = g["ret"].to_numpy()
        buys = int((g["side"] == 1).sum())
        tp_rate = (g["reason"] == "利確").mean()
        lines.append(
            f"| {pair} | {len(g)} | {buys}/{len(g) - buys} "
            f"| {(rets > 0).mean() * 100:.0f}% | {tp_rate * 100:.0f}% "
            f"| {rets.mean() * 10000:+.1f}bp "
            f"| {rets.sum() * 100:+.2f}% | {g['hours'].mean():.1f}h |")

    lines += [
        "",
        "## 注意事項",
        "",
        "- **検証期間は約2ヶ月のみ** (yfinanceの15分足は直近60日が上限)。この期間の相場環境に強く依存し、統計的な確度は低い",
        "- ①〜⑤は終値ベースの約定。⑥のTP/SLは高値/安値タッチで判定しレベル価格で約定 (窓開けは寄付価格)。スリッページ未考慮",
        "- スワップ損益は未考慮",
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
