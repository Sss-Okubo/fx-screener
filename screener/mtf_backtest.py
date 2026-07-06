"""MTFそろいシグナルの売買検証 (ルール変種の比較)

エントリーは①〜⑧共通: 15分/1時間/4時間足のトレンドが3つそろったバーの終値。
- ① 元ルール: そろいが崩れたら即決済 (中立でも決済)
- ② 出口=15分逆転: 15分足がポジションと逆方向になるまで保有
- ③ 出口=1時間崩れ: 1時間足がポジション方向でなくなったら決済
- ④ ② + 入口フレッシュ(2h) / ⑤ ③ + 入口フレッシュ(2h)
- ⑥ 利確=BB3σ / 損切=SMA20−30pips (トレーリング)
- ⑦ 利確=BB3σ / 損切=建値−1.5×ATR14 (建値固定)
- ⑧ 半分をBB3σで利確 → 残りはSMA20割れまでトレイル / 損切=建値−1.5×ATR14
- ⑨ 1時間足ベース: 1時間/4時間/日足のそろいでエントリー、利確=1時間足BB3σ、
  損切=建値−1.5×ATR14(1時間足)。検証期間は約1年 (他と期間が異なる点に注意)

制約: yfinance の15分足は直近60日しか取得できないため①〜⑧の検証期間は約2ヶ月。
進行中の上位足バーは、最新の下位足終値でEMAを1ステップ仮更新して判定する。
TP/SLは高値/安値タッチで判定しレベル価格で約定 (窓開けは寄付)。同一バーは損切優先。
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
NAME_BB = "⑥ 利確BB3σ/損切SMA-30p"
NAME_ATR = "⑦ 利確BB3σ/損切1.5ATR固定"
NAME_HALF = "⑧ 半分利確+中心線トレイル"
NAME_H1 = "⑨ TP=BB3σ/SL=1.5ATR"
NAME_H1_ALIGN = "⑩ そろい崩れで即決済"
NAME_H1_REV = "⑪ 1時間足逆転まで保有"
NAME_H1_BE05 = "⑫ ⑨+建値SL(+0.5ATRで移動)"
NAME_H1_BE10 = "⑬ ⑨+建値SL(+1.0ATRで移動)"
NAME_H1_BE15 = "⑭ ⑨+建値SL(+1.5ATRで移動)"
BB_SPAN = 20      # ボリンジャーバンドの期間
BB_SIGMA = 3.0    # 利確バンドのシグマ
SL_PIPS = 30.0    # ⑥の損切: SMA20 からの逆行幅 (pips)
ATR_MULT = 1.5    # ⑦⑧⑨の損切: 建値からの ATR 倍率


def _dirs_self(close: pd.Series) -> np.ndarray:
    """その足自身のEMA20/50によるトレンド方向"""
    e20 = close.ewm(span=20, adjust=False).mean().to_numpy(float)
    e50 = close.ewm(span=50, adjust=False).mean().to_numpy(float)
    p = close.to_numpy(float)
    return np.where((p > e20) & (e20 > e50), 1,
                    np.where((p < e20) & (e20 < e50), -1, 0))


def _dirs_higher(base: pd.Series, upper_close: pd.Series,
                 bars: pd.Index) -> np.ndarray:
    """上位足の方向。完了バーのEMAに、進行中バーは下位足終値で1ステップ仮更新"""
    p = base.to_numpy(float)
    emas = []
    for span in (20, 50):
        ema = upper_close.ewm(span=span, adjust=False).mean()
        prev = ema.shift(1).reindex(bars, method="ffill").to_numpy(float)
        a = 2 / (span + 1)
        emas.append(a * p + (1 - a) * prev)
    e20, e50 = emas
    return np.where((p > e20) & (e20 > e50), 1,
                    np.where((p < e20) & (e20 < e50), -1, 0))


def _atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    h, l, c = df["High"], df["Low"], df["Close"]
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean().to_numpy(float)


def _simulate(c15: pd.Series, sig: np.ndarray, d15: np.ndarray, d1h: np.ndarray,
              cost: float, exit_mode: str, fresh_bars: int) -> list[dict]:
    """①〜⑤: 終値ベースの決済"""
    prices = c15.to_numpy(float)
    times = c15.index
    trades: list[dict] = []
    pos, p_in, t_in = 0, 0.0, None
    last_nz = -10 ** 9

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
            run_start = i == 0 or int(sig[i - 1]) != s
            if run_start and (i - last_nz - 1) >= fresh_bars:
                pos, p_in, t_in = s, prices[i], times[i]
        if s != 0:
            last_nz = i
    if pos != 0:
        trades.append({
            "entry": t_in, "exit": times[-1], "side": pos,
            "ret": (prices[-1] / p_in - 1) * pos - cost,
            "hours": (times[-1] - t_in).total_seconds() / 3600,
            "reason": "強制",
        })
    return trades


def _simulate_tp_sl(df: pd.DataFrame, sig: np.ndarray, pip: float, cost: float,
                    sl_mode: str, breakeven_atr: float | None = None) -> list[dict]:
    """⑥⑦⑨⑫〜⑭: 利確=BB+3σタッチ、損切=sl_mode ("sma30"=SMA20-30pipsトレーリング /
    "atr"=建値−1.5×ATR固定)。breakeven_atr を指定すると、建値から
    その倍率×ATR だけ有利に進んだ次のバー以降、損切りを建値へ引き上げる"""
    c = df["Close"].to_numpy(float)
    o = df["Open"].to_numpy(float)
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    times = df.index
    sma = df["Close"].rolling(BB_SPAN).mean().to_numpy(float)
    sd = df["Close"].rolling(BB_SPAN).std(ddof=0).to_numpy(float)
    upper, lower = sma + BB_SIGMA * sd, sma - BB_SIGMA * sd
    atr = _atr(df)

    trades: list[dict] = []
    pos, p_in, t_in, i_in, stop0, atr_e = 0, 0.0, None, -1, 0.0, 0.0

    def close_trade(i, price, reason):
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
            if sl_mode == "sma30":
                stop = sma[i] - pos * SL_PIPS * pip
            else:
                stop = stop0
            sl_reason = "建値" if sl_mode == "atr" and stop0 == p_in else "損切"
            tp = upper[i] if pos == 1 else lower[i]
            if pos == 1:
                if o[i] <= stop:
                    close_trade(i, o[i], sl_reason)
                elif l[i] <= stop:
                    close_trade(i, stop, sl_reason)
                elif o[i] >= tp:
                    close_trade(i, o[i], "利確")
                elif h[i] >= tp:
                    close_trade(i, tp, "利確")
            else:
                if o[i] >= stop:
                    close_trade(i, o[i], sl_reason)
                elif h[i] >= stop:
                    close_trade(i, stop, sl_reason)
                elif o[i] <= tp:
                    close_trade(i, o[i], "利確")
                elif l[i] <= tp:
                    close_trade(i, tp, "利確")
            # 建値ストップ: トリガー到達を確認 (発動は次バーの判定から)
            if pos != 0 and breakeven_atr is not None and stop0 != p_in:
                if pos == 1 and h[i] >= p_in + breakeven_atr * atr_e:
                    stop0 = p_in
                elif pos == -1 and l[i] <= p_in - breakeven_atr * atr_e:
                    stop0 = p_in
        if pos == 0:
            s = int(sig[i])
            if s != 0 and (i == 0 or int(sig[i - 1]) != s) and not math.isnan(atr[i]):
                pos, p_in, t_in, i_in = s, c[i], times[i], i
                atr_e = atr[i]
                stop0 = p_in - s * ATR_MULT * atr_e
    if pos != 0:
        close_trade(len(c) - 1, c[-1], "強制")
    return trades


def _simulate_half_trail(df: pd.DataFrame, sig: np.ndarray, pip: float,
                         cost: float) -> list[dict]:
    """⑧: 損切=建値−1.5×ATR固定。BB+3σタッチで半分利確、残りはSMA20割れで決済"""
    c = df["Close"].to_numpy(float)
    o = df["Open"].to_numpy(float)
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    times = df.index
    sma = df["Close"].rolling(BB_SPAN).mean().to_numpy(float)
    sd = df["Close"].rolling(BB_SPAN).std(ddof=0).to_numpy(float)
    upper, lower = sma + BB_SIGMA * sd, sma - BB_SIGMA * sd
    atr = _atr(df)

    trades: list[dict] = []
    pos, p_in, t_in, i_in, stop0 = 0, 0.0, None, -1, 0.0
    half_ret = None  # 半分利確済みならそのリターン

    def close_trade(i, price, reason):
        nonlocal pos, half_ret
        r2 = (price / p_in - 1) * pos
        ret = r2 - cost if half_ret is None else 0.5 * half_ret + 0.5 * r2 - cost
        trades.append({
            "entry": t_in, "exit": times[i], "side": pos, "ret": ret,
            "hours": (times[i] - t_in).total_seconds() / 3600,
            "reason": reason,
        })
        pos, half_ret = 0, None

    for i in range(len(c)):
        if pos != 0 and i > i_in and not math.isnan(sma[i]):
            tp = upper[i] if pos == 1 else lower[i]
            if pos == 1:
                hit_sl = o[i] <= stop0 or l[i] <= stop0
                sl_px = min(o[i], stop0) if o[i] <= stop0 else stop0
                hit_tp = o[i] >= tp or h[i] >= tp
                tp_px = max(o[i], tp) if o[i] >= tp else tp
                hit_trail = o[i] <= sma[i] or l[i] <= sma[i]
                trail_px = min(o[i], sma[i]) if o[i] <= sma[i] else sma[i]
            else:
                hit_sl = o[i] >= stop0 or h[i] >= stop0
                sl_px = max(o[i], stop0) if o[i] >= stop0 else stop0
                hit_tp = o[i] <= tp or l[i] <= tp
                tp_px = min(o[i], tp) if o[i] <= tp else tp
                hit_trail = o[i] >= sma[i] or h[i] >= sma[i]
                trail_px = max(o[i], sma[i]) if o[i] >= sma[i] else sma[i]

            if half_ret is None:
                if hit_sl:
                    close_trade(i, sl_px, "損切")
                elif hit_tp:
                    half_ret = (tp_px / p_in - 1) * pos  # 半分利確、残り継続
            else:
                if hit_sl:
                    close_trade(i, sl_px, "半利確→損切")
                elif hit_trail:
                    close_trade(i, trail_px, "半利確→トレイル")
        if pos == 0:
            s = int(sig[i])
            if s != 0 and (i == 0 or int(sig[i - 1]) != s) and not math.isnan(atr[i]):
                pos, p_in, t_in, i_in = s, c[i], times[i], i
                stop0 = p_in - s * ATR_MULT * atr[i]
    if pos != 0:
        close_trade(len(c) - 1, c[-1], "強制")
    return trades


def _ohlc(data: pd.DataFrame, ticker: str, single: bool) -> pd.DataFrame | None:
    try:
        df = data if single else data[ticker]
        df = df.dropna(subset=["Close"])
        # 日足はtzなしで返るためUTCとして扱い、時間足と比較可能にする
        return (df.tz_localize("UTC") if df.index.tz is None
                else df.tz_convert("UTC"))
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
    logger.info("1時間足を取得中 (730日分)")
    d1h = yf.download(tickers, period="730d", interval="1h", group_by="ticker",
                      auto_adjust=True, threads=True, progress=False)
    logger.info("日足を取得中 (3年分)")
    d1d = yf.download(tickers, period="3y", interval="1d", group_by="ticker",
                      auto_adjust=True, threads=True, progress=False)

    names_15m = [name for name, _, _ in VARIANTS] + [NAME_BB, NAME_ATR, NAME_HALF]
    names_1h = [NAME_H1, NAME_H1_ALIGN, NAME_H1_REV,
                NAME_H1_BE05, NAME_H1_BE10, NAME_H1_BE15]
    names = names_15m + names_1h
    all_trades: dict[str, list[dict]] = {name: [] for name in names}
    period_start, period_end = None, None
    h1_start, h1_end = None, None
    for _, u in uni.iterrows():
        df15 = _ohlc(d15, u["ticker"], single)
        df1h = _ohlc(d1h, u["ticker"], single)
        df1d = _ohlc(d1d, u["ticker"], single)
        if df15 is None or df1h is None or len(df15) < 200:
            logger.warning("%s: データ不足のためスキップ", u["pair"])
            continue
        pip = 0.01 if u["quote"] == "JPY" else 0.0001
        c15, c1h = df15["Close"], df1h["Close"]
        period_start = min(period_start or c15.index[0], c15.index[0])
        period_end = max(period_end or c15.index[-1], c15.index[-1])

        # ---- ①〜⑧: 15分足ベース ----
        dir15 = _dirs_self(c15)
        dir1h = _dirs_higher(c15, c1h, c15.index.floor("1h"))
        c4h = c1h.resample("4h").last().dropna()
        dir4h = _dirs_higher(c15, c4h, c15.index.floor("4h"))
        sig = np.where((dir15 == 1) & (dir1h == 1) & (dir4h == 1), 1,
                       np.where((dir15 == -1) & (dir1h == -1) & (dir4h == -1), -1, 0))

        for name, exit_mode, fresh in VARIANTS:
            trades = _simulate(c15, sig, dir15, dir1h, cost, exit_mode, fresh)
            for t in trades:
                t["pair"] = u["pair"]
            all_trades[name] += trades
        for name, sl_mode in [(NAME_BB, "sma30"), (NAME_ATR, "atr")]:
            trades = _simulate_tp_sl(df15, sig, pip, cost, sl_mode)
            for t in trades:
                t["pair"] = u["pair"]
            all_trades[name] += trades
        trades = _simulate_half_trail(df15, sig, pip, cost)
        for t in trades:
            t["pair"] = u["pair"]
        all_trades[NAME_HALF] += trades

        # ---- ⑨⑩⑪: 1時間足ベース (1h/4h/日足のそろい、上位足版の通知と同じシグナル) ----
        if df1d is not None and len(df1d) > 60:
            h1_start = min(h1_start or c1h.index[0], c1h.index[0])
            h1_end = max(h1_end or c1h.index[-1], c1h.index[-1])
            dh1 = _dirs_self(c1h)
            dh4 = _dirs_higher(c1h, c4h, c1h.index.floor("4h"))
            c1d = df1d["Close"]
            dhd = _dirs_higher(c1h, c1d, c1h.index.normalize())
            sig1h = np.where((dh1 == 1) & (dh4 == 1) & (dhd == 1), 1,
                             np.where((dh1 == -1) & (dh4 == -1) & (dhd == -1), -1, 0))
            for name, trades in [
                (NAME_H1, _simulate_tp_sl(df1h, sig1h, pip, cost, "atr")),
                (NAME_H1_ALIGN, _simulate(c1h, sig1h, dh1, dh4, cost, "align", 0)),
                (NAME_H1_REV, _simulate(c1h, sig1h, dh1, dh4, cost, "m15", 0)),
                (NAME_H1_BE05, _simulate_tp_sl(df1h, sig1h, pip, cost, "atr", 0.5)),
                (NAME_H1_BE10, _simulate_tp_sl(df1h, sig1h, pip, cost, "atr", 1.0)),
                (NAME_H1_BE15, _simulate_tp_sl(df1h, sig1h, pip, cost, "atr", 1.5)),
            ]:
                for t in trades:
                    t["pair"] = u["pair"]
                all_trades[name] += trades

    if not any(all_trades.values()):
        logger.error("トレードが1件も発生しませんでした")
        return

    def _rows(name_list):
        rows = ["| 変種 | 回数 | 勝率 | 平均 | 合計(コスト後) | 合計(コスト前) | 平均保有 |",
                "|---|---:|---:|---:|---:|---:|---:|"]
        for name in name_list:
            s = _summary(all_trades[name], cost)
            rows.append(
                f"| {name} | {s['trades']} | {s['win'] * 100:.1f}% | {s['avg_bp']:+.1f}bp "
                f"| {s['total'] * 100:+.2f}% | {s['total_gross'] * 100:+.2f}% "
                f"| {s['hours']:.1f}h |")
        return rows

    lines = [
        f"# MTFそろい売買検証 (ルール変種比較) {run_date}",
        "",
        "①〜⑧のエントリー: 15分/1時間/4時間足がそろったバーの終値。",
        "⑨〜⑪のエントリー: 1時間/4時間/日足がそろった1時間足バーの終値 (上位足版の通知と同じシグナル)。",
        f"利確=BB({BB_SPAN}期間)+{BB_SIGMA:.0f}σタッチ ／ 損切=⑥はSMA{BB_SPAN}−{SL_PIPS:.0f}pips(トレーリング)、"
        f"⑦⑧⑨は建値−{ATR_MULT}×ATR14(固定) ／ ⑧は半分利確後SMA{BB_SPAN}割れまでトレイル。",
        f"検証期間: ①〜⑧ {period_start:%Y-%m-%d}〜{period_end:%m-%d} (15分足の上限) ／ "
        f"⑨〜⑪ {h1_start:%Y-%m-%d}〜{h1_end:%m-%d} (1時間足の上限)",
        f"取引コスト: 往復 {args.cost_bp:.1f}bp ／ 1トレード=固定ロット、リターンは単純合算",
        "",
        "## 15分足ベース (①〜⑧、約2ヶ月)",
        "",
        *_rows(names_15m),
        "",
        "## 1時間足ベース (⑨〜⑪、約2年)",
        "",
        *_rows(names_1h),
    ]

    # 1時間足ベース変種の決済内訳
    for name in names_1h:
        tdf = pd.DataFrame(all_trades[name])
        if tdf.empty:
            continue
        lines += ["", f"## {name} の決済内訳", "",
                  "| 決済理由 | 回数 | 割合 | 平均リターン |", "|---|---:|---:|---:|"]
        for reason, g in tdf.groupby("reason"):
            lines.append(f"| {reason} | {len(g)} | {len(g) / len(tdf) * 100:.1f}% "
                         f"| {g['ret'].mean() * 10000:+.1f}bp |")

    # 最良の1時間足ベース変種のペア別内訳
    best_name = max(names_1h, key=lambda n: _summary(all_trades[n], cost)["total"])
    bdf = pd.DataFrame(all_trades[best_name])
    lines += ["", f"## ペア別内訳 (1時間足ベースで最良: {best_name} / コスト後)", "",
              "| ペア | 回数 | 買/売 | 勝率 | 平均 | 合計 | 平均保有 |",
              "|---|---:|---|---:|---:|---:|---:|"]
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
        "- ①〜⑧の検証期間は約2ヶ月のみ (yfinanceの15分足は直近60日が上限)。⑨は約1年だが期間が異なるため単純比較不可",
        "- ①〜⑤は終値ベースの約定。⑥〜⑨のTP/SLは高値/安値タッチで判定しレベル価格で約定 (窓開けは寄付)。同一バーは損切優先。スリッページ未考慮",
        "- ⑧の往復コストは全量に1回分で近似 (分割決済の追加コストは未考慮)",
        "- スワップ損益は未考慮",
        "- 上位足の進行中バーは下位足終値でEMAを仮更新して判定 (ライブ判定に近い扱いだが完全一致ではない)",
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
