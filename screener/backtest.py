"""選定ルールの過去検証 (月末リバランス・トップN等ウェイト・買い/売り両方向)

2変種で検証する:
- テクニカル+割安 (金利差抜き): 全指標が各時点の価格履歴から計算でき、先読みバイアスなし
- 複合 (金利差は現在値固定): 政策金利の履歴が無いため現在値で代用 = 楽観バイアスあり (参考値)

リターンは価格変動のみ (複合変種はキャリーの月割りを加算)。スプレッド・実スワップ・
レバレッジコストは未考慮。
"""
from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import fetch, scoring, universe
from .indicators import compute_technical
from .store import Store

logger = logging.getLogger(__name__)

BT_PERIOD = "12y"     # 5年検証 + 5年平均乖離の計算余地
BT_CACHE_DAYS = 7     # 長期データのキャッシュ有効日数


def _metrics(monthly: pd.Series) -> dict:
    m = monthly.dropna()
    cum = float((1 + m).prod() - 1)
    ann = float((1 + cum) ** (12 / len(m)) - 1) if len(m) else float("nan")
    vol = float(m.std(ddof=0) * np.sqrt(12))
    curve = (1 + m).cumprod()
    max_dd = float((curve / curve.cummax() - 1).min())
    return {
        "累積": cum, "年率": ann, "年率ボラ": vol,
        "シャープ": ann / vol if vol > 0 else float("nan"),
        "最大DD": max_dd,
    }


def run(args: argparse.Namespace) -> None:
    base = Path(args.config).resolve().parent
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run_date = date.today().isoformat()

    store = Store(base / "data" / "backtest.db")
    uni = universe.get_universe(args.limit)
    tickers = uni["ticker"].tolist()
    fetch.fetch_prices(store, tickers, BT_PERIOD,
                       0 if args.force_refresh else BT_CACHE_DAYS)
    prices = {t: store.load_prices(t) for t in tickers}
    store.close()

    ref = prices.get("USDJPY=X")
    if ref is None or ref.empty:
        logger.error("基準ペア (USDJPY=X) の価格データがありません")
        return

    # 月末営業日の列 (検証月数 + 1)
    idx = ref.index
    month_ends = idx.to_series().groupby(idx.to_period("M")).max()
    month_ends = month_ends.iloc[-(args.years * 12 + 1):]

    rates: dict[str, float] = cfg["rates"]["policy"]
    w: dict[str, float] = cfg["scoring"]["weights"]
    # 金利差抜き変種: carry の重みを残りに比例配分
    w_nc = {k: (0.0 if k == "carry" else v / (1 - w["carry"])) for k, v in w.items()}
    variants: dict[str, tuple[dict, bool]] = {
        "テクニカル+割安": (w_nc, False),
        "複合(金利差固定)": (w, True),
    }

    recs = []
    for i in range(len(month_ends) - 1):
        t0, t1 = month_ends.iloc[i], month_ends.iloc[i + 1]
        rows = []
        for _, u in uni.iterrows():
            df = prices[u["ticker"]]
            if df.empty:
                continue
            hist = df[df.index <= t0]
            tech = compute_technical(hist)
            if tech is None:
                continue
            p1 = df["close"].asof(t1)
            if pd.isna(p1):
                continue
            rows.append({
                **u.to_dict(), **tech,
                "carry": rates.get(u["base"], 0.0) - rates.get(u["quote"], 0.0),
                "fwd": float(p1 / tech["price"] - 1),
            })
        feats = pd.DataFrame(rows)
        if len(feats) < 5:
            continue

        rec = {"date": t1.date().isoformat(),
               "全ペア買い平均": float(feats["fwd"].mean())}
        u_fwd = feats.loc[feats["ticker"] == "USDJPY=X", "fwd"]
        rec["USD/JPY買い持ち"] = float(u_fwd.iloc[0]) if len(u_fwd) else np.nan

        for name, (wv, with_carry) in variants.items():
            scfg = {**cfg["scoring"], "weights": wv}
            ranked = scoring.score(feats.copy(), scfg)
            top = ranked.head(args.top)  # スコア順の上位 (買い/売り混在)
            ret = float((top["fwd"] * top["dir"]).mean())
            if with_carry:
                ret += float((top["carry_dir"] / 100 / 12).mean())
            rec[name] = ret
        recs.append(rec)

    res = pd.DataFrame(recs).set_index("date")
    logger.info("検証期間: %s 〜 %s (%dヶ月)", res.index[0], res.index[-1], len(res))

    # レポート生成
    out_dir = base / cfg["report"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"backtest-monthly-{run_date}.csv"
    res.to_csv(csv_path, encoding="utf-8")

    cols = list(variants.keys()) + ["全ペア買い平均", "USD/JPY買い持ち"]
    lines = [
        f"# FXバックテスト結果 {run_date}",
        "",
        f"設定: 過去{args.years}年 ／ 月末リバランス ／ スコア上位{args.top}ポジション等ウェイト"
        " (買い/売り混在) ／ スプレッド・実スワップ未考慮",
        f"検証期間: {res.index[0]} 〜 {res.index[-1]} ({len(res)}ヶ月)",
        "",
        "| 戦略 | 累積 | 年率 | 年率ボラ | シャープ | 最大DD |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for c in cols:
        m = _metrics(res[c])
        lines.append(f"| {c} | {m['累積'] * 100:+.1f}% | {m['年率'] * 100:+.1f}% "
                     f"| {m['年率ボラ'] * 100:.1f}% | {m['シャープ']:.2f} "
                     f"| {m['最大DD'] * 100:.1f}% |")
    lines += [
        "",
        "## 注意事項",
        "",
        "- **テクニカル+割安**: 全指標が各時点で計算可能なもののみ。先読みバイアスなし (信頼できる変種)",
        "- **複合(金利差固定)**: 政策金利の履歴が無いため**現在の金利差を全期間に適用 = 楽観バイアスあり**。参考値",
        "- リターンは価格変動のみ (複合変種はキャリー月割りを加算)。**スプレッド・実際のスワップポイント・取引コストは未考慮**",
        "- 対象10ペアの小さなユニバースでの検証であり、統計的な頑健性は限定的",
        "- 過去の成績は将来の成果を保証しない。投資判断は自己責任で",
    ]
    md_path = out_dir / f"backtest-{run_date}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("バックテストレポートを保存: %s", md_path)
    print(md_path)
    print(res.tail(3).to_string())
