"""Markdownレポート生成 (買い/売り 両方向ランキング)"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def _fmt(v, pct: bool = False, digits: int = 1, signed: bool = False) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "-"
    sign = "+" if signed else ""
    return f"{v * 100:{sign}.{digits}f}%" if pct else f"{v:{sign}.{digits}f}"


def _swap_mark(carry_dir: float) -> str:
    return "＋受取" if carry_dir > 0 else ("−支払" if carry_dir < 0 else "±0")


def _table(df: pd.DataFrame, new_keys: set[tuple[str, str]]) -> str:
    lines = [
        "| 順位 | ペア | 名称 | 区分 | スコア | 金利差 | スワップ | 5年乖離 | 3ヶ月 | 年率ボラ | RSI | |",
        "|---:|---|---|---|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for _, r in df.iterrows():
        mark = "🆕" if (r["ticker"], r["direction"]) in new_keys else ""
        lines.append(
            f"| {r['rank']} | {r['pair']} | {r['name']} | {r['market']} "
            f"| {_fmt(r['score'])} | {_fmt(r['carry_dir'], signed=True, digits=2)}% "
            f"| {_swap_mark(r['carry_dir'])} "
            f"| {_fmt(r['value_dev'], pct=True, signed=True)} "
            f"| {_fmt(r['ret_63d'], pct=True, signed=True)} "
            f"| {_fmt(r['vol_60d'], pct=True)} "
            f"| {_fmt(r['rsi'], digits=0)} | {mark} |"
        )
    return "\n".join(lines)


def _breakdown_table(df: pd.DataFrame) -> str:
    lines = [
        "| ペア | 金利差 | 割安 | トレンド | 勢い | 安定 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        cells = " | ".join(_fmt(r[f"z_{c}"], digits=2, signed=True)
                           for c in ["carry", "value", "trend", "momentum", "stability"])
        lines.append(f"| {r['pair']} | {cells} |")
    return "\n".join(lines)


def build_report(ranked: pd.DataFrame, run_date: str, cfg: dict,
                 new_keys: set[tuple[str, str]], rates_as_of: str) -> str:
    top_n = cfg["top_n"]
    buy = ranked[ranked["direction"] == "買い"].sort_values("rank")
    sell = ranked[ranked["direction"] == "売り"].sort_values("rank")

    parts = [
        f"# FXスクリーニング結果 {run_date}",
        "",
        f"対象: {len(buy)}ペア × 買い/売り両方向。"
        "金利差はその方向で建てた場合の政策金利差 (プラス = スワップ受取の概算、マイナス = 支払)。",
        "",
        "## 買い(ロング)ランキング",
        "",
        _table(buy, new_keys),
        "",
        "## 売り(ショート)ランキング",
        "",
        _table(sell, new_keys),
        "",
        f"## スコア内訳 (買いトップ{top_n} / 偏差)",
        "",
        _breakdown_table(buy.head(top_n)),
        "",
        f"## スコア内訳 (売りトップ{top_n} / 偏差)",
        "",
        _breakdown_table(sell.head(top_n)),
        "",
        "---",
        f"※ 金利差は各国政策金利({rates_as_of}時点の設定値)の差であり、実際のスワップポイントとは異なります。",
        "※ 本レポートは機械的なスクリーニング結果であり、投資助言ではありません。"
        "FXはレバレッジにより損失が拡大するリスクがあります。投資判断はご自身の責任で行ってください。",
    ]
    return "\n".join(parts)


def save_report(text: str, run_date: str, output_dir: str | Path) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{run_date}.md"
    path.write_text(text, encoding="utf-8")
    return path
