"""Markdownレポート生成"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def _fmt(v, pct: bool = False, digits: int = 1, signed: bool = False) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "-"
    sign = "+" if signed else ""
    return f"{v * 100:{sign}.{digits}f}%" if pct else f"{v:{sign}.{digits}f}"


def _table(df: pd.DataFrame, new_tickers: set[str]) -> str:
    lines = [
        "| 順位 | ペア | 名称 | 区分 | スコア | 金利差 | 5年乖離 | 3ヶ月 | 年率ボラ | RSI | |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, r in df.iterrows():
        mark = "🆕" if r["ticker"] in new_tickers else ""
        lines.append(
            f"| {r['rank']} | {r['pair']} | {r['name']} | {r['market']} "
            f"| {_fmt(r['score'])} | {_fmt(r['carry'], signed=True, digits=2)}% "
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
                 new_tickers: set[str], rates_as_of: str) -> str:
    top_n = cfg["top_n"]

    parts = [
        f"# FXスクリーニング結果 {run_date}",
        "",
        f"対象: {len(ranked)}ペア。スコアは「基軸通貨の買い(ロング)」の魅力度。"
        "低スコアは売り方向の妙味を示す場合があります。",
        "",
        "## 総合ランキング (全ペア)",
        "",
        _table(ranked, new_tickers),
        "",
        f"## スコア内訳 (トップ{top_n} / 全ペア内偏差)",
        "",
        _breakdown_table(ranked.head(top_n)),
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
