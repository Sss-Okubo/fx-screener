"""HTMLレポート生成 (reports/index.html — GitHub Pages やブラウザ閲覧用)"""
from __future__ import annotations

import html
from pathlib import Path

import pandas as pd

CATEGORY_LABELS = [
    ("carry", "金利差"), ("value", "割安"), ("trend", "トレンド"),
    ("momentum", "勢い"), ("stability", "安定"),
]

WEIGHT_LABELS = {
    "carry": "金利差", "value": "割安", "trend": "トレンド",
    "momentum": "勢い", "stability": "安定",
}

STYLE = """
:root {
  --page: #f9f9f7; --surface: #fcfcfb;
  --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --ring: rgba(11,11,11,0.10);
  --bar: #2a78d6; --bar-track: #f0efec;
  --pos: #2a78d6; --neg: #e34948; --mid: #e1e0d9;
}
@media (prefers-color-scheme: dark) {
  :root {
    --page: #0d0d0d; --surface: #1a1a19;
    --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --ring: rgba(255,255,255,0.10);
    --bar: #3987e5; --bar-track: #262624;
    --pos: #3987e5; --neg: #e66767; --mid: #383835;
  }
}
:root[data-theme="light"] {
  --page: #f9f9f7; --surface: #fcfcfb;
  --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --ring: rgba(11,11,11,0.10);
  --bar: #2a78d6; --bar-track: #f0efec;
  --pos: #2a78d6; --neg: #e34948; --mid: #e1e0d9;
}
:root[data-theme="dark"] {
  --page: #0d0d0d; --surface: #1a1a19;
  --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --ring: rgba(255,255,255,0.10);
  --bar: #3987e5; --bar-track: #262624;
  --pos: #3987e5; --neg: #e66767; --mid: #383835;
}
.scr {
  font-family: system-ui, -apple-system, "Segoe UI", "Yu Gothic UI",
               "Hiragino Kaku Gothic ProN", "Meiryo", sans-serif;
  background: var(--page); color: var(--ink);
  margin: 0; padding: 2.5rem 1.25rem 3rem;
  line-height: 1.6;
}
.scr .wrap { max-width: 62rem; margin: 0 auto; display: flex; flex-direction: column; gap: 2rem; }
.scr header h1 { font-size: 1.45rem; font-weight: 700; margin: 0; letter-spacing: .01em; text-wrap: balance; }
.scr header .date { color: var(--ink-2); font-size: .95rem; margin-top: .2rem; }
.scr .stats { display: flex; flex-wrap: wrap; gap: .75rem; margin-top: 1rem; }
.scr .stat {
  background: var(--surface); border: 1px solid var(--ring); border-radius: 6px;
  padding: .5rem .9rem; min-width: 6.5rem;
}
.scr .stat .k { font-size: .72rem; color: var(--muted); text-transform: uppercase; letter-spacing: .08em; }
.scr .stat .v { font-size: 1.25rem; font-weight: 650; }
.scr .weights { display: flex; flex-wrap: wrap; gap: .4rem; margin-top: .75rem; }
.scr .weights .chip {
  font-size: .78rem; color: var(--ink-2); border: 1px solid var(--grid);
  border-radius: 999px; padding: .1rem .6rem; background: var(--surface);
}
.scr section h2 { font-size: 1.05rem; font-weight: 650; margin: 0 0 .6rem; }
.scr section .note { font-size: .8rem; color: var(--muted); margin: -.3rem 0 .6rem; }
.scr .tblwrap { overflow-x: auto; background: var(--surface); border: 1px solid var(--ring); border-radius: 8px; }
.scr table { border-collapse: collapse; width: 100%; font-size: .88rem; font-variant-numeric: tabular-nums; }
.scr th {
  text-align: right; font-size: .72rem; color: var(--muted); font-weight: 600;
  letter-spacing: .06em; padding: .55rem .7rem; border-bottom: 1px solid var(--grid);
  white-space: nowrap;
}
.scr th.l, .scr td.l { text-align: left; }
.scr td { padding: .45rem .7rem; border-bottom: 1px solid var(--grid); text-align: right; white-space: nowrap; }
.scr tr:last-child td { border-bottom: none; }
.scr td.name { max-width: 14rem; overflow: hidden; text-overflow: ellipsis; }
.scr .mkt {
  display: inline-block; font-size: .72rem; font-weight: 600; letter-spacing: .05em;
  border: 1px solid var(--grid); border-radius: 4px; padding: 0 .35rem; color: var(--ink-2);
}
.scr .new { color: var(--pos); font-size: .72rem; font-weight: 700; margin-left: .3rem; }
.scr .scorecell { display: flex; align-items: center; gap: .55rem; justify-content: flex-end; }
.scr .scorebar { width: 90px; height: 8px; background: var(--bar-track); border-radius: 4px; overflow: hidden; }
.scr .scorebar i { display: block; height: 100%; background: var(--bar); border-radius: 4px; }
.scr .zcell { display: flex; align-items: center; gap: .45rem; justify-content: flex-end; }
.scr .zbar { position: relative; width: 72px; height: 8px; background: var(--bar-track); border-radius: 4px; }
.scr .zbar::after {
  content: ""; position: absolute; left: 50%; top: -2px; bottom: -2px;
  width: 1px; background: var(--mid);
}
.scr .zbar i { position: absolute; top: 0; height: 100%; border-radius: 4px; }
.scr .zbar i.p { left: 50%; background: var(--pos); }
.scr .zbar i.n { right: 50%; background: var(--neg); }
.scr .zval { min-width: 3.2rem; color: var(--ink-2); font-size: .82rem; }
.scr footer { color: var(--muted); font-size: .8rem; border-top: 1px solid var(--grid); padding-top: 1rem; }
"""


def _f(v, pct=False, digits=1, signed=False):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "-"
    sign = "+" if signed else ""
    if pct:
        return f"{v * 100:{sign}.{digits}f}%"
    return f"{v:{sign}.{digits}f}"


def _score_cell(score: float) -> str:
    width = max(0.0, min(100.0, score))
    return (f'<div class="scorecell"><span class="scorebar">'
            f'<i style="width:{width:.0f}%"></i></span>'
            f'<strong>{score:.1f}</strong></div>')


def _z_cell(z: float) -> str:
    if z is None or pd.isna(z):
        return '<div class="zcell"><span class="zval">-</span></div>'
    w = min(abs(z), 3.0) / 3.0 * 50
    cls = "p" if z >= 0 else "n"
    return (f'<div class="zcell"><span class="zbar">'
            f'<i class="{cls}" style="width:{w:.0f}%"></i></span>'
            f'<span class="zval">{z:+.2f}</span></div>')


def _rank_rows(df: pd.DataFrame, new_tickers: set[str]) -> str:
    rows = []
    for _, r in df.iterrows():
        new = '<span class="new">NEW</span>' if r["ticker"] in new_tickers else ""
        rows.append(
            f"<tr><td>{r['rank']}</td>"
            f'<td class="l"><strong>{html.escape(r["pair"])}</strong>{new}</td>'
            f'<td class="l name">{html.escape(str(r["name"]))}</td>'
            f'<td class="l"><span class="mkt">{r["market"]}</span></td>'
            f"<td>{_score_cell(r['score'])}</td>"
            f"<td>{_f(r['carry'], digits=2, signed=True)}%</td>"
            f"<td>{_f(r['value_dev'], pct=True, signed=True)}</td>"
            f"<td>{_f(r['ret_63d'], pct=True, signed=True)}</td>"
            f"<td>{_f(r['vol_60d'], pct=True)}</td>"
            f"<td>{_f(r['rsi'], digits=0)}</td></tr>"
        )
    return "".join(rows)


def _rank_table(df: pd.DataFrame, new_tickers: set[str]) -> str:
    head = ('<tr><th>順位</th><th class="l">ペア</th><th class="l">名称</th>'
            '<th class="l">区分</th><th>スコア</th><th>金利差</th><th>5年乖離</th>'
            '<th>3ヶ月</th><th>年率ボラ</th><th>RSI</th></tr>')
    return (f'<div class="tblwrap"><table><thead>{head}</thead>'
            f"<tbody>{_rank_rows(df, new_tickers)}</tbody></table></div>")


def _breakdown_table(df: pd.DataFrame) -> str:
    head = ('<tr><th class="l">ペア</th>'
            + "".join(f"<th>{label}</th>" for _, label in CATEGORY_LABELS)
            + "</tr>")
    rows = []
    for _, r in df.iterrows():
        cells = "".join(f"<td>{_z_cell(r[f'z_{cat}'])}</td>" for cat, _ in CATEGORY_LABELS)
        rows.append(f'<tr><td class="l"><strong>{html.escape(r["pair"])}</strong></td>{cells}</tr>')
    return (f'<div class="tblwrap"><table><thead>{head}</thead>'
            f"<tbody>{''.join(rows)}</tbody></table></div>")


def build_body(ranked: pd.DataFrame, run_date: str, cfg: dict,
               new_tickers: set[str], weights: dict[str, float],
               rates_as_of: str) -> str:
    top_n = cfg["top_n"]
    jpy = ranked[ranked["market"] == "円クロス"]

    chips = "".join(
        f'<span class="chip">{WEIGHT_LABELS[k]} {v * 100:.0f}%</span>'
        for k, v in weights.items())

    return f"""<style>{STYLE}</style>
<div class="scr"><div class="wrap">
<header>
  <h1>FXスクリーニング結果</h1>
  <div class="date">{run_date} 実行</div>
  <div class="stats">
    <div class="stat"><div class="k">対象ペア</div><div class="v">{len(ranked)}</div></div>
    <div class="stat"><div class="k">円クロス</div><div class="v">{len(jpy)}</div></div>
    <div class="stat"><div class="k">政策金利基準</div><div class="v" style="font-size:1rem">{rates_as_of}</div></div>
  </div>
  <div class="weights">{chips}</div>
</header>
<section>
  <h2>総合ランキング (全ペア)</h2>
  <p class="note">スコアは「基軸通貨の買い(ロング)」の魅力度。低スコアは売り方向の妙味を示す場合があります。</p>
  {_rank_table(ranked, new_tickers)}
</section>
<section>
  <h2>スコア内訳 (トップ{top_n})</h2>
  <p class="note">全ペア内での偏差 (右向き青 = 平均より良い ／ 左向き赤 = 悪い、±3で頭打ち)</p>
  {_breakdown_table(ranked.head(top_n))}
</section>
<footer>金利差は各国政策金利({rates_as_of}時点の設定値)の差であり、実際のスワップポイントとは異なります。本レポートは機械的なスクリーニング結果であり、投資助言ではありません。FXはレバレッジにより損失が拡大するリスクがあります。データ: Yahoo Finance (yfinance)</footer>
</div></div>"""


def build_page(ranked: pd.DataFrame, run_date: str, cfg: dict,
               new_tickers: set[str], weights: dict[str, float],
               rates_as_of: str) -> str:
    body = build_body(ranked, run_date, cfg, new_tickers, weights, rates_as_of)
    return (
        "<!doctype html>\n<html lang=\"ja\">\n<head>\n"
        "<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"<title>FXスクリーニング {run_date}</title>\n"
        "</head>\n<body style=\"margin:0\">\n"
        f"{body}\n</body>\n</html>\n"
    )


def save_html(ranked: pd.DataFrame, run_date: str, cfg: dict,
              new_tickers: set[str], weights: dict[str, float],
              rates_as_of: str, output_dir: str | Path) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "index.html"
    path.write_text(build_page(ranked, run_date, cfg, new_tickers, weights, rates_as_of),
                    encoding="utf-8")
    return path
