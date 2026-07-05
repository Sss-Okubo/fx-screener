"""エントリポイント: python -m screener run"""
from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

import yaml

from . import backtest, fetch, mtf, notify, report, report_html, scoring, universe
from .store import Store

logger = logging.getLogger("screener")


def run(args: argparse.Namespace) -> None:
    base = Path(args.config).resolve().parent
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run_date = date.today().isoformat()

    store = Store(base / cfg["data"]["db_path"])
    limit = args.limit or cfg["universe"]["limit"]

    # 1. ユニバース取得
    uni = universe.get_universe(limit)

    # 2. データ取得 (キャッシュ付き)
    tickers = uni["ticker"].tolist()
    force = 0 if args.force_refresh else 1
    fetch.fetch_prices(store, tickers, cfg["data"]["price_period"],
                       cfg["data"]["prices_cache_days"] * force)

    # 3. スコアリング
    features = scoring.build_features(store, uni, cfg["rates"]["policy"])
    if features.empty:
        logger.error("スコアリング対象のペアがありません")
        return
    ranked = scoring.score(features, cfg["scoring"])

    # 4. 前回結果との比較 → レポート生成
    top_n = cfg["report"]["top_n"]
    rates_as_of = str(cfg["rates"]["as_of"])
    prev_top = store.previous_top_keys(run_date, top_n)
    top_rows = ranked[ranked["rank"] <= top_n]
    current_top = set(zip(top_rows["ticker"], top_rows["direction"]))
    new_keys = current_top - prev_top if prev_top else set()

    text = report.build_report(ranked, run_date, cfg["report"], new_keys, rates_as_of)
    path = report.save_report(text, run_date, base / cfg["report"]["output_dir"])
    logger.info("レポートを保存: %s", path)

    html_path = report_html.save_html(ranked, run_date, cfg["report"], new_keys,
                                      cfg["scoring"]["weights"], rates_as_of,
                                      base / cfg["report"]["output_dir"])
    logger.info("HTMLレポートを保存: %s", html_path)

    store.save_results(run_date, ranked)

    # 5. 通知
    if cfg["notify"]["enabled"] and not args.no_notify:
        notify.send_discord(ranked, run_date, top_n, new_keys)

    store.close()

    # コンソールにもトップ表示
    cols = ["rank", "direction", "pair", "name", "score"]
    for direction in ["買い", "売り"]:
        sub = ranked[ranked["direction"] == direction].sort_values("rank")
        print(sub.head(top_n)[cols].to_string(index=False))


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(prog="screener", description="FX通貨ペアスクリーナー")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="スクリーニングを実行")
    p_run.add_argument("--config", default=str(Path(__file__).parent.parent / "config.yaml"))
    p_run.add_argument("--limit", type=int, help="ペア数を制限 (テスト用)")
    p_run.add_argument("--no-notify", action="store_true", help="通知を送らない")
    p_run.add_argument("--force-refresh", action="store_true", help="キャッシュを無視して再取得")
    p_run.set_defaults(func=run)

    p_bt = sub.add_parser("backtest", help="選定ルールを過去データで検証")
    p_bt.add_argument("--config", default=str(Path(__file__).parent.parent / "config.yaml"))
    p_bt.add_argument("--years", type=int, default=5, help="検証年数 (デフォルト5年)")
    p_bt.add_argument("--top", type=int, default=3, help="保有ポジション数 (デフォルト3)")
    p_bt.add_argument("--limit", type=int, help="ペア数を制限 (テスト用)")
    p_bt.add_argument("--force-refresh", action="store_true", help="キャッシュを無視して再取得")
    p_bt.set_defaults(func=backtest.run)

    p_mtf = sub.add_parser("mtf", help="15分/1時間/4時間足のトレンドそろい判定")
    p_mtf.add_argument("--config", default=str(Path(__file__).parent.parent / "config.yaml"))
    p_mtf.add_argument("--limit", type=int, help="ペア数を制限 (テスト用)")
    p_mtf.add_argument("--no-notify", action="store_true", help="通知を送らない")
    p_mtf.set_defaults(func=mtf.run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
