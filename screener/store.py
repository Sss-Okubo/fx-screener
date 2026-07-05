"""SQLiteキャッシュストア"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    open REAL, high REAL, low REAL, close REAL,
    PRIMARY KEY (ticker, date)
);
CREATE TABLE IF NOT EXISTS price_meta (
    ticker TEXT PRIMARY KEY,
    fetched_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS results (
    run_date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    rank INTEGER,
    score REAL,
    PRIMARY KEY (run_date, ticker)
);
"""


class Store:
    def __init__(self, db_path: str | Path):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.executescript(SCHEMA)

    # ---- 価格 ----
    def fresh_price_tickers(self, max_age_days: float) -> set[str]:
        cutoff = (datetime.now() - timedelta(days=max_age_days)).isoformat()
        rows = self.conn.execute(
            "SELECT ticker FROM price_meta WHERE fetched_at > ?", (cutoff,)
        ).fetchall()
        return {r[0] for r in rows}

    def save_prices(self, ticker: str, df: pd.DataFrame) -> None:
        """df: index=DatetimeIndex, columns=[Open, High, Low, Close]"""
        rows = [
            (ticker, idx.strftime("%Y-%m-%d"),
             float(r["Open"]), float(r["High"]), float(r["Low"]), float(r["Close"]))
            for idx, r in df.iterrows()
        ]
        with self.conn:
            self.conn.executemany(
                "INSERT OR REPLACE INTO prices VALUES (?,?,?,?,?,?)", rows)
            self.conn.execute(
                "INSERT OR REPLACE INTO price_meta VALUES (?,?)",
                (ticker, datetime.now().isoformat()))

    def load_prices(self, ticker: str) -> pd.DataFrame:
        df = pd.read_sql_query(
            "SELECT date, open, high, low, close FROM prices "
            "WHERE ticker = ? ORDER BY date", self.conn, params=(ticker,))
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")

    # ---- 選定結果履歴 ----
    def save_results(self, run_date: str, ranked: pd.DataFrame) -> None:
        rows = [(run_date, r["ticker"], int(r["rank"]), float(r["score"]))
                for _, r in ranked.iterrows()]
        with self.conn:
            self.conn.executemany(
                "INSERT OR REPLACE INTO results VALUES (?,?,?,?)", rows)

    def previous_top_tickers(self, before_date: str, top_n: int) -> set[str]:
        row = self.conn.execute(
            "SELECT MAX(run_date) FROM results WHERE run_date < ?",
            (before_date,)).fetchone()
        if not row or not row[0]:
            return set()
        rows = self.conn.execute(
            "SELECT ticker FROM results WHERE run_date = ? AND rank <= ?",
            (row[0], top_n)).fetchall()
        return {r[0] for r in rows}

    def close(self) -> None:
        self.conn.close()
