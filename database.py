"""
database.py
SQLite persistence for extracted features.  Schema mirrors the Excel
columns; tables: ``features`` (per image) and ``runs`` (training metadata).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import List

import pandas as pd

from . import config as cfg

log = logging.getLogger("hmlff_net.database")


class FeatureDatabase:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or cfg.SQLITE_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created TEXT,
                model TEXT,
                config TEXT,
                metrics TEXT
            )"""
        )

    def insert_dataframe(self, df: pd.DataFrame, table: str = "features") -> None:
        df.to_sql(table, self.conn, if_exists="replace", index=False)
        self.conn.commit()
        log.info("Stored %d rows in SQLite table '%s'.", len(df), table)

    def query(self, sql: str) -> List[dict]:
        cur = self.conn.execute(sql)
        return [dict(r) for r in cur.fetchall()]

    def log_run(self, model: str, config: dict, metrics: dict) -> None:
        import datetime
        self.conn.execute(
            "INSERT INTO runs (created, model, config, metrics) VALUES (?, ?, ?, ?)",
            (datetime.datetime.now().isoformat(), model,
             json.dumps(config), json.dumps(metrics)),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()