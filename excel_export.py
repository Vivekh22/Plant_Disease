"""
excel_export.py
Export the full per-image feature table to Excel (.xlsx) with auto-generated
column headers, plus a CSV mirror.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import pandas as pd

from . import config as cfg

log = logging.getLogger("hmlff_net.excel_export")


# Header columns that come *before* the numeric features.
META_COLS = ["ImageName", "Crop", "Disease"]
TAIL_COLS = ["Prediction", "DiseaseSeverity", "ConfidenceScore", "Probability"]


def build_columns(feature_names: List[str], fusion_names: List[str]) -> List[str]:
    """Assemble the full ordered column list for the export."""
    return META_COLS + feature_names + fusion_names + TAIL_COLS


def rows_to_dataframe(rows: List[dict], columns: List[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=columns)
    return df


def export_excel(df: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or cfg.EXCEL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Features")
        # Summary sheet
        try:
            summary = df.describe(include="all").T
            summary.to_excel(writer, sheet_name="Summary")
        except Exception:
            pass
    log.info("Excel written: %s (%d rows, %d cols)", path, df.shape[0], df.shape[1])
    return path


def export_csv(df: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or cfg.CSV_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info("CSV written: %s", path)
    return path