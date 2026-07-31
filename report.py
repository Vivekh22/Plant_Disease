"""
report.py
Generate a PDF report summarizing the pipeline results using ReportLab.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

from . import config as cfg

log = logging.getLogger("hmlff_net.report")


def generate_pdf_report(df: pd.DataFrame, results: Dict) -> Path:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Table, Image, PageBreak)

    cfg.make_dirs()
    path = cfg.PDF_REPORT_PATH
    doc = SimpleDocTemplate(str(path), pagesize=A4)
    styles = getSampleStyleSheet()
    story: List = []

    story.append(Paragraph("HMLFF-Net — Crop Disease Prediction Report", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "Hybrid Multi-Level Feature Fusion Network for Multi-Crop Disease "
        "Prediction using Deep Learning.", styles["Normal"]))
    story.append(Spacer(1, 12))

    # Dataset summary
    story.append(Paragraph("Dataset Summary", styles["Heading2"]))
    summary = df.groupby(["Crop", "Disease"]).size().reset_index(name="Count")
    tbl_data = [summary.columns.tolist()] + summary.values.tolist()
    story.append(Table(tbl_data, repeatRows=1))
    story.append(Spacer(1, 12))

    # Model results
    story.append(Paragraph("Model Results (5-fold CV mean accuracy)", styles["Heading2"]))
    rows = [["Model", "Mean Acc", "Std Acc"]]
    for name, res in results.items():
        if isinstance(res, dict) and "mean_acc" in res:
            rows.append([name, f"{res['mean_acc']:.4f}", f"{res['std_acc']:.4f}"])
    story.append(Table(rows, repeatRows=1))
    story.append(Spacer(1, 12))

    # Embed available figures
    story.append(PageBreak())
    story.append(Paragraph("Generated Figures", styles["Heading2"]))
    for fig in sorted(cfg.FIGURE_DIR.glob("*.png")):
        try:
            story.append(Paragraph(fig.stem, styles["Normal"]))
            story.append(Image(str(fig), width=380, height=280))
            story.append(Spacer(1, 8))
        except Exception:
            continue

    doc.build(story)
    log.info("PDF report saved: %s", path)
    return path