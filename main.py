"""
main.py
End-to-end orchestrator for the HMLFF-Net pipeline.

Run with::

    python -m hmlff_net.main --pipeline all          # full pipeline
    python -m hmlff_net.main --pipeline features     # feature extraction only
    python -m hmlff_net.main --pipeline train        # training + eval
    python -m hmlff_net.main --pipeline xai           # explainability only

Every stage logs progress with tqdm and writes artifacts under ``outputs/``.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from . import config as cfg
from . import utils
from .augmentation import TorchAugmentor
from .database import FeatureDatabase
from .deep_features import DeepFeatureBank
from .excel_export import build_columns, export_csv, export_excel, rows_to_dataframe
from .feature_extraction import FEATURE_NAMES, HandcraftedExtractor
from .feature_fusion import NumpyAdaptiveFusion, fusion_feature_names
from .feature_selection import FeatureSelector
from .gradcam import explain_all_methods, lime_explain
from .preprocessing import ImagePreprocessor
from .segmentation import SegmentationPipeline
from .testing import predict_dataset
from .evaluation import Evaluator
from . import visualization as viz

log = utils.get_logger()


# ===========================================================================
# Stage 1 — Feature extraction over the whole dataset
# ===========================================================================
def run_feature_extraction(dataset_dir: Path | None = None) -> pd.DataFrame:
    dataset_dir = dataset_dir or cfg.DATASET_DIR
    records = utils.discover_images(dataset_dir)
    if not records:
        log.warning("No images found in %s.", dataset_dir)
        return pd.DataFrame()

    pre = ImagePreprocessor()
    seg = SegmentationPipeline(use_yolo=True)
    hand = HandcraftedExtractor()
    deep = DeepFeatureBank()
    trans = None
    try:
        from .transformer_features import TransformerFeatureBank
        trans = TransformerFeatureBank()
    except Exception as e:
        log.warning("Transformer bank unavailable (%s).", e)

    fusion = NumpyAdaptiveFusion(
        len(FEATURE_NAMES), cfg.NUM_DEEP_CNN, cfg.NUM_DEEP_VIT
    )
    fusion_names = fusion_feature_names()

    columns = build_columns(FEATURE_NAMES, fusion_names)
    rows: List[dict] = []
    from tqdm import tqdm

    for rec in tqdm(records, desc="Extracting features"):
        try:
            img = cv2_imread(rec["path"])
            pimg = pre.process(img)
            seg_res = seg.segment(
                (pimg * 255).astype(np.uint8),
                save_path=cfg.SEGMENTED_DIR / rec["crop"] /
                          rec["disease"] / rec["name"],
            )
            seg_img = seg_res.segmented if seg_res.mask.sum() > 0 else (pimg * 255).astype(np.uint8)

            h_feats = hand.extract(seg_img, seg_res.mask, seg_res.disease_fraction)
            d_feats = deep.extract(seg_img)
            t_feats = trans.extract(seg_img) if trans else {
                "vit": np.zeros(cfg.NUM_DEEP_VIT, np.float32),
                "mobilevit": np.zeros(cfg.NUM_DEEP_MOBILEVIT, np.float32),
                "transformer_concat": np.zeros(cfg.NUM_DEEP_VIT + cfg.NUM_DEEP_MOBILEVIT, np.float32),
            }
            cnn_vec = d_feats["efficientnet"]
            vit_vec = t_feats["vit"]
            fused, _ = fusion.fuse(h_feats["full"], cnn_vec, vit_vec)

            row = {
                "ImageName": rec["name"], "Crop": rec["crop"], "Disease": rec["disease"],
            }
            for n, v in zip(FEATURE_NAMES, h_feats["full"]):
                row[n] = float(v)
            for n, v in zip(fusion_names, fused):
                row[n] = float(v)
            row["Prediction"] = rec["disease"]  # placeholder; filled by testing
            row["DiseaseSeverity"] = cfg.severity_label(seg_res.disease_fraction)
            row["ConfidenceScore"] = 0.0
            row["Probability"] = 0.0
            rows.append(row)
        except Exception as e:
            log.error("Failed %s: %s", rec["path"], e)

    df = rows_to_dataframe(rows, columns)
    cfg.make_dirs()
    export_excel(df)
    export_csv(df)
    with FeatureDatabase() as db:
        db.insert_dataframe(df)
    log.info("Feature extraction complete: %d images.", len(df))
    return df


# ===========================================================================
# Stage 2 — Feature selection
# ===========================================================================
def run_feature_selection(df: pd.DataFrame) -> List[str]:
    feat_cols = [c for c in df.columns if c not in
                 ["ImageName", "Crop", "Disease", "Prediction",
                  "DiseaseSeverity", "ConfidenceScore", "Probability"]]
    label = df.apply(lambda r: f"{r['Crop']}_{r['Disease']}", axis=1)
    y = np.array([cfg.CLASS_INDEX.get(l, 0) for l in label])
    X = df[feat_cols].fillna(0).values.astype(np.float32)
    sel = FeatureSelector()
    sel.fit(X, y, feat_cols)
    viz.plot_feature_importance(sel.importances_)
    return sel.selected()


# ===========================================================================
# Stage 3 — Training (comparative study + proposed model)
# ===========================================================================
def run_training(df: pd.DataFrame) -> Dict:
    from .dataset import CropDiseaseDataset
    from .training import cross_validate, train_one_model, _make_loader

    items = _build_dataset_items(df)
    train_tf = TorchAugmentor(train=True)
    val_tf = TorchAugmentor(train=False)

    results: Dict[str, dict] = {}
    for model_name in cfg.MODEL_NAMES:
        log.info("############ Training %s ############", model_name)
        try:
            res = cross_validate(
                model_name, items, cfg.NUM_CLASSES,
                n_splits=cfg.TRAIN_CFG.n_splits,
                epochs=cfg.TRAIN_CFG.epochs,
                lr=cfg.TRAIN_CFG.learning_rate,
                dropout=cfg.TRAIN_CFG.dropout,
                weight_decay=cfg.TRAIN_CFG.weight_decay,
                optimizer_name=cfg.TRAIN_CFG.optimizer,
                loss_name=cfg.TRAIN_CFG.loss,
                hand_dim=len(FEATURE_NAMES) if model_name == "HMLFFNet" else None,
            )
            results[model_name] = res
            log.info("%s mean_acc=%.4f ± %.4f", model_name,
                     res["mean_acc"], res["std_acc"])
            # plot first fold history
            if res["fold_results"]:
                viz.plot_history(res["fold_results"][0]["history"], model_name)
        except Exception as e:
            log.error("Training %s failed: %s", model_name, e)
            results[model_name] = {"error": str(e)}

    # Train final proposed model on a single split for evaluation + XAI
    try:
        from sklearn.model_selection import train_test_split
        labels = [it["label_idx"] for it in items]
        tr_idx, va_idx = train_test_split(
            range(len(items)), test_size=0.2, random_state=cfg.SEED,
            stratify=labels)
        tr_ds = CropDiseaseDataset([items[i] for i in tr_idx], train_tf)
        va_ds = CropDiseaseDataset([items[i] for i in va_idx], val_tf)
        final = train_one_model(
            "HMLFFNet", _make_loader(tr_ds, train=True),
            _make_loader(va_ds, train=False), cfg.NUM_CLASSES,
            epochs=cfg.TRAIN_CFG.epochs, lr=cfg.TRAIN_CFG.learning_rate,
            dropout=cfg.TRAIN_CFG.dropout,
            weight_decay=cfg.TRAIN_CFG.weight_decay,
            optimizer_name=cfg.TRAIN_CFG.optimizer,
            loss_name=cfg.TRAIN_CFG.loss, hand_dim=len(FEATURE_NAMES),
        )
        # Evaluate
        preds = predict_dataset(final["model"], _make_loader(va_ds, train=False),
                                is_hmlff=True)
        ev = Evaluator.evaluate(preds["labels"], preds["preds"],
                                 preds["probs"], cfg.NUM_CLASSES)
        results["HMLFFNet_final"] = {"metrics": ev,
                                       "best_epoch": final["best_epoch"]}
        viz.plot_history(final["history"], "HMLFFNet_final")
        viz.plot_confusion(ev["confusion_matrix"],
                           list(cfg.INDEX_CLASS.keys()), "HMLFFNet_final")
        if ev.get("roc_curves"):
            viz.plot_roc(ev["roc_curves"], "HMLFFNet_final")
            viz.plot_pr(ev["pr_curves"], "HMLFFNet_final")
        with FeatureDatabase() as db:
            db.log_run("HMLFFNet", asdict(cfg.TRAIN_CFG), ev)
        _save_final_model(final["model"])
    except Exception as e:
        log.error("Final training failed: %s", e)

    return results


def _build_dataset_items(df: pd.DataFrame) -> List[dict]:
    from .dataset import _default_tensor
    items = []
    for _, r in df.iterrows():
        label = f"{r['Crop']}_{r['Disease']}"
        # We reconstruct image path from name (not stored); use segmented dir.
        seg_path = (cfg.SEGMENTED_DIR / r["Crop"] / r["Disease"] / r["ImageName"])
        if not seg_path.exists():
            continue
        img = cv2_imread(seg_path)
        hand = np.array([r.get(n, 0.0) for n in FEATURE_NAMES], dtype=np.float32)
        items.append({
            "image": img, "label_idx": cfg.CLASS_INDEX.get(label, 0),
            "hand": hand, "path": seg_path, "name": r["ImageName"],
            "crop": r["Crop"], "disease": r["Disease"],
        })
    return items


def _save_final_model(model) -> None:
    try:
        import torch
        path = cfg.MODEL_DIR / "hmlff_net_final.pth"
        torch.save(model.state_dict(), path)
        log.info("Saved final model to %s", path)
    except Exception as e:
        log.warning("Model save failed: %s", e)


# ===========================================================================
# Stage 4 — Explainable AI
# ===========================================================================
def run_xai(df: pd.DataFrame, n_samples: int = 20) -> None:
    try:
        import torch
        from .gradcam import explain_all_methods, lime_explain
        from .training import build_hmlff_net
    except Exception as e:
        log.warning("XAI deps missing: %s", e)
        return

    items = _build_dataset_items(df)
    if not items:
        log.warning("No items for XAI.")
        return
    model = build_hmlff_net(cfg.NUM_CLASSES, len(FEATURE_NAMES)).to(cfg.DEVICE)
    ckpt = cfg.MODEL_DIR / "hmlff_net_final.pth"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=cfg.DEVICE))
    model.eval()

    for it in items[:n_samples]:
        try:
            img = it["image"]
            from .dataset import _default_tensor
            x = _default_tensor(img).unsqueeze(0).to(cfg.DEVICE)
            explain_all_methods(model, x, img, Path(it["name"]).stem,
                                target_class=it["label_idx"])
            lime_explain(model, img, Path(it["name"]).stem)
        except Exception as e:
            log.error("XAI for %s failed: %s", it["name"], e)


# ===========================================================================
# Stage 5 — PDF report
# ===========================================================================
def run_pdf_report(df: pd.DataFrame, results: Dict) -> None:
    try:
        from .report import generate_pdf_report
        generate_pdf_report(df, results)
    except Exception as e:
        log.warning("PDF report failed: %s", e)


# ===========================================================================
# IO helpers
# ===========================================================================
def cv2_imread(path: Path) -> "np.ndarray":
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        # try unicode path workaround
        import numpy as np
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


# ===========================================================================
# CLI
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="HMLFF-Net pipeline")
    parser.add_argument("--pipeline", default="all",
                        choices=["all", "features", "selection",
                                 "train", "xai", "report"])
    parser.add_argument("--dataset", default=str(cfg.DATASET_DIR))
    args = parser.parse_args()

    utils.set_seed(cfg.SEED)
    cfg.make_dirs()
    log.info("Device: %s", cfg.DEVICE)

    df = pd.DataFrame()
    csv_path = cfg.CSV_PATH
    if args.pipeline in ("all", "features"):
        df = run_feature_extraction(Path(args.dataset))
    elif csv_path.exists():
        df = pd.read_csv(csv_path)

    if args.pipeline in ("all", "selection") and not df.empty:
        selected = run_feature_selection(df)
        log.info("Selected features: %d", len(selected))

    results: Dict = {}
    if args.pipeline in ("all", "train") and not df.empty:
        results = run_training(df)

    if args.pipeline in ("all", "xai") and not df.empty:
        run_xai(df)

    if args.pipeline in ("all", "report") and not df.empty:
        run_pdf_report(df, results)

    log.info("Pipeline '%s' finished.", args.pipeline)


if __name__ == "__main__":
    main()