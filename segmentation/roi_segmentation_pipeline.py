"""
roi_segmentation_pipeline.py
─────────────────────────────
Orchestrator for the Mask R-CNN ROI Segmentation Pipeline.

This file is intentionally thin — it only handles:
  • CLI argument parsing
  • Loading segmentation_config.ini  (dedicated) + config.ini  (project-wide)
  • Logger initialisation
  • Dispatching to the three stage modules

Module layout
─────────────
  seg_dataset.py      — VIA JSON parsing, mask rasterisation, Dataset, COCO export
  seg_trainer.py      — Mask R-CNN construction, training loop, checkpointing
  seg_predictor.py    — Inference, mask + overlay saving, CSV update
  segmentation_config.ini — All hyper-parameters and paths

Usage
─────
    # Run all three stages in sequence
    python roi_segmentation_pipeline.py --stage all

    # Run individual stages
    python roi_segmentation_pipeline.py --stage prepare
    python roi_segmentation_pipeline.py --stage train
    python roi_segmentation_pipeline.py --stage predict

    # Override checkpoint for inference
    python roi_segmentation_pipeline.py --stage predict \
        --checkpoint path/to/custom_model.pth

    # Use a non-default segmentation config file
    python roi_segmentation_pipeline.py --stage all \
        --seg-config path/to/segmentation_config.ini

Requirements
────────────
    pip install torch torchvision opencv-python numpy pandas tqdm
"""

from __future__ import annotations

import argparse
import configparser
import json
import sys
from pathlib import Path
from typing import List

# ── Project utilities (mirrors existing pipeline conventions) ─────────────────
from src.common import intraoral_logger as iolog
from utils.load_configuration import load_config
from utils.log_handler import CustomSizeDayRotatingFileHandler

# ── Stage modules ─────────────────────────────────────────────────────────────
from seg_dataset   import stage_prepare
from seg_trainer   import stage_train
from seg_predictor import stage_predict


# ─────────────────────────────────────────────────────────────────────────────
# Config loader
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_SEG_CONFIG = Path(__file__).parent / "segmentation_config.ini"


def load_seg_config(path: str) -> configparser.ConfigParser:
    """
    Load the dedicated segmentation config INI file.

    Raises FileNotFoundError with a clear message if the file is absent
    so users know exactly which file is missing.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Segmentation config not found: {p}\n"
            "Create segmentation_config.ini or pass --seg-config <path>."
        )
    cfg = configparser.ConfigParser()
    cfg.read(str(p))
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Split-record reload helper (for standalone --stage train / predict)
# ─────────────────────────────────────────────────────────────────────────────

def _load_records_from_coco(
    coco_dir: str,
    split_name: str,
    metadata_csv: str,
) -> List[dict]:
    """
    Reconstruct a record list from a previously written COCO JSON split.
    Used when --stage train or --stage predict is called without --stage prepare
    having run in the same process.
    """
    import pandas as pd

    coco_path = Path(coco_dir) / f"{split_name}.json"
    if not coco_path.exists():
        return []

    df = pd.read_csv(metadata_csv)
    records: List[dict] = []

    with open(coco_path) as fh:
        coco = json.load(fh)

    for img_entry in coco["images"]:
        fp   = img_entry["file_name"]
        rows = df[df["image_path"] == fp]
        if len(rows):
            rec = {
                "image_path": fp,
                "json_file":  str(rows["json_file"].iloc[0]),
                "patient_id": str(rows["patient_id"].iloc[0]),
                "label":      str(rows["label"].iloc[0]),
            }
        else:
            rec = {"image_path": fp, "json_file": "", "patient_id": "", "label": ""}
        records.append(rec)

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Logger initialisation
# ─────────────────────────────────────────────────────────────────────────────

def initialize_logger(config) -> CustomSizeDayRotatingFileHandler:
    log_filename = config.get("LOGGER", "logger.filename")
    return iolog.getLogger(log_filename)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Mask R-CNN ROI Segmentation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--stage",
        choices=["prepare", "train", "predict", "all"],
        default="all",
        help="Pipeline stage to run (default: all).",
    )
    parser.add_argument(
        "--seg-config",
        default=str(_DEFAULT_SEG_CONFIG),
        metavar="PATH",
        help=(
            "Path to the segmentation config INI file "
            f"(default: {_DEFAULT_SEG_CONFIG})."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        metavar="PATH",
        help="Override the model checkpoint path for the predict stage.",
    )
    args = parser.parse_args()

    # ── Load configs ──────────────────────────────────────────────────────────
    main_cfg = load_config()           # project-wide config.ini
    seg_cfg  = load_seg_config(args.seg_config)   # segmentation_config.ini

    # ── Logger ────────────────────────────────────────────────────────────────
    logger = initialize_logger(main_cfg)

    logger.info("=" * 60)
    logger.info("  ROI Segmentation Pipeline  (Mask R-CNN)")
    logger.info(f"  Stage           : {args.stage}")
    logger.info(f"  Seg config file : {args.seg_config}")
    logger.info("=" * 60)

    # ── Stage: prepare ────────────────────────────────────────────────────────
    train_recs, val_recs, test_recs = [], [], []

    if args.stage in ("prepare", "all"):
        train_recs, val_recs, test_recs = stage_prepare(
            logger=logger,
            seg_cfg=seg_cfg,
            main_cfg=main_cfg,
        )

    # ── Stage: train ──────────────────────────────────────────────────────────
    if args.stage in ("train", "all"):
        # If records not populated (standalone --stage train), reload from COCO
        if not train_recs:
            coco_dir     = seg_cfg.get("PATHS", "seg.coco.dir")
            metadata_csv = seg_cfg.get("PATHS", "merged.metadata.csv")
            train_recs = _load_records_from_coco(coco_dir, "train", metadata_csv)
            val_recs   = _load_records_from_coco(coco_dir, "val",   metadata_csv)
            test_recs  = _load_records_from_coco(coco_dir, "test",  metadata_csv)

            if not train_recs:
                logger.error(
                    "No train records found. "
                    "Run --stage prepare first or provide COCO JSON splits."
                )
                sys.exit(1)

        stage_train(
            logger=logger,
            seg_cfg=seg_cfg,
            train_recs=train_recs,
            val_recs=val_recs,
        )

    # ── Stage: predict ────────────────────────────────────────────────────────
    if args.stage in ("predict", "all"):
        stage_predict(
            logger=logger,
            seg_cfg=seg_cfg,
            main_cfg=main_cfg,
            checkpoint_override=args.checkpoint,
        )

    logger.info("=" * 60)
    logger.info("  Pipeline finished.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
