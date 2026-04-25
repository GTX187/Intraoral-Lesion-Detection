"""
seg_dataset.py
──────────────
Dataset module for the Mask R-CNN ROI Segmentation Pipeline.

Responsibilities:
  • Parse VIA-format polygon / rect JSON annotations
  • Rasterise polygon regions into binary instance masks
  • PyTorch Dataset class (OralLesionDataset)
  • Patient-level train / val / test split
  • COCO-format JSON export
  • stage_prepare() orchestrator

Public API
----------
  stage_prepare(logger, seg_cfg, main_cfg) -> (train_recs, val_recs, test_recs)
  OralLesionDataset(records)
  collate_fn(batch)
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data_utils
import torchvision.transforms.functional as TF

from utils.log_handler import CustomSizeDayRotatingFileHandler


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

FOREGROUND_LABEL = 1   # class index for "lesion"; 0 = background


# ─────────────────────────────────────────────────────────────────────────────
# VIA JSON helpers
# ─────────────────────────────────────────────────────────────────────────────


def load_via_json(json_path: str) -> dict:
    """Load a VIA annotation JSON from disk."""
    with open(json_path, "r") as fh:
        return json.load(fh)


def _image_key_from_via(via_data: dict, image_filename: str) -> Optional[str]:
    """
    VIA keys are '<filename><filesize>' or just '<filename>'.
    Locate the entry whose stored filename matches image_filename.
    Falls back to stem-only comparison for exports that strip extensions.
    """
    target_name = Path(image_filename).name
    target_stem = Path(image_filename).stem
    for key, entry in via_data.items():
        stored = entry.get("filename", "")
        if stored == target_name:
            return key
        if Path(stored).stem == target_stem:
            return key
    return None


def polygon_to_mask(
    all_points_x: List[float],
    all_points_y: List[float],
    img_h: int,
    img_w: int,
) -> np.ndarray:
    """
    Rasterise one VIA polygon into a binary uint8 mask of shape (H, W).
    Uses cv2.fillPoly for sub-pixel accuracy.
    """
    pts = np.array(
        [[int(round(x)), int(round(y))]
         for x, y in zip(all_points_x, all_points_y)],
        dtype=np.int32,
    )
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def extract_instance_masks(
    via_data: dict,
    image_filename: str,
    img_h: int,
    img_w: int,
) -> Tuple[List[np.ndarray], List[List[int]]]:
    """
    Extract all annotated regions for one image from a VIA JSON dict.

    Handles:
      • polygon  — used as-is
      • rect     — converted to a 4-corner polygon so both shapes share the
                   same rasterisation path

    Returns
    -------
    masks : list of (H × W) uint8 binary arrays, one per instance
    boxes : list of [x1, y1, x2, y2] tight bounding boxes (ints)
    """
    key = _image_key_from_via(via_data, image_filename)
    if key is None:
        return [], []

    regions = via_data[key].get("regions", [])
    # VIA v1 stores regions as a dict keyed by string index
    if isinstance(regions, dict):
        regions = list(regions.values())

    masks: List[np.ndarray] = []
    boxes: List[List[int]]  = []

    for region in regions:
        sa = region.get("shape_attributes", {})
        shape_name = sa.get("name", "")

        if shape_name == "rect":
            # Normalise rect → polygon so both take the same path below
            rx, ry, rw, rh = sa["x"], sa["y"], sa["width"], sa["height"]
            sa = {
                "name":         "polygon",
                "all_points_x": [rx,      rx + rw, rx + rw, rx],
                "all_points_y": [ry,      ry,      ry + rh, ry + rh],
            }
            shape_name = "polygon"

        if shape_name != "polygon":
            continue

        xs = sa.get("all_points_x", [])
        ys = sa.get("all_points_y", [])
        if len(xs) < 3:
            continue

        mask = polygon_to_mask(xs, ys, img_h, img_w)
        if mask.sum() == 0:
            continue

        # Derive tight bounding box from filled mask (handles concave polygons)
        row_any = np.any(mask, axis=1)
        col_any = np.any(mask, axis=0)
        rmin, rmax = int(np.where(row_any)[0][[0, -1]].tolist()[0]), \
                     int(np.where(row_any)[0][[0, -1]].tolist()[1])
        cmin, cmax = int(np.where(col_any)[0][[0, -1]].tolist()[0]), \
                     int(np.where(col_any)[0][[0, -1]].tolist()[1])

        boxes.append([cmin, rmin, cmax, rmax])
        masks.append(mask)

    return masks, boxes


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────


class OralLesionDataset(data_utils.Dataset):
    """
    One item = one image with its full set of instance masks and boxes.

    Each record dict must contain:
        image_path  : str   path to the image file
        json_file   : str   path to VIA annotation JSON (may be NaN / empty)
        label       : str   e.g. 'opmd', 'variation', 'normal'
        patient_id  : str

    Images with no valid annotation return an empty target (zero instances),
    which is still valid for Mask R-CNN — the model learns to output nothing
    for unannotated frames.
    """

    def __init__(self, records: List[dict]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec       = self.records[idx]
        img_path  = rec["image_path"]
        json_path = rec.get("json_file")

        # ── Load image ────────────────────────────────────────────────────────
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        img_rgb      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_h, img_w = img_rgb.shape[:2]

        # ── Load instance masks from VIA JSON ─────────────────────────────────
        masks_list: List[np.ndarray] = []
        boxes_list: List[List[int]]  = []

        if json_path and Path(str(json_path)).exists():
            try:
                via_data = load_via_json(str(json_path))
                masks_list, boxes_list = extract_instance_masks(
                    via_data, Path(img_path).name, img_h, img_w
                )
            except Exception:
                pass  # Silently fall back to zero instances

        # ── Build tensors ─────────────────────────────────────────────────────
        img_t = TF.to_tensor(img_rgb)  # (3, H, W) float32 in [0, 1]

        if masks_list:
            masks_t = torch.as_tensor(
                np.stack(masks_list, axis=0), dtype=torch.uint8
            )  # (N, H, W)
            boxes_t  = torch.as_tensor(boxes_list, dtype=torch.float32)  # (N, 4)
            labels_t = torch.ones(len(masks_list), dtype=torch.int64)    # all = lesion
        else:
            masks_t  = torch.zeros((0, img_h, img_w), dtype=torch.uint8)
            boxes_t  = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros(0, dtype=torch.int64)

        area    = (boxes_t[:, 3] - boxes_t[:, 1]) * (boxes_t[:, 2] - boxes_t[:, 0])
        iscrowd = torch.zeros(len(masks_list), dtype=torch.int64)

        target = {
            "boxes":    boxes_t,
            "labels":   labels_t,
            "masks":    masks_t,
            "image_id": torch.tensor([idx]),
            "area":     area,
            "iscrowd":  iscrowd,
        }

        return img_t, target


def collate_fn(batch):
    """Custom collate for variable-size instance counts per image."""
    return tuple(zip(*batch))


# ─────────────────────────────────────────────────────────────────────────────
# COCO JSON export
# ─────────────────────────────────────────────────────────────────────────────


def build_coco_json(
    records: List[dict],
    split_name: str,
    coco_dir: str,
    logger: CustomSizeDayRotatingFileHandler,
) -> str:
    """
    Build and write a minimal COCO-format JSON for one data split.

    Segmentation is stored as polygon contours (COCO polygon format),
    which is compatible with pycocotools for downstream mAP evaluation.

    Returns the path to the written file.
    """
    images:      List[dict] = []
    annotations: List[dict] = []
    ann_id  = 1
    skipped = 0

    for img_id, rec in enumerate(records, start=1):
        img_path  = rec["image_path"]
        json_path = rec.get("json_file")

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            skipped += 1
            continue
        img_h, img_w = img_bgr.shape[:2]

        images.append({
            "id":        img_id,
            "file_name": str(img_path),
            "height":    img_h,
            "width":     img_w,
        })

        if json_path and Path(str(json_path)).exists():
            try:
                via_data = load_via_json(str(json_path))
                masks_list, boxes_list = extract_instance_masks(
                    via_data, Path(img_path).name, img_h, img_w
                )
            except Exception:
                masks_list, boxes_list = [], []

            for mask, box in zip(masks_list, boxes_list):
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                seg = [c.flatten().tolist() for c in contours if c.size >= 6]
                if not seg:
                    continue

                x1, y1, x2, y2 = box
                annotations.append({
                    "id":           ann_id,
                    "image_id":     img_id,
                    "category_id":  FOREGROUND_LABEL,
                    "segmentation": seg,
                    "bbox":         [x1, y1, x2 - x1, y2 - y1],
                    "area":         float((x2 - x1) * (y2 - y1)),
                    "iscrowd":      0,
                })
                ann_id += 1

    coco_dict = {
        "info":       {"description": f"Oral Lesion — {split_name} split"},
        "categories": [{"id": FOREGROUND_LABEL, "name": "lesion",
                        "supercategory": "oral"}],
        "images":     images,
        "annotations": annotations,
    }

    os.makedirs(coco_dir, exist_ok=True)
    out_path = os.path.join(coco_dir, f"{split_name}.json")
    with open(out_path, "w") as fh:
        json.dump(coco_dict, fh)

    logger.info(
        f"    [{split_name:5s}]  images={len(images):4d}  "
        f"annotations={len(annotations):5d}  skipped={skipped}"
    )
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Prepare
# ─────────────────────────────────────────────────────────────────────────────


def stage_prepare(
    logger: CustomSizeDayRotatingFileHandler,
    seg_cfg,
    main_cfg,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Read merged metadata CSV, filter to annotated rows, perform a
    patient-level stratified split, and write COCO JSON files.

    Parameters
    ----------
    seg_cfg  : ConfigParser loaded from segmentation_config.ini
    main_cfg : ConfigParser loaded from the project config.ini
               (only used to locate the merged CSV if seg_cfg does not
               override it)

    Returns
    -------
    (train_records, val_records, test_records)
    Each record is a dict with keys: image_path, json_file, label, patient_id
    """
    logger.info("=" * 60)
    logger.info("  STAGE 1  —  Prepare dataset")
    logger.info("=" * 60)

    # ── Config reads ──────────────────────────────────────────────────────────
    metadata_csv    = seg_cfg.get("PATHS",   "merged.metadata.csv")
    coco_dir        = seg_cfg.get("PATHS",   "seg.coco.dir")
    val_frac        = seg_cfg.getfloat("DATASET", "seg.val.split",    fallback=0.15)
    test_frac       = seg_cfg.getfloat("DATASET", "seg.test.split",   fallback=0.10)
    split_seed      = seg_cfg.getint  ("DATASET", "seg.split.seed",   fallback=42)
    include_labels_raw = seg_cfg.get  ("DATASET", "seg.include.labels", fallback="")
    include_labels  = (
        [lbl.strip().lower() for lbl in include_labels_raw.split(",") if lbl.strip()]
        if include_labels_raw.strip()
        else []
    )

    logger.info(f"  Metadata CSV   : {metadata_csv}")
    logger.info(f"  COCO output dir: {coco_dir}")
    logger.info(f"  Val fraction   : {val_frac}")
    logger.info(f"  Test fraction  : {test_frac}")
    logger.info(f"  Split seed     : {split_seed}")
    logger.info(
        f"  Label filter   : "
        f"{'ALL' if not include_labels else ', '.join(include_labels)}"
    )

    # ── Load CSV ──────────────────────────────────────────────────────────────
    df = pd.read_csv(metadata_csv)
    logger.info(f"  Total rows in merged CSV: {len(df)}")

    # Optionally restrict to configured labels
    if include_labels:
        df = df[df["label"].str.lower().isin(include_labels)].copy()
        logger.info(f"  Rows after label filter  : {len(df)}")

    # Keep only rows that have a resolvable VIA JSON path
    json_col = df["json_file"].astype(str).str.strip()
    annotated_mask = (
        df["json_file"].notna() &
        json_col.ne("") &
        json_col.str.lower().ne("nan")
    )
    df_ann   = df[annotated_mask].copy()
    n_dropped = len(df) - len(df_ann)
    logger.info(
        f"  Annotated rows           : {len(df_ann)}  "
        f"(dropped {n_dropped} unannotated)"
    )

    # ── Label distribution ────────────────────────────────────────────────────
    logger.info("  Label distribution in annotated set:")
    for lbl, cnt in df_ann["label"].value_counts().items():
        pct = 100 * cnt / len(df_ann)
        logger.info(f"    {lbl:15s}  {cnt:5d}  ({pct:5.1f}%)")

    # ── Patient-level split ───────────────────────────────────────────────────
    patients = df_ann["patient_id"].unique().tolist()
    rng = random.Random(split_seed)
    rng.shuffle(patients)

    n         = len(patients)
    n_test    = max(1, int(n * test_frac))
    n_val     = max(1, int(n * val_frac))
    test_pts  = set(patients[:n_test])
    val_pts   = set(patients[n_test: n_test + n_val])
    train_pts = set(patients[n_test + n_val:])

    logger.info(
        f"  Patient split  →  "
        f"train: {len(train_pts)}  val: {len(val_pts)}  test: {len(test_pts)}"
    )

    records = df_ann[
        ["image_path", "json_file", "label", "patient_id"]
    ].to_dict("records")

    train_recs = [r for r in records if r["patient_id"] in train_pts]
    val_recs   = [r for r in records if r["patient_id"] in val_pts]
    test_recs  = [r for r in records if r["patient_id"] in test_pts]

    logger.info(
        f"  Image split    →  "
        f"train: {len(train_recs)}  val: {len(val_recs)}  test: {len(test_recs)}"
    )

    # ── Write COCO JSONs ──────────────────────────────────────────────────────
    logger.info("  Writing COCO JSON splits …")
    for split_name, split_recs in [
        ("train", train_recs), ("val", val_recs), ("test", test_recs)
    ]:
        build_coco_json(split_recs, split_name, coco_dir, logger)

    logger.info("  STAGE 1 complete.")
    return train_recs, val_recs, test_recs
