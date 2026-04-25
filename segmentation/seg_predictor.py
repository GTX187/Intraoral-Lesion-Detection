"""
seg_predictor.py
────────────────
Inference module for the Mask R-CNN ROI Segmentation Pipeline.

Responsibilities:
  • Load a trained Mask R-CNN checkpoint
  • Run per-image inference with configurable score / IoU thresholds
  • Save combined binary mask PNGs and colour-overlay JPEG visualisations
  • Append mask_path and pred_score columns back to the merged metadata CSV
  • stage_predict() orchestrator

Public API
----------
  stage_predict(logger, seg_cfg, main_cfg, checkpoint_override=None)
  predict_single(model, img_path, device, score_threshold, iou_threshold)
      -> (masks, scores, boxes)
  save_mask_and_overlay(img_path, masks, scores, out_mask, out_overlay,
                        alpha, colour_bgr, jpeg_quality)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm

from seg_trainer import build_maskrcnn
from utils.log_handler import CustomSizeDayRotatingFileHandler


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

FOREGROUND_LABEL = 1   # must match seg_dataset.FOREGROUND_LABEL


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────


def predict_single(
    model:           torch.nn.Module,
    img_path:        str,
    device:          torch.device,
    score_threshold: float = 0.50,
    iou_threshold:   float = 0.30,
    min_mask_area_fraction: float = 0.0,
) -> Tuple[List[np.ndarray], List[float], List[List[int]]]:
    """
    Run Mask R-CNN inference on one image.

    Detections are filtered by:
      • class label == FOREGROUND_LABEL (lesion)
      • confidence score >= score_threshold
      • mask pixel fraction >= min_mask_area_fraction  (0 = disabled)

    Parameters
    ----------
    model                  : trained Mask R-CNN (eval mode is set internally)
    img_path               : path to source image
    device                 : torch.device
    score_threshold        : minimum confidence to keep a detection
    iou_threshold          : NMS IoU threshold (applied by the model internally)
    min_mask_area_fraction : minimum fraction of image pixels the mask must
                             cover; useful for filtering noise detections

    Returns
    -------
    masks  : list of (H × W) uint8 binary arrays
    scores : list of float confidence scores
    boxes  : list of [x1, y1, x2, y2] ints
    """
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return [], [], []

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_h, img_w = img_rgb.shape[:2]
    total_pixels = img_h * img_w

    img_t = TF.to_tensor(img_rgb).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(img_t)[0]

    masks, scores, boxes = [], [], []
    for i in range(len(outputs["scores"])):
        score = float(outputs["scores"][i].cpu())
        label = int(outputs["labels"][i].cpu())

        if score < score_threshold or label != FOREGROUND_LABEL:
            continue

        # Threshold soft mask at 0.5 to produce binary instance mask
        m = (outputs["masks"][i, 0].cpu().numpy() >= 0.5).astype(np.uint8)

        # Optional: discard tiny noise masks
        if min_mask_area_fraction > 0:
            if m.sum() / total_pixels < min_mask_area_fraction:
                continue

        b = outputs["boxes"][i].cpu().tolist()
        masks.append(m)
        scores.append(score)
        boxes.append([int(b[0]), int(b[1]), int(b[2]), int(b[3])])

    return masks, scores, boxes


# ─────────────────────────────────────────────────────────────────────────────
# Output saving
# ─────────────────────────────────────────────────────────────────────────────


def save_mask_and_overlay(
    img_path:     str,
    masks:        List[np.ndarray],
    scores:       List[float],
    out_mask:     str,
    out_overlay:  str,
    alpha:        float            = 0.45,
    colour_bgr:   Tuple[int, int, int] = (0, 255, 0),
    jpeg_quality: int              = 92,
):
    """
    Persist two output files for one image:

    1. Binary mask PNG  — union of all detected instance masks (pixel=255 inside lesion)
    2. Overlay JPEG     — original image with semi-transparent coloured mask
                          and per-instance confidence score annotations

    Parameters
    ----------
    img_path     : path to the source image
    masks        : list of (H × W) uint8 binary instance masks
    scores       : confidence score for each mask
    out_mask     : destination path for the combined binary mask PNG
    out_overlay  : destination path for the overlay JPEG
    alpha        : blend weight for the mask colour layer  (0=invisible, 1=opaque)
    colour_bgr   : BGR tuple for the mask highlight colour
    jpeg_quality : JPEG quality for the overlay  (1–100)
    """
    img_bgr = cv2.imread(str(img_path))
    img_h, img_w = img_bgr.shape[:2]

    # ── Combined binary mask ──────────────────────────────────────────────────
    combined = np.zeros((img_h, img_w), dtype=np.uint8)
    for m in masks:
        combined = np.maximum(combined, m)
    cv2.imwrite(out_mask, combined * 255)

    # ── Colour overlay ────────────────────────────────────────────────────────
    overlay     = img_bgr.copy()
    colour_layer = np.zeros_like(img_bgr)
    colour_layer[combined == 1] = colour_bgr
    overlay = cv2.addWeighted(overlay, 1.0, colour_layer, alpha, 0)

    # Annotate each instance with its score
    for m, score in zip(masks, scores):
        rows = np.any(m, axis=1)
        cols = np.any(m, axis=0)
        if not rows.any() or not cols.any():
            continue
        y_top  = int(np.where(rows)[0][0])
        x_left = int(np.where(cols)[0][0])
        cv2.putText(
            overlay,
            f"{score:.2f}",
            (x_left, max(y_top - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),   # yellow text
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(out_overlay, overlay, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Predict
# ─────────────────────────────────────────────────────────────────────────────


def stage_predict(
    logger:              CustomSizeDayRotatingFileHandler,
    seg_cfg,
    main_cfg,
    checkpoint_override: Optional[str] = None,
):
    """
    Run inference over all rows in the merged metadata CSV
    (original images and augmented images alike).

    For each image:
      • saves a binary mask PNG  →  seg.output.dir/masks/<stem>_mask.png
      • saves an overlay JPEG   →  seg.output.dir/overlays/<stem>_overlay.jpg

    Appends two columns to the merged metadata CSV in-place:
      • mask_path  : path to the binary mask PNG  (empty string if image missing)
      • pred_score : mean confidence of detected instances  (0.0 if none)

    Parameters
    ----------
    seg_cfg             : ConfigParser from segmentation_config.ini
    main_cfg            : ConfigParser from project config.ini  (fallback paths)
    checkpoint_override : if given, overrides seg.model.checkpoint
    """
    logger.info("=" * 60)
    logger.info("  STAGE 3  —  Predict & save ROI masks")
    logger.info("=" * 60)

    # ── Config reads ──────────────────────────────────────────────────────────
    metadata_csv    = seg_cfg.get     ("PATHS",     "merged.metadata.csv")
    seg_out_dir     = seg_cfg.get     ("PATHS",     "seg.output.dir")
    checkpoint      = checkpoint_override or seg_cfg.get("PATHS", "seg.model.checkpoint")
    num_classes     = seg_cfg.getint  ("MODEL",     "seg.num.classes",            fallback=2)
    mask_hidden     = seg_cfg.getint  ("MODEL",     "seg.mask.predictor.hidden",  fallback=256)
    score_threshold = seg_cfg.getfloat("INFERENCE", "seg.score.threshold",        fallback=0.50)
    iou_threshold   = seg_cfg.getfloat("INFERENCE", "seg.iou.threshold",          fallback=0.30)
    min_area_frac   = seg_cfg.getfloat("INFERENCE", "seg.min.mask.area.fraction", fallback=0.001)
    jpeg_quality    = seg_cfg.getint  ("INFERENCE", "seg.overlay.jpeg.quality",   fallback=92)
    overlay_alpha   = seg_cfg.getfloat("INFERENCE", "seg.overlay.alpha",          fallback=0.45)
    colour_raw      = seg_cfg.get     ("INFERENCE", "seg.overlay.colour.bgr",     fallback="0,255,0")
    device_cfg      = seg_cfg.get     ("TRAINING",  "seg.device",                 fallback="auto")

    colour_bgr = tuple(int(c.strip()) for c in colour_raw.split(","))

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_cfg == "auto"
        else torch.device(device_cfg)
    )

    # ── Log resolved inference parameters ────────────────────────────────────
    logger.info("  ── Resolved inference parameters ─────────────────────")
    logger.info(f"    device              : {device}")
    logger.info(f"    checkpoint          : {checkpoint}")
    logger.info(f"    num_classes         : {num_classes}")
    logger.info(f"    score_threshold     : {score_threshold}")
    logger.info(f"    iou_threshold       : {iou_threshold}")
    logger.info(f"    min_mask_area_frac  : {min_area_frac}")
    logger.info(f"    overlay_alpha       : {overlay_alpha}")
    logger.info(f"    overlay_colour_bgr  : {colour_bgr}")
    logger.info(f"    jpeg_quality        : {jpeg_quality}")
    logger.info("  ────────────────────────────────────────────────────────")

    # ── Load model ────────────────────────────────────────────────────────────
    if not Path(checkpoint).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}. "
            "Run  --stage train  before predict."
        )
    model = build_maskrcnn(
        num_classes=num_classes,
        pretrained=False,
        mask_hidden=mask_hidden,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    logger.info(f"  Loaded checkpoint: {checkpoint}")

    # ── Load metadata ─────────────────────────────────────────────────────────
    df = pd.read_csv(metadata_csv)
    logger.info(f"  Total images to process: {len(df)}")

    # ── Output directories ────────────────────────────────────────────────────
    mask_dir    = Path(seg_out_dir) / "masks"
    overlay_dir = Path(seg_out_dir) / "overlays"
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    # ── Inference loop ────────────────────────────────────────────────────────
    mask_paths:  List[str]   = []
    pred_scores: List[float] = []
    n_detected = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
        img_path = str(row.get("image_path", ""))
        if not img_path or not Path(img_path).exists():
            mask_paths.append("")
            pred_scores.append(float("nan"))
            continue

        stem        = Path(img_path).stem
        out_mask    = str(mask_dir    / f"{stem}_mask.png")
        out_overlay = str(overlay_dir / f"{stem}_overlay.jpg")

        masks, scores, boxes = predict_single(
            model, img_path, device,
            score_threshold, iou_threshold, min_area_frac,
        )

        if masks:
            save_mask_and_overlay(
                img_path, masks, scores,
                out_mask, out_overlay,
                alpha=overlay_alpha,
                colour_bgr=colour_bgr,
                jpeg_quality=jpeg_quality,
            )
            mask_paths.append(out_mask)
            pred_scores.append(float(np.mean(scores)))
            n_detected += 1
        else:
            # Write blank mask so downstream steps can always find a file
            img_bgr = cv2.imread(img_path)
            h, w    = img_bgr.shape[:2]
            cv2.imwrite(out_mask, np.zeros((h, w), dtype=np.uint8))
            mask_paths.append(out_mask)
            pred_scores.append(0.0)

        # Progress log every 100 images
        if (idx + 1) % 100 == 0:
            logger.info(
                f"  Progress: {idx + 1}/{len(df)} images processed  |  "
                f"detections so far: {n_detected}"
            )

    # ── Append columns and save ───────────────────────────────────────────────
    df["mask_path"]  = mask_paths
    df["pred_score"] = pred_scores
    df.to_csv(metadata_csv, index=False)

    n_missing = sum(1 for p in mask_paths if not p)
    logger.info(f"  Lesion detected    : {n_detected} / {len(df)} images")
    logger.info(f"  Missing images     : {n_missing}")
    logger.info(f"  Masks saved        → {mask_dir}")
    logger.info(f"  Overlays saved     → {overlay_dir}")
    logger.info(f"  Updated CSV        → {metadata_csv}")
    logger.info("  STAGE 3 complete.")
