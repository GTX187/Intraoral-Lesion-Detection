"""
seg_trainer.py
──────────────
Training module for the Mask R-CNN ROI Segmentation Pipeline.

Responsibilities:
  • Build torchvision Mask R-CNN (ResNet-50 + FPN) with configurable heads
  • Fine-tune with AdamW optimiser + StepLR scheduler
  • Per-step loss logging (configurable frequency)
  • Per-epoch summary logging on a configurable interval
  • Save best checkpoint based on validation loss
  • stage_train() orchestrator

Public API
----------
  stage_train(logger, seg_cfg, train_recs, val_recs)
  build_maskrcnn(num_classes, pretrained, mask_hidden) -> nn.Module
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.utils.data as data_utils
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn,
    MaskRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator

from seg_dataset import OralLesionDataset, collate_fn
from utils.log_handler import CustomSizeDayRotatingFileHandler


# ─────────────────────────────────────────────────────────────────────────────
# Model construction
# ─────────────────────────────────────────────────────────────────────────────


def build_maskrcnn(
    num_classes:   int  = 2,
    pretrained:    bool = True,
    mask_hidden:   int  = 256,
    anchor_sizes:  Optional[List[int]]   = None,
    anchor_ratios: Optional[List[float]] = None,
) -> nn.Module:
    """
    Instantiate a torchvision Mask R-CNN with a ResNet-50 + FPN backbone.

    The box predictor and mask predictor heads are replaced to match
    the configured number of classes (background + N lesion types).

    Parameters
    ----------
    num_classes   : total classes INCLUDING background  (e.g. 2)
    pretrained    : initialise backbone with COCO weights
    mask_hidden   : hidden channels in the mask prediction head (default 256)
    anchor_sizes  : RPN anchor sizes per FPN level
    anchor_ratios : RPN anchor aspect ratios
    """
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model   = maskrcnn_resnet50_fpn(weights=weights)

    # Optionally replace the RPN anchor generator
    if anchor_sizes is not None and anchor_ratios is not None:
        # torchvision expects one tuple of sizes per FPN level (5 levels)
        sizes  = tuple((s,) for s in anchor_sizes)
        ratios = (tuple(anchor_ratios),) * len(sizes)
        model.rpn.anchor_generator = AnchorGenerator(
            sizes=sizes, aspect_ratios=ratios
        )

    # Replace box predictor head
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)

    # Replace mask predictor head
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, mask_hidden, num_classes
    )

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────


def _compute_val_loss(
    model:  nn.Module,
    loader: data_utils.DataLoader,
    device: torch.device,
) -> float:
    """
    Compute mean total loss on the validation loader.
    The model is kept in train() mode so loss_dict is populated,
    but gradients are disabled.
    """
    model.train()
    total, count = 0.0, 0
    with torch.no_grad():
        for images, targets in loader:
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            total += sum(loss_dict.values()).item()
            count += 1
    return total / max(count, 1)


def _train_one_epoch(
    model:        nn.Module,
    optimiser:    torch.optim.Optimizer,
    loader:       data_utils.DataLoader,
    device:       torch.device,
    logger:       CustomSizeDayRotatingFileHandler,
    epoch:        int,
    step_log_freq: int = 20,
    grad_clip:    float = 0.0,
) -> dict:
    """
    Run one training epoch.

    Returns a dict of mean per-loss-term values for the epoch:
        {loss_classifier, loss_box_reg, loss_mask, loss_objectness,
         loss_rpn_box_reg, total}
    """
    model.train()
    accum: dict = {}
    step_count = 0

    for step, (images, targets) in enumerate(loader, start=1):
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses    = sum(loss_dict.values())

        optimiser.zero_grad()
        losses.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimiser.step()

        # Accumulate per-term losses
        for k, v in loss_dict.items():
            accum[k] = accum.get(k, 0.0) + v.item()
        accum["total"] = accum.get("total", 0.0) + losses.item()
        step_count += 1

        if step % step_log_freq == 0 or step == len(loader):
            logger.info(
                f"    Epoch {epoch:3d} | step {step:4d}/{len(loader)} | "
                f"loss={losses.item():.4f}  "
                f"[cls={loss_dict.get('loss_classifier', torch.tensor(0)).item():.4f}  "
                f"box={loss_dict.get('loss_box_reg',     torch.tensor(0)).item():.4f}  "
                f"mask={loss_dict.get('loss_mask',        torch.tensor(0)).item():.4f}  "
                f"rpn_obj={loss_dict.get('loss_objectness', torch.tensor(0)).item():.4f}  "
                f"rpn_box={loss_dict.get('loss_rpn_box_reg', torch.tensor(0)).item():.4f}]"
            )

    # Compute means
    return {k: v / max(step_count, 1) for k, v in accum.items()}


def _log_epoch_summary(
    logger:       CustomSizeDayRotatingFileHandler,
    epoch:        int,
    total_epochs: int,
    train_losses: dict,
    val_loss:     float,
    elapsed:      float,
    lr:           float,
    best_val:     float,
    is_best:      bool,
):
    """
    Emit a formatted epoch summary block.
    Called only on epochs where epoch % log_interval == 0 (or on the last epoch).
    """
    sep = "─" * 60
    logger.info(sep)
    logger.info(
        f"  EPOCH SUMMARY  {epoch:3d} / {total_epochs}"
        f"  |  time: {elapsed:6.1f}s  |  lr: {lr:.2e}"
    )
    logger.info(
        f"    Train  →  total={train_losses.get('total', 0):.4f}  "
        f"cls={train_losses.get('loss_classifier', 0):.4f}  "
        f"box_reg={train_losses.get('loss_box_reg', 0):.4f}  "
        f"mask={train_losses.get('loss_mask', 0):.4f}  "
        f"rpn_obj={train_losses.get('loss_objectness', 0):.4f}  "
        f"rpn_box={train_losses.get('loss_rpn_box_reg', 0):.4f}"
    )
    logger.info(
        f"    Val    →  total={val_loss:.4f}  "
        f"(best so far: {best_val:.4f})"
    )
    if is_best:
        logger.info("    ✓  New best checkpoint saved.")
    logger.info(sep)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Train
# ─────────────────────────────────────────────────────────────────────────────


def stage_train(
    logger:     CustomSizeDayRotatingFileHandler,
    seg_cfg,
    train_recs: List[dict],
    val_recs:   List[dict],
) -> nn.Module:
    """
    Fine-tune Mask R-CNN on the prepared train / val splits.

    Logging behaviour
    -----------------
    • Per-step loss is logged every seg.step.log.freq steps (default 20).
    • A full epoch summary (all loss terms + val loss) is logged every
      seg.log.interval epochs (default 10) AND always on the final epoch.
      seg.log.interval is fully configurable in segmentation_config.ini.

    Parameters
    ----------
    seg_cfg    : ConfigParser from segmentation_config.ini
    train_recs : list of record dicts from stage_prepare
    val_recs   : list of record dicts from stage_prepare

    Returns
    -------
    The trained nn.Module (weights of best checkpoint already loaded).
    """
    logger.info("=" * 60)
    logger.info("  STAGE 2  —  Train Mask R-CNN")
    logger.info("=" * 60)

    # ── Read training config ──────────────────────────────────────────────────
    checkpoint    = seg_cfg.get     ("PATHS",    "seg.model.checkpoint")
    num_classes   = seg_cfg.getint  ("MODEL",    "seg.num.classes",           fallback=2)
    pretrained    = seg_cfg.getboolean("MODEL",  "seg.pretrained",            fallback=True)
    mask_hidden   = seg_cfg.getint  ("MODEL",    "seg.mask.predictor.hidden", fallback=256)
    anchor_sizes_raw  = seg_cfg.get ("MODEL",    "seg.anchor.sizes",
                                     fallback="32,64,128,256,512")
    anchor_ratios_raw = seg_cfg.get ("MODEL",    "seg.anchor.ratios",
                                     fallback="0.5,1.0,2.0")

    epochs        = seg_cfg.getint  ("TRAINING", "seg.train.epochs",   fallback=30)
    batch_size    = seg_cfg.getint  ("TRAINING", "seg.batch.size",      fallback=2)
    lr            = seg_cfg.getfloat("TRAINING", "seg.learning.rate",   fallback=5e-4)
    weight_decay  = seg_cfg.getfloat("TRAINING", "seg.weight.decay",    fallback=1e-4)
    step_size_cfg = seg_cfg.getint  ("TRAINING", "seg.lr.step.size",    fallback=0)
    lr_gamma      = seg_cfg.getfloat("TRAINING", "seg.lr.gamma",        fallback=0.5)
    warmup_epochs = seg_cfg.getint  ("TRAINING", "seg.warmup.epochs",   fallback=2)
    grad_clip     = seg_cfg.getfloat("TRAINING", "seg.grad.clip",       fallback=1.0)
    log_interval  = seg_cfg.getint  ("TRAINING", "seg.log.interval",    fallback=10)
    step_log_freq = seg_cfg.getint  ("TRAINING", "seg.step.log.freq",   fallback=20)
    num_workers   = seg_cfg.getint  ("TRAINING", "seg.num.workers",     fallback=2)
    device_cfg    = seg_cfg.get     ("TRAINING", "seg.device",          fallback="auto")

    anchor_sizes  = [int(s.strip())   for s in anchor_sizes_raw.split(",")]
    anchor_ratios = [float(r.strip()) for r in anchor_ratios_raw.split(",")]
    step_size     = step_size_cfg if step_size_cfg > 0 else max(1, epochs // 3)

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_cfg == "auto"
        else torch.device(device_cfg)
    )

    # ── Log all resolved training parameters ─────────────────────────────────
    logger.info("  ── Resolved training parameters ──────────────────────")
    logger.info(f"    device          : {device}")
    logger.info(f"    num_classes     : {num_classes}")
    logger.info(f"    pretrained      : {pretrained}")
    logger.info(f"    mask_hidden     : {mask_hidden}")
    logger.info(f"    anchor_sizes    : {anchor_sizes}")
    logger.info(f"    anchor_ratios   : {anchor_ratios}")
    logger.info(f"    epochs          : {epochs}")
    logger.info(f"    batch_size      : {batch_size}")
    logger.info(f"    learning_rate   : {lr}")
    logger.info(f"    weight_decay    : {weight_decay}")
    logger.info(f"    lr_step_size    : {step_size}  (gamma={lr_gamma})")
    logger.info(f"    warmup_epochs   : {warmup_epochs}")
    logger.info(f"    grad_clip       : {grad_clip}")
    logger.info(f"    log_interval    : every {log_interval} epoch(s)")
    logger.info(f"    step_log_freq   : every {step_log_freq} step(s)")
    logger.info(f"    checkpoint      : {checkpoint}")
    logger.info(f"    train images    : {len(train_recs)}")
    logger.info(f"    val   images    : {len(val_recs)}")
    logger.info("  ────────────────────────────────────────────────────────")

    # ── Build model ───────────────────────────────────────────────────────────
    model = build_maskrcnn(
        num_classes=num_classes,
        pretrained=pretrained,
        mask_hidden=mask_hidden,
        anchor_sizes=anchor_sizes,
        anchor_ratios=anchor_ratios,
    ).to(device)

    # Resume from checkpoint if present
    if Path(checkpoint).exists():
        logger.info(f"  Resuming from checkpoint: {checkpoint}")
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader = data_utils.DataLoader(
        OralLesionDataset(train_recs),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    val_loader = data_utils.DataLoader(
        OralLesionDataset(val_recs),
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    # ── Optimiser & scheduler ─────────────────────────────────────────────────
    params    = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimiser, step_size=step_size, gamma=lr_gamma
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    os.makedirs(Path(checkpoint).parent, exist_ok=True)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Warmup: hold LR at lr/10 for the first warmup_epochs
        if epoch <= warmup_epochs:
            for pg in optimiser.param_groups:
                pg["lr"] = lr * (epoch / max(warmup_epochs, 1))

        train_losses = _train_one_epoch(
            model, optimiser, train_loader, device,
            logger, epoch, step_log_freq, grad_clip
        )

        # Always compute val loss (cheap with torch.no_grad)
        val_loss = _compute_val_loss(model, val_loader, device)
        elapsed  = time.time() - t0

        if epoch > warmup_epochs:
            scheduler.step()

        current_lr = optimiser.param_groups[0]["lr"]

        # Save best checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint)

        # ── Epoch-interval logging ────────────────────────────────────────────
        # Always log on the configured interval AND on the very last epoch
        if epoch % log_interval == 0 or epoch == epochs:
            _log_epoch_summary(
                logger, epoch, epochs,
                train_losses, val_loss, elapsed,
                current_lr, best_val_loss, is_best,
            )
        else:
            # Lightweight single-line log on non-summary epochs
            logger.info(
                f"  Epoch {epoch:3d}/{epochs} | "
                f"train={train_losses.get('total', 0):.4f}  "
                f"val={val_loss:.4f}  "
                f"lr={current_lr:.2e}  "
                f"({elapsed:.0f}s)"
                + ("  ✓ best" if is_best else "")
            )

    logger.info(
        f"  Training complete. Best val_loss = {best_val_loss:.4f}  "
        f"checkpoint → {checkpoint}"
    )
    logger.info("  STAGE 2 complete.")

    # Load best weights before returning
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model
