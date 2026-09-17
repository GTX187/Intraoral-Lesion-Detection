"""
train_resnet.py
----------------
Fine-tunes a ResNet classifier on the annotated SMART intraoral dataset
for whole-image lesion classification: Normal / Variation / OPMD.

This is the "direct image input" pipeline — it takes the exact same
image_path/patient_id/label rows as the UNet pipeline (same csv_path, via
the same get_dataset_path()), but classifies the whole image into one of
three classes instead of segmenting a region within it.

Pipeline
--------
1. build_data_loaders()   →  patient-wise train / val / test split
2. build_classifier()     →  ImageNet ResNet + randomly-initialised fc head
3. Training loop          →  SGD with separate backbone / head LRs
4. Validation loop        →  accuracy, macro-F1, per-class precision/recall/F1,
                              confusion matrix
5. Checkpointing          →  best val macro-F1 model saved; resume supported

Targets
-------
    0 = Normal
    1 = Variation
    2 = OPMD
(see CLASS_LABEL_MAP in resnet_builder.py)

Usage
-----
    python train_resnet.py               # config/config.ini
    python train_resnet.py -p kaggle     # config/kaggle_config.ini
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.common.intraoral_logger import initialize_logger
from utils.load_configuration import load_config
from src.segmentation.unet.unet_config import UNetConfig
from src.segmentation.unet.train_unet import get_dataset_path  # reuse dataset assembly
from src.segmentation.unet.unet_builder import _resolve_device, save_checkpoint, load_checkpoint
from src.classification.resnet.resnet_config import ResNetConfig
from src.classification.resnet.resnet_builder import (
    CLASS_LABEL_MAP,
    INDEX_LABEL_MAP,
    NUM_CLASSES,
    build_classifier,
    build_data_loaders,
)

# ══════════════════════════════════════════════════════════════════════════════
# Loss functions
# ══════════════════════════════════════════════════════════════════════════════


def _compute_class_weights(class_counts: Dict[str, int], device: torch.device) -> torch.Tensor:
    """
    Inverse-frequency class weights, ordered by CLASS_LABEL_MAP index
    (0=normal, 1=variation, 2=opmd), computed from the TRAIN split's own
    class counts (not val/test — those should reflect the true, imbalanced
    population).

    weight_c = total_train_rows / (num_classes * count_c)

    This is the standard inverse-frequency scheme: a class with half as
    many examples gets roughly double the loss weight, so rare OPMD/
    Variation rows aren't drowned out by the much larger Normal class.
    """
    total = sum(class_counts.values())
    weights = torch.zeros(NUM_CLASSES, dtype=torch.float32)
    for label, idx in CLASS_LABEL_MAP.items():
        count = max(class_counts.get(label, 0), 1)  # avoid div-by-zero
        weights[idx] = total / (NUM_CLASSES * count)
    return weights.to(device)


def _focal_loss_multiclass(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Multiclass focal loss (Lin et al., 2017), using class_weights as the
    per-class alpha term.

    FL(p_t) = -alpha_c * (1 - p_t)^gamma * log(p_t)

    gamma down-weights already-easy, correctly-classified examples so
    gradient focuses on hard/rare cases — same rationale as the UNet
    pipeline's focal_dice option, applied here to whole-image classification
    instead of per-pixel segmentation.
    """
    log_probs = F.log_softmax(logits, dim=1)
    probs = log_probs.exp()
    p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    log_p_t = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    alpha_t = class_weights[targets]
    focal = -alpha_t * (1 - p_t) ** gamma * log_p_t
    return focal.mean()


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_function: str,
    class_weights: Optional[torch.Tensor] = None,
    focal_gamma: float = 2.0,
) -> torch.Tensor:
    """
    Dispatch to the configured loss function. logits: [B, num_classes],
    targets: [B] int64 class indices — standard CrossEntropyLoss shapes,
    no squeeze/unsqueeze gymnastics needed since this isn't per-pixel.
    """
    if loss_function == "ce":
        return F.cross_entropy(logits, targets)
    elif loss_function == "weighted_ce":
        return F.cross_entropy(logits, targets, weight=class_weights)
    elif loss_function == "focal":
        return _focal_loss_multiclass(logits, targets, class_weights, focal_gamma)
    else:
        raise ValueError(f"Unknown loss_function '{loss_function}'")


# ══════════════════════════════════════════════════════════════════════════════
# Metrics — confusion matrix, per-class precision/recall/F1, macro-F1
# ══════════════════════════════════════════════════════════════════════════════


@torch.no_grad()
def compute_confusion_matrix(preds: torch.Tensor, targets: torch.Tensor) -> np.ndarray:
    """
    NUM_CLASSES x NUM_CLASSES matrix, rows = true class, cols = predicted
    class. Implemented with a manual bincount rather than sklearn, since
    scikit-learn isn't in requirements.txt for this pipeline.
    """
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    idx = (targets * NUM_CLASSES + preds).cpu().numpy()
    counts = np.bincount(idx, minlength=NUM_CLASSES * NUM_CLASSES)
    return counts.reshape(NUM_CLASSES, NUM_CLASSES) + cm


def metrics_from_confusion_matrix(cm: np.ndarray) -> dict:
    """
    Per-class precision/recall/F1 + macro-F1 + accuracy from a confusion
    matrix. Precision/recall use a small epsilon to avoid 0/0 for classes
    with zero predictions or zero support in a given batch/epoch.
    """
    eps = 1e-9
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)  # true count per class
    predicted = cm.sum(axis=0).astype(np.float64)  # predicted count per class

    precision = tp / (predicted + eps)
    recall = tp / (support + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    per_class = {}
    for label, idx in CLASS_LABEL_MAP.items():
        per_class[label] = {
            "precision": round(float(precision[idx]), 4),
            "recall": round(float(recall[idx]), 4),
            "f1": round(float(f1[idx]), 4),
            "support": int(support[idx]),
        }

    accuracy = float(tp.sum() / max(cm.sum(), 1))
    macro_f1 = float(f1.mean())

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
    }


# ══════════════════════════════════════════════════════════════════════════════
# One training epoch
# ══════════════════════════════════════════════════════════════════════════════


def train_one_epoch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
    logger: logging.Logger,
    loss_function: str,
    class_weights: torch.Tensor,
    focal_gamma: float,
    gradient_clip: float,
    log_every: int,
) -> dict:
    model.train()
    total_loss = 0.0
    n_batches = 0
    t_start = time.time()

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = compute_loss(logits, labels, loss_function, class_weights, focal_gamma)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if (batch_idx + 1) % log_every == 0 or (batch_idx + 1) == len(loader):
            logger.info(
                "Epoch %d [%d/%d]  loss=%.4f",
                epoch, batch_idx + 1, len(loader), total_loss / n_batches,
            )

    elapsed = time.time() - t_start
    avg_loss = total_loss / max(n_batches, 1)
    logger.info("Epoch %d  train  loss=%.4f  time=%.1fs", epoch, avg_loss, elapsed)
    return {"train_loss": round(avg_loss, 4)}


# ══════════════════════════════════════════════════════════════════════════════
# Validation epoch
# ══════════════════════════════════════════════════════════════════════════════


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
    logger: logging.Logger,
    loss_function: str,
    class_weights: torch.Tensor,
    focal_gamma: float,
) -> dict:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = compute_loss(logits, labels, loss_function, class_weights, focal_gamma)
        total_loss += loss.item()
        n_batches += 1

        preds = logits.argmax(dim=1)
        cm += compute_confusion_matrix(preds, labels)

    if n_batches == 0:
        logger.warning("Validation loader was empty — no metrics computed.")
        return {}

    m = metrics_from_confusion_matrix(cm)
    avg_loss = total_loss / n_batches

    logger.info(
        "Epoch %d  val  loss=%.4f  acc=%.4f  macro_f1=%.4f",
        epoch, avg_loss, m["accuracy"], m["macro_f1"],
    )
    for label, stats in m["per_class"].items():
        logger.info(
            "    %-10s  precision=%.4f  recall=%.4f  f1=%.4f  support=%d",
            label, stats["precision"], stats["recall"], stats["f1"], stats["support"],
        )

    return {
        "val_loss": round(avg_loss, 4),
        "val_accuracy": m["accuracy"],
        "val_macro_f1": m["macro_f1"],
        "val_per_class": m["per_class"],
        "val_confusion_matrix": cm.tolist(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main training loop
# ══════════════════════════════════════════════════════════════════════════════


def train(logger: logging.Logger, cfg: ResNetConfig, csv_path: str) -> None:
    device = _resolve_device(logger, cfg.device)
    logger.info("=" * 60)
    logger.info("  ResNet Lesion Classifier Fine-Tuning  |  device=%s", device)
    logger.info("=" * 60)

    # ── Data ──────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, class_counts = build_data_loaders(
        logger=logger,
        csv_path=csv_path,
        val_split=cfg.val_split,
        test_split=cfg.test_split,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
        image_size=cfg.image_size,
    )
    class_weights = _compute_class_weights(class_counts, device)
    logger.info(
        "  Class weights (normal/variation/opmd): %s",
        [round(w, 3) for w in class_weights.tolist()],
    )

    # ── Model ─────────────────────────────────────────────────────────
    model = build_classifier(
        logger=logger,
        num_classes=cfg.num_classes,
        device=str(device),
        pretrained_backbone=cfg.pretrained,
        encoder_name=cfg.backbone,
    )

    # ── Optimiser — separate backbone (lower LR) / head (higher LR) ───
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("fc"):
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = optim.SGD(
        [
            {"params": backbone_params, "lr": cfg.backbone_lr},
            {"params": head_params, "lr": cfg.head_lr},
        ],
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )

    # ── LR Scheduler ──────────────────────────────────────────────────
    if cfg.lr_scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr
        )
    elif cfg.lr_scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.lr_patience, gamma=cfg.lr_factor
        )
    else:  # plateau
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=cfg.lr_factor,
            patience=cfg.lr_patience, min_lr=cfg.min_lr,
        )

    # ── Checkpoint setup ──────────────────────────────────────────────
    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = checkpoint_dir / "best.pth"
    last_ckpt_path = checkpoint_dir / "last.pth"

    start_epoch = 1
    best_macro_f1 = 0.0
    history = []

    resume_path = best_ckpt_path if best_ckpt_path.exists() else (
        last_ckpt_path if last_ckpt_path.exists() else None
    )
    if resume_path:
        start_epoch, prev_metrics = load_checkpoint(
            logger=logger, model=model, optimizer=optimizer,
            path=resume_path, device=str(device),
        )
        best_macro_f1 = prev_metrics.get("val_macro_f1", 0.0)
        start_epoch += 1
        logger.info(
            "Resuming from epoch %d  (best val_macro_f1=%.4f)",
            start_epoch, best_macro_f1,
        )
    else:
        logger.info("No checkpoint found — starting fresh.")

    # ── Epoch loop ────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.epochs + 1):
        logger.info("\n--- Epoch %d / %d ---", epoch, cfg.epochs)

        train_metrics = train_one_epoch(
            model=model, optimizer=optimizer, loader=train_loader,
            device=device, epoch=epoch, logger=logger,
            loss_function=cfg.loss_function, class_weights=class_weights,
            focal_gamma=cfg.focal_gamma, gradient_clip=cfg.gradient_clip,
            log_every=cfg.log_every,
        )

        if cfg.lr_scheduler in ("cosine", "step"):
            scheduler.step()
        current_lrs = [pg["lr"] for pg in optimizer.param_groups]
        logger.info(
            "  LR after step: backbone=%.2e  head=%.2e",
            current_lrs[0], current_lrs[1],
        )

        val_metrics = {}
        if epoch % cfg.val_every == 0 or epoch == cfg.epochs:
            val_metrics = validate_one_epoch(
                model=model, loader=val_loader, device=device,
                epoch=epoch, logger=logger, loss_function=cfg.loss_function,
                class_weights=class_weights, focal_gamma=cfg.focal_gamma,
            )
            if cfg.lr_scheduler == "plateau":
                scheduler.step(val_metrics.get("val_macro_f1", 0.0))

        all_metrics = {**train_metrics, **val_metrics}

        current_macro_f1 = val_metrics.get("val_macro_f1", best_macro_f1)
        if val_metrics and current_macro_f1 > best_macro_f1:
            best_macro_f1 = current_macro_f1
            save_checkpoint(logger, model, optimizer, epoch, all_metrics, best_ckpt_path)
            logger.info("  ★ New best val_macro_f1=%.4f — saved to %s", best_macro_f1, best_ckpt_path)

        save_checkpoint(logger, model, optimizer, epoch, all_metrics, last_ckpt_path)
        history.append({"epoch": epoch, **{k: v for k, v in all_metrics.items() if k != "val_confusion_matrix"}})

    # ── Save final model ──────────────────────────────────────────────
    model_dir = Path(cfg.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    final_model_path = model_dir / "best_resnet_classifier.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "best_macro_f1": best_macro_f1,
            "class_label_map": CLASS_LABEL_MAP,
            "config": cfg.__dict__ if hasattr(cfg, "__dict__") else str(cfg),
        },
        final_model_path,
    )
    logger.info("✅ Best model saved to: %s", final_model_path)

    # ── Held-out test-set evaluation ─────────────────────────────────
    if test_loader is not None and len(test_loader) > 0:
        logger.info("Evaluating on held-out test set...")
        test_metrics = validate_one_epoch(
            model=model, loader=test_loader, device=device, epoch=epoch,
            logger=logger, loss_function=cfg.loss_function,
            class_weights=class_weights, focal_gamma=cfg.focal_gamma,
        )
        test_metrics = {k.replace("val_", "test_", 1): v for k, v in test_metrics.items()}
        logger.info("Test set metrics: %s", {k: v for k, v in test_metrics.items() if k != "test_confusion_matrix"})
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "test_metrics.json", "w") as f:
            json.dump(test_metrics, f, indent=2)

    # ── Save training history ─────────────────────────────────────────
    history_path = checkpoint_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Training history saved → %s", history_path)
    logger.info("Training complete. Best val_macro_f1=%.4f", best_macro_f1)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════


def get_configpath():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--profile")
    args = parser.parse_args()
    config_path = "config/config.ini"
    if args.profile and args.profile.lower() == "kaggle":
        config_path = "config/kaggle_config.ini"
    return config_path


if __name__ == "__main__":
    config_path = get_configpath()
    print(f"config_path: {config_path}")
    config = load_config(config_path)
    logger = initialize_logger(config=config)

    # Reuse the exact same assembled CSV the UNet pipeline trains on —
    # same image_path/patient_id/label rows, since this is the
    # "direct image input" classification pipeline.
    unet_ini = load_config(config.get("SEGMENT-UNET", "unet.config"))
    unet_cfg = UNetConfig(unet_ini)
    csv_path = get_dataset_path(logger=logger, config=config, cfg=unet_cfg)

    cls_ini = load_config(config.get("CLASSIFY-RESNET", "resnet.config"))
    cfg = ResNetConfig(cls_ini)
    train(logger=logger, cfg=cfg, csv_path=csv_path)
