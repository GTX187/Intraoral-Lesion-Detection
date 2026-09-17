"""
resnet_builder.py
------------------
Dataset, model, and DataLoader construction for the direct-image ResNet
lesion classifier (Normal / Variation / OPMD).

Deliberately reuses — rather than re-implements — the generic pieces that
already exist in the UNet pipeline:
    _resolve_device        (device string -> torch.device, with CUDA fallback)
    _normalize_patient_id   (case/whitespace-insensitive patient key)
    save_checkpoint / load_checkpoint
This is the same fix pattern flagged in the pipeline observation log for
duplicated patient-id parsing logic — a shared helper here instead of a
second copy of the same code.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torchvision import transforms

from src.segmentation.unet.unet_builder import (
    _resolve_device,
    _normalize_patient_id,
    save_checkpoint,
    load_checkpoint,
)

# ══════════════════════════════════════════════════════════════════════════════
# Fixed label ↔ index mapping
# ══════════════════════════════════════════════════════════════════════════════

# Index order is load-bearing — a saved checkpoint's final Linear layer is
# ordered by this map. Do not reorder without retraining from scratch.
CLASS_LABEL_MAP = {"normal": 0, "variation": 1, "opmd": 2}
INDEX_LABEL_MAP = {v: k for k, v in CLASS_LABEL_MAP.items()}
NUM_CLASSES = len(CLASS_LABEL_MAP)

# torchvision ResNets pretrained on ImageNet all share this normalisation —
# matches the SMP encoder preprocessing used in the UNet pipeline.
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_BACKBONE_CTORS = {
    "resnet18": torchvision.models.resnet18,
    "resnet34": torchvision.models.resnet34,
    "resnet50": torchvision.models.resnet50,
    "resnet101": torchvision.models.resnet101,
}
_BACKBONE_WEIGHTS = {
    "resnet18": "IMAGENET1K_V1",
    "resnet34": "IMAGENET1K_V1",
    "resnet50": "IMAGENET1K_V2",
    "resnet101": "IMAGENET1K_V2",
}


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════


class LesionClassificationDataset(torch.utils.data.Dataset):
    """
    Direct-image classification dataset: reads image_path, returns
    (image_tensor, class_index). Takes the same rows DataFrame shape as
    CocoSegDataset (image_path, patient_id, label, coco_file, ...) but only
    ever touches image_path + label — no mask/annotation is loaded here,
    since this pipeline classifies the whole image rather than segmenting
    a region within it.
    """

    def __init__(
        self,
        rows: pd.DataFrame,
        image_size: int = 512,
        train: bool = False,
    ):
        self.rows = rows.reset_index(drop=True)
        self.image_size = image_size
        self.train = train

        # No horizontal/vertical flips — same rationale as the augmentation
        # pipeline: left/right and top/bottom are anatomically meaningful in
        # an intraoral photo. Only mild rotation + subtle photometric jitter
        # on the train split; val/test get a plain resize + normalize.
        if train:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.RandomRotation(degrees=12),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
                ]
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.rows.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        image_t = self.transform(image)
        label_idx = CLASS_LABEL_MAP[str(row["label"]).strip().lower()]
        return image_t, label_idx


def _collate_fn(batch):
    images, labels = zip(*batch)
    return torch.stack(images, dim=0), torch.tensor(labels, dtype=torch.long)


# ══════════════════════════════════════════════════════════════════════════════
# Model builder
# ══════════════════════════════════════════════════════════════════════════════


def build_classifier(
    logger: logging.Logger,
    num_classes: int,
    device: str,
    pretrained_backbone: bool,
    encoder_name: str = "resnet34",
) -> nn.Module:
    """
    Builds a torchvision ResNet with its final fc layer replaced by a
    Linear(in_features, num_classes) head. No sigmoid/softmax baked in —
    raw logits out, exactly like the UNet's activation=None convention, so
    the same "apply the activation only where the loss/metric needs it"
    rule holds here too (CrossEntropyLoss expects raw logits).
    """
    if encoder_name not in _BACKBONE_CTORS:
        raise ValueError(
            f"Unsupported backbone '{encoder_name}'. "
            f"Supported: {sorted(_BACKBONE_CTORS)}"
        )

    logger.info(
        "Building ResNet classifier  encoder=%s  num_classes=%d  pretrained=%s",
        encoder_name,
        num_classes,
        pretrained_backbone,
    )

    ctor = _BACKBONE_CTORS[encoder_name]
    weights = _BACKBONE_WEIGHTS[encoder_name] if pretrained_backbone else None
    model = ctor(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    dev = _resolve_device(logger, device)
    model.to(dev)
    logger.info("Lesion ResNet classifier built on %s", dev)
    return model


# ══════════════════════════════════════════════════════════════════════════════
# DataLoader builder — same patient-wise split pattern as the UNet pipeline
# ══════════════════════════════════════════════════════════════════════════════


def build_data_loaders(
    logger: logging.Logger,
    csv_path: str,
    val_split: float = 0.15,
    test_split: float = 0.15,
    batch_size: int = 16,
    num_workers: int = 2,
    seed: int = 42,
    image_size: int = 512,
    path_rewrite: Optional[dict] = None,
) -> Tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    Optional[torch.utils.data.DataLoader],
    dict,
]:
    """
    Reads the same merged CSV the UNet pipeline uses (image_path, patient_id,
    label, source, ...), keeps rows whose label is one of Normal/Variation/
    OPMD and whose image_path exists on disk, and returns
    (train_loader, val_loader, test_loader, class_counts).

    Patient-wise split: identical logic to unet_builder.build_data_loaders —
    normalise patient_id (case/whitespace) before grouping so a patient
    can't leak across splits via a spelling difference between the original
    and augmented CSVs, then a plain random.Random(seed).shuffle() of the
    normalised patient-id list, sliced into disjoint train/val/test sets.
    """
    logger.info("Building ResNet classifier data loaders from: %s", csv_path)
    df = pd.read_csv(csv_path, dtype=str)
    logger.info("  Total CSV rows: %d", len(df))

    df["_label_key"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["_label_key"].isin(CLASS_LABEL_MAP.keys())].copy()
    logger.info("  Rows with a recognised label (Normal/Variation/OPMD): %d", len(df))

    rewrite = path_rewrite or {}

    def _rw(p: str) -> Path:
        for old, new in rewrite.items():
            if str(p).startswith(old):
                return Path(new + str(p)[len(old):])
        return Path(p)

    exists_mask = df["image_path"].apply(lambda p: _rw(str(p)).exists())
    df = df[exists_mask].reset_index(drop=True)
    logger.info("  Rows with image_path on disk: %d", len(df))

    if len(df) == 0:
        raise RuntimeError(
            "No labelled images found on disk. Check csv_path and path_rewrite."
        )

    # ── Patient-wise split (train / val / test, no overlap) ────────────
    if "patient_id" not in df.columns:
        raise RuntimeError("Column 'patient_id' is required for patient-wise split.")

    df["_patient_key"] = df["patient_id"].map(_normalize_patient_id)
    patient_ids = sorted(df["_patient_key"].dropna().unique().tolist())
    min_patients_needed = 3 if test_split > 0 else 2
    if len(patient_ids) < min_patients_needed:
        raise RuntimeError(
            f"Need ≥{min_patients_needed} unique patients for the requested "
            f"split, got {len(patient_ids)}."
        )

    rng = random.Random(seed)
    rng.shuffle(patient_ids)

    n_total = len(patient_ids)
    n_val = max(1, round(n_total * val_split))
    n_test = max(1, round(n_total * test_split)) if test_split > 0 else 0
    if n_val + n_test >= n_total:
        raise RuntimeError(
            f"val_split ({val_split}) + test_split ({test_split}) leave no "
            f"patients for training out of {n_total} total patients."
        )

    val_pids = set(patient_ids[:n_val])
    test_pids = set(patient_ids[n_val : n_val + n_test])
    train_pids = set(patient_ids[n_val + n_test :])

    assert train_pids.isdisjoint(val_pids), "train/val patient overlap detected"
    assert train_pids.isdisjoint(test_pids), "train/test patient overlap detected"
    assert val_pids.isdisjoint(test_pids), "val/test patient overlap detected"

    train_df = df[df["_patient_key"].isin(train_pids)].reset_index(drop=True)
    val_df = df[df["_patient_key"].isin(val_pids)].reset_index(drop=True)
    test_df = (
        df[df["_patient_key"].isin(test_pids)].reset_index(drop=True)
        if test_pids
        else df.iloc[0:0].copy()
    )

    # Per-class row counts on the TRAIN split only — this is what a
    # weighted-loss / focal-loss setup should be tuned against, since val
    # and test are meant to reflect the true (imbalanced) population.
    class_counts = {
        label: int((train_df["_label_key"] == label).sum())
        for label in CLASS_LABEL_MAP
    }
    logger.info(
        "  Patient split → train=%d patients (%d rows)  val=%d patients (%d rows)"
        "  test=%d patients (%d rows)",
        len(train_pids), len(train_df),
        len(val_pids), len(val_df),
        len(test_pids), len(test_df),
    )
    logger.info("  Train-split class counts: %s", class_counts)

    train_ds = LesionClassificationDataset(train_df, image_size=image_size, train=True)
    val_ds = LesionClassificationDataset(val_df, image_size=image_size, train=False)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = None
    if len(test_df) > 0:
        test_ds = LesionClassificationDataset(test_df, image_size=image_size, train=False)
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=_collate_fn,
            pin_memory=torch.cuda.is_available(),
        )

    return train_loader, val_loader, test_loader, class_counts
