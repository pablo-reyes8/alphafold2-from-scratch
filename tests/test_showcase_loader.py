from __future__ import annotations

from pathlib import Path

import pytest
from torch.utils.data import DataLoader

pytest.importorskip("Bio")

from data.collate_proteins import collate_proteins
from data.dataloaders import (
    EXTRA_MSA_FEATURE_DIM,
    TEMPLATE_ANGLE_FEATURE_DIM,
    TEMPLATE_PAIR_FEATURE_DIM,
    FoldbenchProteinDataset,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
SHOWCASE_MANIFEST = ROOT_DIR / "data" / "showcase_manifest.csv"


def test_showcase_dataset_and_dataloader_expose_template_and_extra_msa_features():
    dataset = FoldbenchProteinDataset(
        manifest_csv=str(SHOWCASE_MANIFEST),
        max_msa_seqs=8,
        max_extra_msa_seqs=16,
        max_templates=4,
        verbose=False,
    )

    assert len(dataset) == 2

    sample = dataset[0]
    assert sample["masked_msa_true"].shape == sample["msa_tokens"].shape
    assert sample["masked_msa_mask"].shape == sample["msa_tokens"].shape
    assert sample["extra_msa_feat"].ndim == 3
    assert sample["extra_msa_feat"].shape[-1] == EXTRA_MSA_FEATURE_DIM
    assert sample["extra_msa_mask"].shape == sample["extra_msa_feat"].shape[:2]
    assert sample["template_angle_feat"].ndim == 3
    assert sample["template_angle_feat"].shape[-1] == TEMPLATE_ANGLE_FEATURE_DIM
    assert sample["template_pair_feat"].ndim == 4
    assert sample["template_pair_feat"].shape[-1] == TEMPLATE_PAIR_FEATURE_DIM
    assert sample["template_mask"].shape == sample["template_angle_feat"].shape[:2]
    assert len(sample["template_chain_ids"]) >= 1

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_proteins,
    )
    batch = next(iter(loader))

    assert batch["seq_tokens"].shape[0] == 2
    assert batch["msa_tokens"].shape[0] == 2
    assert batch["masked_msa_true"].shape == batch["msa_tokens"].shape
    assert batch["masked_msa_mask"].shape == batch["msa_mask"].shape
    assert batch["extra_msa_feat"] is not None
    assert batch["extra_msa_mask"] is not None
    assert batch["template_angle_feat"] is not None
    assert batch["template_pair_feat"] is not None
    assert batch["template_mask"] is not None

    assert batch["extra_msa_feat"].shape[:3] == batch["extra_msa_mask"].shape
    assert batch["extra_msa_feat"].shape[-1] == EXTRA_MSA_FEATURE_DIM
    assert batch["template_angle_feat"].shape[-1] == TEMPLATE_ANGLE_FEATURE_DIM
    assert batch["template_pair_feat"].shape[-1] == TEMPLATE_PAIR_FEATURE_DIM
    assert batch["template_mask"].shape == batch["template_angle_feat"].shape[:3]
    assert all(len(chain_ids) >= 1 for chain_ids in batch["template_chain_ids"])


def test_showcase_dataset_random_crop_keeps_all_modalities_aligned():
    dataset = FoldbenchProteinDataset(
        manifest_csv=str(SHOWCASE_MANIFEST),
        max_msa_seqs=8,
        max_extra_msa_seqs=16,
        max_templates=4,
        crop_size=64,
        random_crop=False,
        verbose=False,
    )

    sample = dataset[0]
    assert len(sample["sequence_str"]) == 64
    assert sample["seq_tokens"].shape == (64,)
    assert sample["msa_tokens"].shape[1] == 64
    assert sample["msa_mask"].shape[1] == 64
    assert sample["masked_msa_true"].shape[1] == 64
    assert sample["masked_msa_mask"].shape[1] == 64
    assert sample["coords_n"].shape == (64, 3)
    assert sample["coords_ca"].shape == (64, 3)
    assert sample["coords_c"].shape == (64, 3)
    assert sample["valid_res_mask"].shape == (64,)
    assert sample["valid_backbone_mask"].shape == (64,)
    assert sample["torsion_true"].shape == (64, 3, 2)
    assert sample["torsion_mask"].shape == (64, 3)
    assert sample["dist_map"].shape == (64, 64)
    assert sample["extra_msa_feat"].shape[1] == 64
    assert sample["extra_msa_mask"].shape[1] == 64
    assert sample["template_angle_feat"].shape[1] == 64
    assert sample["template_pair_feat"].shape[1:3] == (64, 64)
    assert sample["template_mask"].shape[1] == 64

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_proteins,
    )
    batch = next(iter(loader))

    assert batch["seq_tokens"].shape == (2, 64)
    assert batch["msa_tokens"].shape[-1] == 64
    assert batch["masked_msa_true"].shape[-1] == 64
    assert batch["masked_msa_mask"].shape[-1] == 64
    assert batch["extra_msa_feat"].shape[2] == 64
    assert batch["template_angle_feat"].shape[2] == 64
    assert batch["template_pair_feat"].shape[2:4] == (64, 64)
    assert batch["coords_ca"].shape == (2, 64, 3)
    assert batch["dist_map"].shape == (2, 64, 64)
