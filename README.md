<p align="center">
  <img src="assets/Inital Banner.png" width="1000"/>
</p>

# AlphaFold2: From Scratch

<div align="center">

**A modular PyTorch reconstruction of AlphaFold2 built for architectural transparency, controlled ablations, and serious geometric deep learning research.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#installation)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](#installation)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)
[![Status](https://img.shields.io/badge/status-Research%20Prototype-orange)](#project-status)

</div>

> **🚧 Development Status:** Active research prototype focused on core architectural modules, geometric forward modeling, and structural validation.

> **⚠️​ Scope:** This implementation aims to reproduce the **deep learning pipeline of AlphaFold2** as faithfully as possible for research. We intentionally omit the final all-atom chemical geometry and side-chain reconstruction pipeline, keeping the repository centered on **DL research, modularity, and controlled experimentation**.

---

## Overview

This repository provides a **from-scratch, modular PyTorch implementation of the AlphaFold2 architecture**.

Where the original DeepMind release and frameworks like OpenFold are designed for large-scale production, this project is built for **architectural transparency, research experimentation, and real hands-on understanding**. It breaks the AlphaFold2 pipeline into inspectable, hackable components so researchers and students can study how Multiple Sequence Alignments (MSA), pair representations, Evoformer updates, and geometric heads interact at the tensor level.

It is also designed with accessibility in mind for people who do not have access to large training clusters. For that reason, we include [Alpha_Fold_English.ipynb](notebooks/Alpha_Fold_English.ipynb) and [Alpha_Fold_Spanish.ipynb](notebooks/Alpha_Fold_Spanish.ipynb): self-contained notebooks that gather the full pipeline in one place, with clear explanations of both the dataset and the forward pass, making the project easy to study end-to-end in environments such as **Google Colab** or **Kaggle** without requiring a heavy local setup.

More broadly, the goal is to make this architecture genuinely accessible as an object of serious study rather than a black box reserved for large-scale infrastructure. To support that goal, the repository includes both a structured ablation suite ([Ablations Suite](#ablations-suite)) and multiple YAML experiment presets in ([Experiments](#config/experiments)), so researchers and students can systematically probe, stress-test, reinterpret, and scale the model in ways that match their available hardware. The aim is not only to move AlphaFold2-style experimentation beyond the canonical high-cost setup, but to make the architecture easy to manipulate, isolate, and study through controlled variation.

## Index

- [Architectural Focus](#architectural-focus)
- [Data & Reproducibility](#data--reproducibility)
- [Repository structure](#repository-structure)
- [Quickstart](#quickstart)
- [Training](#training)
- [Ablations Suite](#ablations-suite)
- [CLI workflows](#cli-workflows)
- [Configs](#configs)
- [Docker](#docker)
- [Design Philosophy](#design-philosophy)
- [Intended audience](#intended-audience)
- [Roadmap](#roadmap)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Architectural Focus

<p align="center">
  <img src="assets/Ia_showcase_image.png" width="1000"/>
</p>

The implementation strictly follows the representational flow of the original paper, providing clean PyTorch modules for:

- **Representational Flow:** Explicit handling of MSA, Pair, and Single state embeddings.
- **The Evoformer:** Fully implemented axial attention mechanisms and triangle updates for spatial reasoning.
- **Structure Module:** Native PyTorch implementations of **Invariant Point Attention (IPA)**, rigid body transformations, and structural loss computations (FAPE).
- **Geometric Precision:** Robust unit testing suite specifically targeting structural losses and rotational invariants.

## Data & Reproducibility

To make experimentation easier to reproduce, the repository follows a **manifest-based workflow**. This keeps the data pipeline more organized and makes it easier to move between local environments, notebooks, and scripted runs.

- **Foldbench Support:** Includes scripts to download and preprocess a subset of Foldbench.
- **Config-Driven Experiments:** Main settings such as model size, depth, learning rate, and EMA can be adjusted through YAML files.
- **Feature-Rich Loader:** The current dataloader returns sequence/MSA tensors plus `extra_msa_feat`, `extra_msa_mask`, `template_angle_feat`, `template_pair_feat`, and `template_mask` when those artifacts are present in the Foldbench assets.
- **Data Inspection Utilities:** Provides simple CLI tools to inspect manifests, preview A3M files, and visualize CA distance maps before training.
- **Notebook-Friendly Workflow:** The main walkthrough notebook is [Alpha_Fold_English.ipynb](notebooks/Alpha_Fold_English.ipynb), and a local training-focused version is available in [notebooks\train_model_setup_examples.ipynb](notebooks/train_model_local.ipynb).

---

## Repository structure

```text
.
├── config/                    # YAML configs for data paths and experiment presets
│   ├── data/
│   └── experiments/
├── data/                      # manifest-based data pipeline plus a tiny bundled showcase subset
│   ├── download_data.sh
│   ├── foldbench.py
│   ├── preproces_data.py
│   ├── dataloaders.py
│   ├── collate_proteins.py
│   ├── visualize_data.py
│   ├── showcase_manifest.csv
│   └── af_subset_showcase/
├── model/                     # AlphaFold2 architecture, heads, geometric blocks, and losses
│   └── losses/
├── training/                  # single-device training loop, ablation registry, AMP, EMA, checkpoints, and metrics
│   ├── ablations/             # predefined architecture and loss ablation presets
│   └── train_paralel/         # DDP and model-parallel helpers
├── scripts/                   # operational CLIs for data prep, validation, and training
│   ├── prepare_data.py
│   ├── inspect_data.py
│   ├── validate_model.py
│   ├── train_model.py
│   ├── train_parallel.py
│   ├── train_ablation.py
│   ├── train_ablation_parallel.py
│   └── ablations/
├── tests/                     # data, model, loss, and CLI coverage
├── notebooks/                 # interactive experiments for Colab or local exploration
├── paper/                     # reference material from the AlphaFold paper and notes
├── assets/                    # README visuals and showcase media
├── requirements.txt
├── Dockerfile
└── README.md
```

### Key files

- [data/download_data.sh](data/download_data.sh) — downloads the Foldbench subset from a target list or CSV input.
- [data/preproces_data.py](data/preproces_data.py) — rebuilds manifests, normalizes local paths, and emits YAML summaries.
- [data/dataloaders.py](data/dataloaders.py) — dataset layer that maps manifests, mmCIF structures, MSA files, and torsion targets into tensors.
- [scripts/prepare_data.py](scripts/prepare_data.py) — high-level CLI for downloading data, refreshing manifests, and smoke-testing loaders.
- [model/alphafold2.py](model/alphafold2.py) — top-level AlphaFold2-like model that wires embeddings, Evoformer, structure, recycling, and heads.
- [model/evoformer_stack.py](model/evoformer_stack.py) — stacked Evoformer trunk used to refine MSA and pair representations.
- [model/structure_block.py](model/structure_block.py) — structure module built around IPA, transitions, and rigid-frame updates.
- [model/alphafold2_full_loss.py](model/alphafold2_full_loss.py) — full training loss orchestrator combining FAPE, distogram, pLDDT, and torsion supervision.
- [model/losses/](model/losses/) — component losses and helpers for geometry-aware supervision.
- [training/train_one_epoch.py](training/train_one_epoch.py) — per-epoch optimization routine with AMP, recycling, logging, and metric collection.
- [training/train_alphafold2.py](training/train_alphafold2.py) — full training orchestrator for checkpointing, resume, monitoring, and epoch scheduling.
- [training/ablations/catalog.py](training/ablations/catalog.py) — registry of prebuilt architecture and loss ablations resolved on top of a base experiment config.
- [training/ablations/runtime.py](training/ablations/runtime.py) — resolves baseline or named ablations into a safe config variant without changing the default training path.
- [training/train_paralel/data_parallel.py](training/train_paralel/data_parallel.py) — DDP utilities, distributed samplers, and rank synchronization helpers.
- [training/train_paralel/model_parallel.py](training/train_paralel/model_parallel.py) — two-stage model-parallel wrapper for splitting AlphaFold2 across GPUs.
- [scripts/train_model.py](scripts/train_model.py) — standard config-driven single-device training launcher.
- [scripts/train_parallel.py](scripts/train_parallel.py) — multi-GPU launcher for DDP, model parallelism, and hybrid setups.
- [scripts/train_ablation.py](scripts/train_ablation.py) — single-device launcher for named architecture and loss ablations.
- [scripts/ablations/run_suite.py](scripts/ablations/run_suite.py) — runs multiple presets sequentially and exports a comparison table.
- [scripts/ablations/README.md](scripts/ablations/README.md) — detailed documentation for the proposed ablation suite and each preset.

### Bundled test data

The repository includes a tiny downloaded test subset under [data/af_subset_showcase](data/af_subset_showcase) together with [data/showcase_manifest.csv](data/showcase_manifest.csv), so the data pipeline can be sanity-checked without downloading the full dataset first.

---

## Quickstart

### 1) Create an environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Download the subset

```bash
bash data/download_data.sh --targets-csv data/Proteinas_secuencias.csv
```

Or through the repo CLI:

```bash
python3 scripts/prepare_data.py download --targets-csv data/Proteinas_secuencias.csv
```

### 3) Rebuild the manifest with local paths

```bash
python3 -m data.preproces_data \
  --config config/data/foldbench_subset.yaml \
  --json-path data/af_subset/jsons/fb_protein.json \
  --msa-root data/af_subset/foldbench_msas \
  --cif-root data/af_subset/reference_structures
```

### 4) Inspect the dataset

```bash
python3 -m data.visualize_data manifest-summary --manifest-csv data/Proteinas_secuencias.csv
python3 -m data.visualize_data msa-preview --a3m-path data/af_subset/foldbench_msas/7qrj_A/cfdb_hits.a3m
```

Or use the higher-level inspection scripts:

```bash
python3 scripts/inspect_data.py loader-preview --config config/experiments/af2_poc.yaml --max-samples 2
python3 scripts/inspect_data.py protein-3d \
  --cif-path data/af_subset/reference_structures/7qrj-assembly1_1.cif \
  --chain-id A \
  --output artifacts/7qrj_A_backbone.png
```

### 5) Use the manifest in the dataset

```python
from data.dataloaders import FoldbenchProteinDataset

dataset = FoldbenchProteinDataset(manifest_csv="data/Proteinas_secuencias.csv")
```

## Training

### Minimal Python setup

The full notebook [notebooks/train_model_local.ipynb](notebooks/train_model_local.ipynb) exposes many knobs, but the smallest useful training setup looks like this:

```python
import torch
from torch.utils.data import DataLoader

from data.collate_proteins import collate_proteins
from data.dataloaders import AA_VOCAB, FoldbenchProteinDataset
from model.alphafold2 import AlphaFold2
from model.alphafold2_full_loss import AlphaFoldLoss
from training.autocast import build_amp_config
from training.ema import EMA
from training.scheduler_warmup import build_optimizer_and_scheduler
from training.train_alphafold2 import train_alphafold2

device = "cuda" if torch.cuda.is_available() else "cpu"


# You need to download first the data
dataset = FoldbenchProteinDataset(
    manifest_csv="data/showcase_manifest.csv",
    max_msa_seqs=128, crop_size=64, random_crop=True,)

loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_proteins)

model = AlphaFold2(
    n_tokens=max(AA_VOCAB.values()) + 1,
    pad_idx=AA_VOCAB["-"],
    num_evoformer_blocks=2,
    num_structure_blocks=4,
    n_torsions=3, num_res_blocks_torsion=2,
    extra_msa_stack_enabled=True, template_stack_enabled=True,
    ).to(device)

criterion = AlphaFoldLoss()

total_steps = 20 * len(loader)
optimizer, scheduler = build_optimizer_and_scheduler(
    model=model,
    lr=1e-4,
    weight_decay=1e-4,
    total_steps=total_steps,
    warmup_steps=max(10, int(0.05 * total_steps)))

ema = EMA(model, decay=0.999, device="cpu", use_num_updates=True)
amp_cfg = build_amp_config(device=device, amp_enabled=True, amp_dtype="bf16")

ideal_backbone_local = torch.tensor([
    [-1.458, 0.000, 0.000],
    [0.000, 0.000, 0.000],
    [0.547, 1.426, 0.000],
    [0.224, 2.617, 0.000]], dtype=torch.float32, device=device)

result = train_alphafold2(
    model=model,
    train_loader=loader,
    optimizer=optimizer,
    criterion=criterion,
    scheduler=scheduler,
    ema=ema,
    scaler=amp_cfg["scaler"],
    grad_clip=1.0,
    device=device,
    epochs=20,
    amp_enabled=amp_cfg["amp_enabled"],
    amp_dtype=amp_cfg["amp_dtype_requested"],
    ideal_backbone_local=ideal_backbone_local,
    ckpt_dir="checkpoints_af2",
    num_recycles=3, 
    stochastic_recycling=True,
    run_name="af2_poc")

```

### CLI training

For the standard single-device launcher:

```bash
python3 scripts/train_model.py --config config/experiments/af2_poc.yaml --device cuda
```

For 2 GPUs with data parallelism through DDP:

```bash
torchrun --nproc_per_node=2 scripts/train_parallel.py \
  --config config/experiments/af2_poc.yaml \
  --manifest-csv data/showcase_manifest.csv \
  --parallel-mode ddp
```

For 2 GPUs with model parallelism:

```bash
python3 scripts/train_parallel.py \
  --config config/experiments/af2_poc.yaml \
  --manifest-csv data/showcase_manifest.csv \
  --parallel-mode model \
  --model-devices cuda:0,cuda:1
```

The hybrid mode `--parallel-mode hybrid` is also available, but it is intended for multi-replica setups and typically needs at least 4 GPUs.

### Ablations Suite

The repository includes an opt-in ablation suite documented in [scripts/ablations/README.md](scripts/ablations/README.md). The baseline path remains unchanged: if you instantiate `AlphaFold2(...)` and `AlphaFoldLoss(...)` without `ablation=...`, every ablation switch stays off and the normal training route is preserved.

The current presets combine high-level architectural and loss interventions:

| Ablation     | Type         | Intervention                                     | Research Question                                 |
| ------------ | ------------ | ------------------------------------------------ | ------------------------------------------------- |
| **BASELINE** | —            | None                                             | Reference performance                             |
| **AF2_1**    | Architecture | Disable pair stack + recycling                   | Does pair representation evolution matter?        |
| **AF2_2**    | Architecture | Triangle attention disabled, multiplication kept | Which geometric update is more critical?          |
| **AF2_3**    | Loss         | FAPE only (distogram, pLDDT, torsion disabled)   | Is FAPE sufficient for structure supervision?     |
| **AF2_4**    | Architecture | Block-specific structure module params           | Do independent structure blocks improve learning? |
| **AF2_5**    | Architecture | Evoformer bypass (minimal baseline)              | What is the lower-bound performance?              |

For detailed rationale, hypotheses, and expected results for each preset, see [scripts/ablations/README.md](scripts/ablations/README.md).

List the available presets:

```bash
python3 scripts/train_ablation.py --list
```

Inspect a resolved ablation config without building the full training stack:

```bash
python3 scripts/train_ablation.py \
  --config config/experiments/af2_poc.yaml \
  --ablation AF2_1 \
  --show
```

Train one ablation on a single device:

```bash
python3 scripts/train_ablation.py \
  --config config/experiments/af2_poc.yaml \
  --ablation AF2_3 \
  --device cuda
```

Train the same ablation with multi-GPU parallelism:

```bash
torchrun --nproc_per_node=2 scripts/train_ablation_parallel.py \
  --config config/experiments/af2_poc.yaml \
  --manifest-csv data/showcase_manifest.csv \
  --ablation AF2_3 \
  --parallel-mode ddp
```

Run a whole sweep and export a comparison table:

```bash
python3 scripts/ablations/run_suite.py \
  --config config/experiments/af2_poc.yaml \
  --include-baseline \
  --all \
  --output-dir artifacts/ablation_suite
```

---

## CLI workflows

### Prepare data and loader

```bash
python3 scripts/prepare_data.py bootstrap \
  --data-config config/data/foldbench_subset.yaml \
  --experiment-config config/experiments/af2_poc.yaml
```

### Validate the model stack

```bash
python3 scripts/validate_model.py instantiate --config config/experiments/af2_poc.yaml
python3 scripts/validate_model.py forward-smoke --config config/experiments/af2_poc.yaml --device cpu
python3 scripts/validate_model.py pytest --target tests --pytest-arg=-q
```

### Run a safe training smoke test

```bash
python3 scripts/train_model.py --config config/experiments/af2_poc.yaml --device cpu --dry-run
```

### Launch training with recycling overrides

```bash
python3 scripts/train_model.py \
  --config config/experiments/af2_poc.yaml \
  --device cuda \
  --num-recycles 0 \
  --stochastic-recycling \
  --max-recycles 3
```

---

## Configs

### [config/experiments/af2_poc.yaml](config/experiments/af2_poc.yaml)

This config mirrors the current notebook-scale proof of concept and is suitable for smaller experimental runs. It now includes the loader fields for extra MSA and template features used by the current pipeline.

Current example values:

- `max_msa_seqs: 128`
- `max_extra_msa_seqs: 256`
- `max_templates: 4`
- `batch_size: 2`
- `epochs: 20`
- `lr: 1e-4`
- `num_evoformer_blocks: 2`
- `num_structure_blocks: 4`

### [config/experiments/af2_canonical.yaml](config/experiments/af2_canonical.yaml)

Canonical current-scope preset: 48 Evoformer blocks, 8 structure blocks, `c_m=256`, `c_z=128`, `c_s=384`, recycling enabled, `max_msa_seqs=128`, `max_extra_msa_seqs=1024`, and `max_templates=4`.

### [config/experiments/af2_4xa100.yaml](config/experiments/af2_4xa100.yaml)

Cluster-oriented preset for distributed runs on a typical `4x A100` setup, with moderate trunk width/depth and the full extra-MSA/template path still enabled.

### [config/experiments/af2_single_a100_40gb.yaml](config/experiments/af2_single_a100_40gb.yaml)

Single-GPU preset sized for roughly one `40 GB A100`, keeping template and extra-MSA conditioning but with a smaller trunk than the canonical reference.

### [config/experiments/af2_low_vram.yaml](config/experiments/af2_low_vram.yaml)

Low-VRAM preset for Colab-class GPUs in the `15-20 GB` range, using a reduced trunk, shallow template/extra-MSA stacks, and `data/showcase_manifest.csv` by default.

### [config/experiments/alphafold2_full_reference.yaml](config/experiments/alphafold2_full_reference.yaml)

This file is a **reference document**, not a statement that the current code already consumes every field end-to-end.

Its role is to provide a structured target for future extension and to document the broader AlphaFold/OpenFold design space.

---

## Docker

A small CPU-oriented image can be built with:

```bash
docker build -t alphafold-from-scratch .
```

This image is intended for **environment setup, utilities, and data tooling**, not for serious GPU training.

---

## Design Philosophy

This repository is architected with a singular premise: **true understanding of geometric deep learning requires unconstrained access to its atomic components.**

Rather than providing a monolithic black box or a superficial tutorial, this codebase is engineered specifically for deep architectural study and rapid ablation. It strips away the distributed production overhead of frameworks like OpenFold to expose the bare mathematical and algorithmic reality of the network.

This implementation is designed to cover the **core deep learning pipeline of AlphaFold2** as faithfully as possible. It includes backbone torsion supervision and template conditioning, while intentionally stopping short of the final all-atom chemical reconstruction, side-chain geometry, and post-processing refinement stack.

**Core Principles:**

- **Architectural Transparency:** Designed to be read, debugged, and mathematically verified at the tensor level. There is no hidden logic; the mapping from the original paper's equations to PyTorch modules is direct and explicit.
- **Modular Extensibility:** Every mechanism—from the Evoformer’s axial attention to the Invariant Point Attention (IPA)—is fully decoupled. Researchers can isolate, modify, or completely redesign structural modules without fighting the framework.
- **Rigorous Prototyping:** Provides a robust, high-fidelity environment for testing novel geometric learning hypotheses, custom attention mechanisms, and alternative structural losses before scaling them to production clusters.

This makes the repository a specialized tool for researchers dissecting structural biology models, engineers debugging complex 3D equivariance, and anyone focused on advancing the theoretical foundations of the AlphaFold family.

---

## Intended audience

This project may be useful for:

- ML researchers studying geometric deep learning or protein structure prediction,
- students implementing AlphaFold2-style systems to truly understand them,
- engineers who want a smaller environment for experimentation,
- researchers building derivatives, ablations, or teaching materials.

It is probably **not** the best starting point if your main goal is immediately obtaining state-of-the-art folding performance with industrial robustness. In that case, official or mature large-scale implementations will usually be a better operational choice.

---

## Roadmap

A realistic roadmap for this repository could include:

- [ ] tighter end-to-end training validation
- [ ] expanded benchmark and evaluation scripts
- [ ] example inference notebook or script
- [ ] reproducibility report for a reference training run
- [ ] Visualizations for understanding AlphaFold

---

## Citation

If this repository contributes to your research, please cite:

1. **This repository**, for the educational from-scratch implementation.
2. **The original AlphaFold2 paper**, for the core architecture and methodology.

A suggested BibTeX entry for this repository is:

```bibtex
@software{reyes2026alphafold2,
  author       = {Pablo Reyes},
  title        = {AlphaFold2 From Scratch},
  year         = {2026},
  url          = {https://github.com/pablo-reyes8/alpha-fold2},
  note         = {GitHub repository. From-scratch modular PyTorch implementation of core AlphaFold2 components for research, education, and experimentation.}
}
```

Please also cite the foundational papers:

```bibtex
@article{jumper2021alphafold,
  author  = {Jumper, John and Evans, Richard and Pritzel, Alexander and Green, Tim and Figurnov, Michael and Ronneberger, Olaf and Tunyasuvunakool, Kathryn and Bates, Russ and {\v{Z}}{\'\i}dek, Augustin and Potapenko, Anna and Bridgland, Alex and Meyer, Clemens and Kohl, Simon A. A. and Ballard, Andrew J. and Cowie, Andrew and Romera-Paredes, Bernardino and Nikolov, Stanislav and Jain, Rishub and Adler, Jonas and Back, Trevor and Petersen, Stig and Reiman, David and Clancy, Ellen and Zielinski, Michal and Steinegger, Martin and Pacholska, Michalina and Berghammer, Tamas and Bodenstein, Sebastian and Silver, David and Vinyals, Oriol and Senior, Andrew W. and Kavukcuoglu, Koray and Kohli, Pushmeet and Hassabis, Demis},
  title   = {Highly accurate protein structure prediction with AlphaFold},
  journal = {Nature},
  volume  = {596},
  number  = {7873},
  pages   = {583--589},
  year    = {2021},
  doi     = {10.1038/s41586-021-03819-2}
}

```

---

## Acknowledgments

This repository is inspired by the AlphaFold2 line of work and the broader ecosystem of open implementations and educational reverse-engineering efforts around protein structure prediction.

Special credit belongs to the original AlphaFold work and to the open-source community that has made this field far more accessible to study.

---

## License

This project is licensed under the **MIT License**.
