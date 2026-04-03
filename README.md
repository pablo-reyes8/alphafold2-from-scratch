<p align="center">
  <img src="assets/Inital Banner.png" width="1000"/>
</p>


# AlphaFold2: 

<div align="center">

**Dissecting geometric deep learning and structural biology representations from scratch.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#installation)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](#installation)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)
[![Status](https://img.shields.io/badge/status-Research%20Prototype-orange)](#project-status)

</div>

> **🚧 Development Status:** This repository is a living research environment under active development. We are currently iterating on core architectural modules, optimizing the geometric forward pass, and expanding our structural validation suite. Updates are frequent as we refine the implementation toward full end-to-end parity with the AlphaFold2 design space.

---

## Overview

This repository provides a **from-scratch, modular PyTorch implementation of the core AlphaFold2 architecture**.

While the original DeepMind release and frameworks like OpenFold are designed for large-scale production, this project is built for **architectural transparency, research experimentation, and hands-on learning**. It breaks down the structural biology pipeline into inspectable, hackable modules, allowing researchers and students to study how Multiple Sequence Alignments (MSA), pair representations, and geometric heads interact at the tensor level.

It is also designed with accessibility in mind for people who do not have access to large training clusters. For that reason, we include `notebooks/Alpha_Fold_Spanish.ipynb`, a complete notebook that makes it easier to explore the project end-to-end from environments such as **Google Colab** or **Kaggle**, without needing a heavy local setup.

More broadly, the goal is to make this architecture genuinely accessible to study: anyone should be able to inspect, modify, and run meaningful experiments with the model, adapting its scale to the hardware they actually have rather than being excluded by the need for large training infrastructure.



## Architectural Focus

<p align="center">
  <img src="assets/Ia_showcase_image.png" width="1000"/>
</p>


The implementation strictly follows the representational flow of the original paper, providing clean PyTorch modules for:

* **Representational Flow:** Explicit handling of MSA, Pair, and Single state embeddings.
* **The Evoformer:** Fully implemented axial attention mechanisms and triangle updates for spatial reasoning.
* **Structure Module:** Native PyTorch implementations of **Invariant Point Attention (IPA)**, rigid body transformations, and structural loss computations (FAPE).
* **Geometric Precision:** Robust unit testing suite specifically targeting structural losses and rotational invariants.


## Data & Reproducibility

To make experimentation easier to reproduce, the repository follows a **manifest-based workflow**. This keeps the data pipeline more organized and makes it easier to move between local environments, notebooks, and scripted runs.

* **Foldbench Support:** Includes scripts to download and preprocess a subset of Foldbench.
* **Config-Driven Experiments:** Main settings such as model size, depth, learning rate, and EMA can be adjusted through YAML files.
* **Data Inspection Utilities:** Provides simple CLI tools to inspect manifests, preview A3M files, and visualize CA distance maps before training.
* **Notebook-Friendly Workflow:** A complete notebook is included in `notebooks/Alpha_Fold.ipynb` so the full pipeline can also be explored interactively in Colab or Kaggle.

---

## Repository structure

```text
.
├── config/
│   ├── data/
│   │   └── foldbench_subset.yaml
│   └── experiments/
│       ├── af2_poc.yaml
│       └── alphafold2_full_reference.yaml
├── data/
│   ├── download_data.sh
│   ├── preproces_data.py
│   ├── dataloaders.py
│   ├── visualize_data.py
│   └── Proteinas_secuencias.csv
├── scripts/
│   ├── prepare_data.py
│   ├── inspect_data.py
│   ├── validate_model.py
│   └── train_model.py
├── model/
├── training/
├── tests/
├── notebooks/
├── requirements.txt
├── Dockerfile
└── README.md
```

### Key files

- `data/download_data.sh` — downloads the Foldbench subset using a target file or CSV input.
- `data/preproces_data.py` — builds or rewrites the manifest and emits YAML summaries.
- `data/dataloaders.py` — dataset code supporting both manifest-based and raw-folder loading.
- `data/visualize_data.py` — command-line inspection utilities for manifests, A3M previews, and CA distance maps.
- `scripts/prepare_data.py` — high-level CLI to download data, refresh manifests, and smoke-test dataloaders.
- `scripts/inspect_data.py` — concise dataset inspection CLI, including batch previews and a simple 3D backbone renderer.
- `scripts/validate_model.py` — instantiates the model stack, runs synthetic forward validation, and dispatches pytest.
- `scripts/train_model.py` — config-driven launcher for end-to-end training runs.
- `config/experiments/af2_poc.yaml` — lightweight proof-of-concept experiment config.
- `config/experiments/alphafold2_full_reference.yaml` — reference values collected from AlphaFold/OpenFold-style configs.

### Bundled test data

The repository includes a tiny downloaded test subset under `data/af_subset_showcase` together with `data/showcase_manifest.csv`, so the data pipeline can be sanity-checked without downloading the full dataset first.

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

The full notebook `notebooks/train_model_local.ipynb` version exposes many knobs, but the smallest useful training setup looks like this:

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
    max_msa_seqs=128,
)
loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_proteins)

model = AlphaFold2(
    n_tokens=max(AA_VOCAB.values()) + 1,
    pad_idx=AA_VOCAB["-"],
    c_m=256,
    c_z=128,
    c_s=256,
    num_evoformer_blocks=2,
    num_structure_blocks=4,
    n_torsions=3,
).to(device)

criterion = AlphaFoldLoss()
total_steps = 20 * len(loader)
optimizer, scheduler = build_optimizer_and_scheduler(
    model=model,
    lr=1e-4,
    weight_decay=1e-4,
    total_steps=total_steps,
    warmup_steps=max(10, int(0.05 * total_steps)),
)
ema = EMA(model, decay=0.999, device="cpu", use_num_updates=True)
amp_cfg = build_amp_config(device=device, amp_enabled=True, amp_dtype="bf16")

ideal_backbone_local = torch.tensor([
    [-1.458, 0.000, 0.000],
    [0.000, 0.000, 0.000],
    [0.547, 1.426, 0.000],
    [0.224, 2.617, 0.000],
], dtype=torch.float32, device=device)

result = train_alphafold2(
    model=model,
    train_loader=loader,
    optimizer=optimizer,
    criterion=criterion,
    scheduler=scheduler,
    ema=ema,
    scaler=amp_cfg["scaler"],
    device=device,
    epochs=20,
    amp_enabled=amp_cfg["amp_enabled"],
    amp_dtype=amp_cfg["amp_dtype_requested"],
    ideal_backbone_local=ideal_backbone_local,
    ckpt_dir="checkpoints_af2",
    run_name="af2_poc",
)
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

### `config/experiments/af2_poc.yaml`

This config mirrors the current notebook-scale proof of concept and is suitable for smaller experimental runs.

Current example values:

- `max_msa_seqs: 128`
- `batch_size: 2`
- `epochs: 20`
- `lr: 1e-4`
- `num_evoformer_blocks: 2`
- `num_structure_blocks: 4`

### `config/experiments/alphafold2_full_reference.yaml`

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

**Core Principles:**

* **Architectural Transparency:** Designed to be read, debugged, and mathematically verified at the tensor level. There is no hidden logic; the mapping from the original paper's equations to PyTorch modules is direct and explicit.
* **Modular Extensibility:** Every mechanism—from the Evoformer's axial attention to the Invariant Point Attention (IPA)—is fully decoupled. Researchers can isolate, modify, or completely redesign structural modules without fighting the framework.
* **Rigorous Prototyping:** Provides a robust, high-fidelity environment for testing novel geometric learning hypotheses, custom attention mechanisms, and alternative structural losses before scaling them to production clusters.

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

If this repository helps your work, please cite the repository and also cite the original AlphaFold and related implementation papers that inspired the architecture.

A simple placeholder BibTeX entry for the repository could look like:

```bibtex
@software{reyes_alphafold2_from_scratch,
  author = {Pablo Reyes},
  title = {AlphaFold2 From Scratch},
  year = {2026},
  url = {https://github.com/pablo-reyes8/alpha-fold2}
}
```

---

## Acknowledgments

This repository is inspired by the AlphaFold2 line of work and the broader ecosystem of open implementations and educational reverse-engineering efforts around protein structure prediction.

Special credit belongs to the original AlphaFold work and to the open-source community that has made this field far more accessible to study.

---

## License

This project is licensed under the **MIT License**.
