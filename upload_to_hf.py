"""
Prepare and upload CLIMP models to HuggingFace Hub.

Usage:
    python upload_to_hf.py --model mamba1 --checkpoint /path/to/model.safetensors --repo-id <user>/climp-mamba1
    python upload_to_hf.py --model mamba2 --checkpoint /path/to/model.safetensors --repo-id <user>/climp-mamba2
"""

import argparse
import json
import os
import shutil
import tempfile

from huggingface_hub import HfApi, create_repo


MODEL_CARDS = {
    "mamba1": """---
license: apache-2.0
tags:
- clip
- mamba
- vision-language
- contrastive-learning
- vmamba
datasets:
- conceptual_12m
pipeline_tag: zero-shot-image-classification
---

# CLIMP-Mamba1

**C**ontrastive **L**anguage-**I**mage pretraining with **M**amba **P**retraining (CLIMP) using Mamba-1.4B as text encoder.

## Model Description

| Component | Details |
|-----------|---------|
| Vision Encoder | VMamba-Base (128→256→512→1024 dims, depths [2,2,15,2]) |
| Text Encoder | Mamba-1.4B (`state-spaces/mamba-1.4b-hf`) |
| Projection Dim | 768 |
| Training Data | CC12M |
| Image Resolution | 224×224 |
| Loss | Symmetric InfoNCE (learned temperature) |

## Usage

```python
from models import load_climp
from data.utils import transform_image

model = load_climp("mamba1")  # downloads from HuggingFace automatically
transform = transform_image(224)
```

See the [demo repository](https://github.com/TBD) for full evaluation code.

## Results (NoCaps Validation)

| Metric | Score |
|--------|-------|
| Image Retrieval R@1 | TBD |
| Image Retrieval R@5 | TBD |
| Text Retrieval R@1 | TBD |
| Text Retrieval R@5 | TBD |

## Paper

[CLIMP: Contrastive Language-Image Mamba Pretraining](https://arxiv.org/abs/2601.06891)

```bibtex
@article{climp2026,
  title={CLIMP: Contrastive Language-Image Mamba Pretraining},
  author={Shabtay, Nimrod and Zimerman, Itamar and Schwartz, Eli and Giryes, Raja},
  journal={arXiv preprint arXiv:2601.06891},
  year={2026}
}
```
""",
    "mamba2": """---
license: apache-2.0
tags:
- clip
- mamba
- mamba2
- vision-language
- contrastive-learning
- vmamba
datasets:
- conceptual_12m
pipeline_tag: zero-shot-image-classification
---

# CLIMP-Mamba2

**C**ontrastive **L**anguage-**I**mage pretraining with **M**amba **P**retraining (CLIMP) using Mamba2-1.3B as text encoder.

## Model Description

| Component | Details |
|-----------|---------|
| Vision Encoder | VMamba-Base (128→256→512→1024 dims, depths [2,2,15,2]) |
| Text Encoder | Mamba2-1.3B (`AntonV/mamba2-1.3b-hf`) |
| Projection Dim | 768 |
| Training Data | CC12M |
| Image Resolution | 224×224 |
| Loss | Symmetric InfoNCE (learned temperature) |

## Usage

```python
from models import load_climp
from data.utils import transform_image

model = load_climp("mamba2")  # downloads from HuggingFace automatically
transform = transform_image(224)
```

See the [demo repository](https://github.com/TBD) for full evaluation code.

## Results (NoCaps Validation)

| Metric | Score |
|--------|-------|
| Image Retrieval R@1 | TBD |
| Image Retrieval R@5 | TBD |
| Text Retrieval R@1 | TBD |
| Text Retrieval R@5 | TBD |

## Paper

[CLIMP: Contrastive Language-Image Mamba Pretraining](https://arxiv.org/abs/2601.06891)

```bibtex
@article{climp2026,
  title={CLIMP: Contrastive Language-Image Mamba Pretraining},
  author={Shabtay, Nimrod and Zimerman, Itamar and Schwartz, Eli and Giryes, Raja},
  journal={arXiv preprint arXiv:2601.06891},
  year={2026}
}
```
""",
}


MODEL_CONFIGS = {
    "mamba1": {
        "architecture": "MambaCLIP",
        "vision_encoder": "VMamba-Base",
        "text_encoder": "state-spaces/mamba-1.4b-hf",
        "text_encoder_type": "mamba1",
        "projection_dim": 768,
        "image_size": 224,
        "temperature": 0.07,
        "vmamba_config": {
            "patch_size": 4,
            "in_chans": 3,
            "depths": [2, 2, 15, 2],
            "embed_dim": 128,
            "ssm_d_state": 1,
            "ssm_ratio": 2.0,
            "ssm_conv": 3,
            "forward_type": "v05_noz",
            "mlp_ratio": 4.0,
            "norm_layer": "ln2d",
            "downsample_version": "v3",
            "patchembed_version": "v2",
        },
    },
    "mamba2": {
        "architecture": "MambaCLIP",
        "vision_encoder": "VMamba-Base",
        "text_encoder": "AntonV/mamba2-1.3b-hf",
        "text_encoder_type": "mamba2",
        "projection_dim": 768,
        "image_size": 224,
        "temperature": 0.07,
        "vmamba_config": {
            "patch_size": 4,
            "in_chans": 3,
            "depths": [2, 2, 15, 2],
            "embed_dim": 128,
            "ssm_d_state": 1,
            "ssm_ratio": 2.0,
            "ssm_conv": 3,
            "forward_type": "v05_noz",
            "mlp_ratio": 4.0,
            "norm_layer": "ln2d",
            "downsample_version": "v3",
            "patchembed_version": "v2",
        },
    },
}


def prepare_and_upload(model_type, checkpoint_path, repo_id, private=False):
    api = HfApi()

    print(f"Creating repo: {repo_id}")
    create_repo(repo_id, repo_type="model", exist_ok=True, private=private)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Copy checkpoint
        dst_checkpoint = os.path.join(tmp_dir, "model.safetensors")
        print(f"Copying checkpoint: {checkpoint_path} -> {dst_checkpoint}")
        shutil.copy2(checkpoint_path, dst_checkpoint)

        # Write config.json
        config_path = os.path.join(tmp_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(MODEL_CONFIGS[model_type], f, indent=2)
        print(f"Written config.json")

        # Write model card
        readme_path = os.path.join(tmp_dir, "README.md")
        with open(readme_path, "w") as f:
            f.write(MODEL_CARDS[model_type])
        print(f"Written README.md")

        # Upload
        print(f"Uploading to {repo_id}...")
        api.upload_folder(
            folder_path=tmp_dir,
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"Done! Model available at: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Upload CLIMP model to HuggingFace")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["mamba1", "mamba2"],
        help="Model variant",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model.safetensors",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace repo ID (e.g., user/climp-mamba1)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repo private",
    )
    args = parser.parse_args()

    prepare_and_upload(args.model, args.checkpoint, args.repo_id, args.private)


if __name__ == "__main__":
    main()
