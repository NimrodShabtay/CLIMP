# CLIMP: Contrastive Language-Image Mamba Pretraining

Evaluation code for the [CLIMP paper](https://arxiv.org/abs/2601.06891) on the NoCaps image-text retrieval benchmark.

CLIMP pairs a VMamba-Base vision encoder with Mamba-based text encoders, trained with bidirectional contrastive loss on CC12M.

## Models

Pretrained weights are available on HuggingFace:

| Model | Vision Encoder | Text Encoder | Projection Dim | HuggingFace |
|-------|---------------|--------------|----------------|-------------|
| CLIMP-Mamba1 | VMamba-Base | Mamba-1.4B | 768 | [NimrodShabtay1986/CLIMP-Mamba1](https://huggingface.co/NimrodShabtay1986/CLIMP-Mamba1) |
| CLIMP-Mamba2 | VMamba-Base | Mamba2-1.3B | 768 | [NimrodShabtay1986/CLIMP-Mamba2](https://huggingface.co/NimrodShabtay1986/CLIMP-Mamba2) |

## Installation

```bash
pip install -r requirements.txt
```

**Note:** `mamba-ssm` and `causal-conv1d` require CUDA compilation. See [mamba-ssm installation](https://github.com/state-spaces/mamba#installation) for details.

## Usage

```bash
# Evaluate CLIMP-Mamba1 (downloads weights from HuggingFace automatically)
python eval_nocaps.py --model mamba1

# Evaluate CLIMP-Mamba2
python eval_nocaps.py --model mamba2

# With custom image resolution
python eval_nocaps.py --model mamba1 --image-size 512

# Or with a local checkpoint
python eval_nocaps.py --model mamba1 --checkpoint /path/to/model.safetensors
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | required | Model variant: `mamba1` or `mamba2` |
| `--checkpoint` | None (auto-download) | Path to `model.safetensors` file, downloads from HF if not provided |
| `--image-size` | 224 | Input image resolution |
| `--batch-size` | 16 | Evaluation batch size |
| `--device` | cuda | Device (`cuda` or `cpu`) |

## Results on NoCaps (Validation, 224px)

| Model | Image R@1 | Image R@5 | Image R@10 | Text R@1 | Text R@5 | Text R@10 |
|-------|-----------|-----------|------------|----------|----------|-----------|
| CLIMP-Mamba1 | 38.2 | 70.5 | 81.8 | 51.8 | 81.9 | 90.9 |
| CLIMP-Mamba2 | 37.4 | 70.2 | 81.6 | 51.0 | 81.3 | 90.9 |

## Architecture

CLIMP uses:
- **Vision**: VMamba-Base (SSM-based vision encoder with 2D selective scan)
- **Text**: Mamba-1.4B or Mamba2-1.3B (SSM-based causal language model, backbone only)
- **Projection**: Linear layers mapping both modalities to a shared 768-dim space
- **Training objective**: Symmetric contrastive loss (InfoNCE) with learned temperature

Text pooling extracts the embedding at the last non-padding token position.

## Citation

```bibtex
@article{climp2026,
  title={CLIMP: Contrastive Language-Image Mamba Pretraining},
  author={Shabtay, Nimrod and Zimerman, Itamar and Schwartz, Eli and Giryes, Raja},
  journal={arXiv preprint arXiv:2601.06891},
  year={2026}
}
```

## License

Apache-2.0
