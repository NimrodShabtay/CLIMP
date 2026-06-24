import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
import argparse
import types

from transformers import (
    MambaForCausalLM,
    Mamba2ForCausalLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    MambaConfig,
    AutoConfig,
)
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(__file__))
from vmamba import VSSM


VMAMBA_BASE_CONFIG = dict(
    patch_size=4,
    in_chans=3,
    num_classes=1000,
    depths=[2, 2, 15, 2],
    dims=128,
    ssm_d_state=1,
    ssm_ratio=2.0,
    ssm_dt_rank="auto",
    ssm_act_layer="silu",
    ssm_conv=3,
    ssm_conv_bias=False,
    ssm_drop_rate=0.0,
    ssm_init="v2",
    forward_type="v05_noz",
    mlp_ratio=4.0,
    mlp_act_layer="gelu",
    mlp_drop_rate=0.0,
    drop_path_rate=0.6,
    patch_norm=True,
    norm_layer="ln2d",
    downsample_version="v3",
    patchembed_version="v2",
    gmlp=False,
    use_checkpoint=False,
    posembed=False,
    imgsize=224,
)


MODEL_CONFIGS = {
    "mamba1": {
        "text_model": "state-spaces/mamba-1.4b-hf",
        "text_hidden_size": 2048,
        "projection_dim": 768,
        "temperature": 0.07,
    },
    "mamba2": {
        "text_model": "AntonV/mamba2-1.3b-hf",
        "text_hidden_size": 2048,
        "projection_dim": 768,
        "temperature": 0.07,
    },
}


def find_last_one_and_extract_values(binary_tensor, values_tensor, default_value=0):
    B, L = binary_tensor.shape
    D = values_tensor.shape[-1]

    flipped = torch.flip(binary_tensor, dims=[1])
    first_one_in_flipped = torch.argmax(flipped, dim=1)
    last_one_index = L - 1 - first_one_in_flipped

    mask = binary_tensor.gather(1, last_one_index.unsqueeze(1)).squeeze(1) == 1
    last_one_index = torch.where(mask, last_one_index, torch.tensor(-1))

    valid_indices = torch.clamp(last_one_index, min=0)
    expanded_indices = valid_indices.unsqueeze(1).unsqueeze(2).expand(B, 1, D)
    extracted = values_tensor.gather(1, expanded_indices).squeeze(1)

    valid_mask = (last_one_index >= 0).unsqueeze(1).expand(B, D)
    result = torch.where(valid_mask, extracted, torch.tensor(default_value))

    return last_one_index, result


def build_vmamba(img_size=224):
    config = dict(VMAMBA_BASE_CONFIG)
    config["imgsize"] = img_size
    model = VSSM(**config)

    # Remove classifier head
    if hasattr(model, "classifier"):
        if hasattr(model.classifier, "head"):
            delattr(model.classifier, "head")

    # Add config attribute for compatibility
    model.config = types.SimpleNamespace()
    model.config.dims = [model.classifier.norm.normalized_shape[0]]

    return model


class MambaCLIP(nn.Module):
    def __init__(self, model_type="mamba1", image_size=224):
        super().__init__()
        cfg = MODEL_CONFIGS[model_type]

        # Vision encoder
        self.vision_encoder = build_vmamba(img_size=image_size)
        self.vision_projection = nn.Linear(
            self.vision_encoder.config.dims[-1], cfg["projection_dim"]
        )

        # Text encoder
        if model_type == "mamba1":
            self.text_encoder = MambaForCausalLM.from_pretrained(
                cfg["text_model"]
            ).backbone
        else:
            self.text_encoder = AutoModelForCausalLM.from_pretrained(
                cfg["text_model"]
            ).backbone

        self.text_tokenizer = AutoTokenizer.from_pretrained(cfg["text_model"])
        self.text_projection = nn.Linear(
            self.text_encoder.config.hidden_size, cfg["projection_dim"]
        )

        self.logit_scale = nn.Parameter(
            torch.ones([]) * np.log(1 / cfg["temperature"])
        )

    def encode_text(self, input_ids, attention_mask=None, output_hidden_states=True):
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )[0]
        last_text_embeds = find_last_one_and_extract_values(attention_mask, text_outputs)
        last_token = last_text_embeds[1].to(self.text_projection.weight.dtype)
        proj_text_embeds = self.text_projection(last_token)
        return proj_text_embeds

    def encode_image(self, pixel_values, output_hidden_states=True):
        image_embeds = self.vision_encoder(pixel_values)
        image_embeds = self.vision_projection(image_embeds)
        return image_embeds


HF_REPOS = {
    "mamba1": "NimrodShabtay1986/CLIMP-Mamba1",
    "mamba2": "NimrodShabtay1986/CLIMP-Mamba2",
}


def load_climp(model_type, checkpoint_path=None, image_size=224, device="cuda"):
    model = MambaCLIP(model_type=model_type, image_size=image_size)

    if checkpoint_path is None:
        from huggingface_hub import hf_hub_download
        checkpoint_path = hf_hub_download(
            repo_id=HF_REPOS[model_type], filename="model.safetensors"
        )

    state_dict = load_file(checkpoint_path)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    return model
