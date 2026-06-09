import logging
import os

import torch
import torch.nn as nn
from transformers import AutoModel

logger = logging.getLogger(__name__)

# Embedding dimension of every supported DINO variant.
CHANNELS = {
    "dinov2-base": 768,
    "dinov2-large": 1024,
    "dinov2-giant": 1536,
    # DINOv3 Base
    "dinov3-base": 768,
    # DINOv3 Large
    "dinov3-large": 1024,
}


def resolve_local_dino_path(name: str, pretrained_root: str) -> str:
    assert pretrained_root is not None, (
        "[DINO] pretrained_root must be provided and point to a local checkpoint."
    )

    candidates = [
        pretrained_root,
        os.path.join(pretrained_root, name),
    ]
    for cand in candidates:
        if os.path.isdir(cand) and os.path.exists(os.path.join(cand, "config.json")):
            logger.info("[DINO] Using local checkpoint at: %s", cand)
            return cand

    raise FileNotFoundError(
        f"[DINO] Could not find local checkpoint for '{name}'. "
        f"Looked under: {pretrained_root}. "
        f"Please download the model locally and point pretrained_root to it."
    )


def _num_register_tokens(backbone: nn.Module) -> int:
    """Return the number of register tokens used by a DINO backbone.

    Hugging Face DINOv3 stores this value in ``backbone.config``. Native
    implementations may expose it directly on the backbone. DINOv2 variants
    without registers fall back to zero.
    """
    config = getattr(backbone, "config", None)
    num_register_tokens = getattr(config, "num_register_tokens", None)
    if num_register_tokens is None:
        num_register_tokens = getattr(backbone, "num_register_tokens", 0)
    return int(num_register_tokens or 0)


def _extract_cls_and_tokens(backbone: nn.Module, rgb: torch.Tensor):
    if hasattr(backbone, "forward_features"):
        feats = backbone.forward_features(rgb)
        if isinstance(feats, dict):
            x_norm = feats.get("x_norm", None)
            cls_tok = feats.get("x_norm_clstoken", None)
            patch_tok = feats.get("x_norm_patchtokens", None)

            # When only x_norm is available, skip [CLS] and DINOv3 register
            # tokens. PGCM must receive image patch tokens only.
            patch_start = 1 + _num_register_tokens(backbone)
            if (
                patch_tok is None
                and x_norm is not None
                and x_norm.dim() == 3
                and x_norm.size(1) >= patch_start
            ):
                patch_tok = x_norm[:, patch_start:]

            if (
                cls_tok is None
                and x_norm is not None
                and x_norm.dim() == 3
                and x_norm.size(1) >= 1
            ):
                cls_tok = x_norm[:, 0]

            if cls_tok is not None and patch_tok is not None:
                return cls_tok, patch_tok

    outputs = backbone(rgb)
    if isinstance(outputs, dict):
        last_hidden = outputs.get("last_hidden_state", None)
    else:
        # ModelOutput exposes attributes; fall back to tuple/list as last resort.
        last_hidden = getattr(outputs, "last_hidden_state", None)

    if last_hidden is None and isinstance(outputs, (tuple, list)) and len(outputs) > 0:
        last_hidden = outputs[0]

    if last_hidden is None:
        raise RuntimeError(
            "DINO backbone output does not contain 'last_hidden_state' (dict/attr/tuple)."
        )

    patch_start = 1 + _num_register_tokens(backbone)
    if last_hidden.dim() != 3 or last_hidden.size(1) < patch_start:
        raise RuntimeError(
            f"Unexpected DINO hidden state shape: {list(last_hidden.shape)}"
        )

    cls_tok = last_hidden[:, 0]  # [B, D]
    patch_tok = last_hidden[:, patch_start:]  # [B, N, D]
    return cls_tok, patch_tok


class RgbStream(nn.Module):
    def __init__(self, name: str, pretrained_root: str):
        super().__init__()

        if name not in CHANNELS:
            raise ValueError(
                f"Unsupported DINO variant '{name}'. "
                f"Supported variants: {list(CHANNELS.keys())}"
            )

        self.name = name
        self._embed_dim = CHANNELS[name]

        model_path = resolve_local_dino_path(name, pretrained_root)
        logger.info("[DINO] Loading backbone '%s' from: %s", name, model_path)
        self.backbone = AutoModel.from_pretrained(model_path)

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def forward(self, rgb: torch.Tensor):
        """Run the backbone and return ``(f_rgb_cls, f_rgb_tokens)``."""
        cls_tok, patch_tok = _extract_cls_and_tokens(self.backbone, rgb)
        return cls_tok, patch_tok