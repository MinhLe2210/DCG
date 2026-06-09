import logging
import math

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


class LoRALayer(nn.Module):

    def __init__(self, in_dim, out_dim, rank=4, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.rank = rank

        # A: [r, in_dim], B: [out_dim, r]; effective weight delta = B @ A.
        self.lora_A = nn.Parameter(torch.zeros((rank, in_dim)))
        self.lora_B = nn.Parameter(torch.zeros((out_dim, rank)))

        # Standard LoRA init: A ~ kaiming, B = 0 so the adapter starts as
        # identity (zero perturbation) at the beginning of training.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # einsum keeps arbitrary leading dimensions intact.
        lora_out = torch.einsum("...d, rd -> ...r", x, self.lora_A)
        lora_out = torch.einsum("...r, or -> ...o", lora_out, self.lora_B)
        return lora_out * (self.alpha / self.rank)


class LoRALinear(nn.Module):

    def __init__(self, original_layer, rank=4, alpha=1.0, trainable_orig=False, dropout=0.1):
        super().__init__()
        self.original_layer = original_layer

        if not trainable_orig:
            for param in self.original_layer.parameters():
                param.requires_grad = False

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        self.lora = LoRALayer(in_features, out_features, rank, alpha)
        # Dropout is applied only on the LoRA branch to regularize the adapter
        # without disturbing the frozen path.
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        original_output = self.original_layer(x)
        lora_output = self.lora(x)
        lora_output = self.dropout(lora_output)

        if original_output.shape != lora_output.shape:
            raise ValueError(
                f"dimension mismatch: original output {original_output.shape}, "
                f"LoRA output {lora_output.shape}"
            )

        return original_output + lora_output

    def __getattr__(self, name):
        # Forward common ``nn.Linear`` attributes (weight / bias) to the
        # underlying layer so external code that introspects them keeps working.
        if name == "weight":
            return self.original_layer.weight
        if name == "bias":
            return self.original_layer.bias
        return super().__getattr__(name)


def _matches_lora_target(module_name: str, target: str) -> bool:
    """
    Match either:
      - an exact leaf name: q_proj
      - a full suffix: attention.q_proj

    Avoid loose substring matching such as:
        "proj" in "attention.q_proj"
    """
    leaf_name = module_name.rsplit(".", 1)[-1]

    return (
        leaf_name == target
        or module_name == target
        or module_name.endswith(f".{target}")
    )

def apply_lora_to_linear_layers(
    model,
    rank=4,
    alpha=1.0,
    target_modules=None,
    trainable_orig=False,
    dropout=0.1,
):
    """
    Replace selected nn.Linear layers with LoRALinear wrappers.

    Matching is based on exact leaf names or dotted suffixes. The function
    raises an error when zero modules are injected to prevent silent training
    with a completely frozen RGB backbone.
    """
    if not target_modules:
        raise ValueError("target_modules must contain at least one module name.")

    injected_names = []

    # Snapshot the module list before replacing children in-place.
    for name, module in list(model.named_modules()):
        # Prevent nested wrapping when this function is accidentally called
        # more than once.
        if ".original_layer" in name:
            continue

        if not isinstance(module, nn.Linear):
            continue

        if not any(
            _matches_lora_target(name, target)
            for target in target_modules
        ):
            continue

        parent_name, _, child_name = name.rpartition(".")
        parent = model if not parent_name else get_submodule(model, parent_name)

        current_child = getattr(parent, child_name)

        if isinstance(current_child, LoRALinear):
            continue

        setattr(
            parent,
            child_name,
            LoRALinear(
                original_layer=module,
                rank=rank,
                alpha=alpha,
                trainable_orig=trainable_orig,
                dropout=dropout,
            ),
        )

        injected_names.append(name)

    if not injected_names:
        raise RuntimeError(
            "LoRA injection matched 0 nn.Linear layers. "
            f"Requested targets: {target_modules}"
        )

    logger.info("[LoRA] Injected %d linear layers", len(injected_names))

    for name in injected_names[:20]:
        logger.info("  - %s", name)

    if len(injected_names) > 20:
        logger.info("  ... and %d more", len(injected_names) - 20)

    return model


def get_submodule(model, submodule_name):
    """Resolve a dotted path (e.g. ``"blocks.0.mlp.fc1"``) to a submodule."""

    if not submodule_name:
        return model

    parts = submodule_name.split(".")
    current_module = model

    for part in parts:
        if part.isdigit():
            current_module = current_module[int(part)]
        else:
            current_module = getattr(current_module, part)

    return current_module


def get_lora_params(model):

    lora_params = []
    seen = set()

    for _, module in model.named_modules():
        if isinstance(module, LoRALinear):
            for p in module.lora.parameters():
                pid = id(p)
                if pid not in seen:
                    seen.add(pid)
                    lora_params.append(p)

    return lora_params
