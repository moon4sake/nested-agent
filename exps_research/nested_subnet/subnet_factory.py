import copy
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel


@dataclass
class SubnetSpec:
    model_name: str
    sub_layers: int
    torch_dtype: torch.dtype = torch.bfloat16
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def load_full_model_and_tokenizer(spec: SubnetSpec) -> Tuple[PreTrainedModel, AutoTokenizer]:
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_name,
        torch_dtype=spec.torch_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        spec.model_name,
        padding_side="left",
        add_eos_token=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(spec.device)
    return model, tokenizer


def _get_backbone(model: PreTrainedModel):
    if hasattr(model, "model"):
        return model.model
    if hasattr(model, "transformer"):
        return model.transformer
    raise ValueError("Unsupported model backbone layout.")


def _get_layers(backbone):
    if hasattr(backbone, "layers"):
        return backbone.layers
    if hasattr(backbone, "h"):
        return backbone.h
    raise ValueError("Unsupported layer attribute on backbone.")


def _set_layers(backbone, layers):
    if hasattr(backbone, "layers"):
        backbone.layers = layers
        return
    if hasattr(backbone, "h"):
        backbone.h = layers
        return
    raise ValueError("Unsupported layer attribute on backbone.")


def _share_backbone_parts(full_backbone, sub_backbone):
    if hasattr(full_backbone, "embed_tokens"):
        sub_backbone.embed_tokens = full_backbone.embed_tokens
    if hasattr(full_backbone, "wte"):
        sub_backbone.wte = full_backbone.wte
    if hasattr(full_backbone, "norm"):
        sub_backbone.norm = full_backbone.norm
    if hasattr(full_backbone, "ln_f"):
        sub_backbone.ln_f = full_backbone.ln_f
    if hasattr(full_backbone, "rotary_emb"):
        sub_backbone.rotary_emb = full_backbone.rotary_emb


def _set_num_layers(config, num_layers: int) -> None:
    if hasattr(config, "num_hidden_layers"):
        config.num_hidden_layers = num_layers
    if hasattr(config, "n_layer"):
        config.n_layer = num_layers


def _unwrap_base_model(model: PreTrainedModel) -> PreTrainedModel:
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return model.base_model.model
    if hasattr(model, "get_base_model"):
        return model.get_base_model()
    return model


def build_subnet(full_model: PreTrainedModel, sub_layers: int) -> PreTrainedModel:
    base_model = _unwrap_base_model(full_model)
    full_backbone = _get_backbone(base_model)
    full_layers = _get_layers(full_backbone)
    if sub_layers > len(full_layers):
        raise ValueError(f"sub_layers={sub_layers} exceeds full depth {len(full_layers)}")

    sub_config = copy.deepcopy(base_model.config)
    _set_num_layers(sub_config, sub_layers)
    sub_model = base_model.__class__(sub_config)
    sub_backbone = _get_backbone(sub_model)

    _share_backbone_parts(full_backbone, sub_backbone)
    _set_layers(sub_backbone, torch.nn.ModuleList(full_layers[:sub_layers]))

    if hasattr(sub_model, "lm_head") and hasattr(base_model, "lm_head"):
        sub_model.lm_head = base_model.lm_head
    if hasattr(sub_model, "embed_tokens") and hasattr(base_model, "embed_tokens"):
        sub_model.embed_tokens = base_model.embed_tokens

    sub_model.to(next(base_model.parameters()).device)
    sub_model.tie_weights()
    return sub_model


def set_adapter_enabled(model: PreTrainedModel, enabled: bool) -> None:
    if hasattr(model, "enable_adapter") and hasattr(model, "disable_adapter"):
        if enabled:
            model.enable_adapter()
        else:
            model.disable_adapter()


def maybe_load_peft_adapter(model: PreTrainedModel, adapter_path: Optional[str]) -> PreTrainedModel:
    if adapter_path is None or adapter_path == "":
        return model
    from peft import PeftModel

    return PeftModel.from_pretrained(model, adapter_path)
