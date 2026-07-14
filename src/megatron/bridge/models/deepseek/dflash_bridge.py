# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DFlash/DSpark extension for DeepSeek-V4 checkpoints.

This bridge is intentionally not registered as the default ``deepseek_v4``
handler.  Use it explicitly when the checkpoint contains DFlash/DSpark heads
stored under the ``mtp.*`` namespace.
"""

import json
import logging
import re
from dataclasses import fields
from pathlib import Path
from typing import Any

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    GatedMLPMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.deepseek.deepseek_v4_bridge import (
    DeepSeekV4Bridge,
    _HCAlphaMapping,
    _HCAlphaSecondaryMapping,
)
from megatron.bridge.models.deepseek.dflash_provider import DFlashModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.utils.common_utils import print_rank_0


logger = logging.getLogger(__name__)
_MTP_KEY_RE = re.compile(r"(?:^|\.)mtp\.(\d+)\.")


def _first_config_value(config: Any, *names: str) -> Any | None:
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return value
    return None


def _infer_num_dflash_layers_from_index(model_name_or_path: str | Path | None) -> int | None:
    """Infer DFlash MTP layer count from a local HF safetensors index."""
    if model_name_or_path is None:
        return None

    index_path = Path(model_name_or_path) / "model.safetensors.index.json"
    if not index_path.is_file():
        return None

    try:
        with index_path.open() as f:
            weight_map = json.load(f).get("weight_map", {})
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read DFlash safetensors index %s: %s", index_path, exc)
        return None

    mtp_indices = [int(match.group(1)) for key in weight_map if (match := _MTP_KEY_RE.search(key))]
    if not mtp_indices:
        return None
    return max(mtp_indices) + 1


def _get_num_dflash_layers(hf_pretrained: PreTrainedCausalLM) -> int | None:
    hf_config = hf_pretrained.config
    explicit_value = _first_config_value(hf_config, "dflash_num_layers", "num_dflash_layers")
    if explicit_value:
        return int(explicit_value)

    inferred_value = _infer_num_dflash_layers_from_index(getattr(hf_pretrained, "model_name_or_path", None))
    if inferred_value:
        return inferred_value

    dspark_target_layer_ids = getattr(hf_config, "dspark_target_layer_ids", None)
    if dspark_target_layer_ids:
        return len(dspark_target_layer_ids)

    num_nextn_predict_layers = getattr(hf_config, "num_nextn_predict_layers", None)
    return int(num_nextn_predict_layers or 0) or None


def _get_dflash_target_layer_indices(hf_config: Any) -> list[int]:
    target_layers = _first_config_value(
        hf_config,
        "dflash_target_layer_indices",
        "target_layer_indices",
        "aux_hidden_state_layer_ids",
        "dspark_target_layer_ids",
    )
    return [int(layer_idx) for layer_idx in (target_layers or [])]


def _get_dflash_num_draft_tokens(hf_config: Any) -> int:
    value = _first_config_value(
        hf_config,
        "dflash_num_draft_tokens",
        "num_draft_tokens",
        "dspark_block_size",
    )
    return int(value if value is not None else 2)


def _get_csa_compress_ratios(hf_config: Any, num_dflash_layers: int | None) -> list[int]:
    ratios = [int(ratio) for ratio in (getattr(hf_config, "compress_ratios", None) or [])]
    if not ratios:
        return []

    expected_len = int(hf_config.num_hidden_layers) + int(num_dflash_layers or 0)
    if len(ratios) < expected_len:
        ratios.extend([0] * (expected_len - len(ratios)))
    return ratios[:expected_len]


def _normalize_csa_compress_ratios(provider: DFlashModelProvider, hf_config: Any) -> None:
    expected_len = int(hf_config.num_hidden_layers) + int(provider.dflash_num_layers or 0)
    ratios = _get_csa_compress_ratios(hf_config, provider.dflash_num_layers)
    if not ratios:
        ratios = [int(ratio) for ratio in (provider.csa_compress_ratios or [])]

    if len(ratios) < expected_len:
        ratios.extend([0] * (expected_len - len(ratios)))

    provider.csa_compress_ratios = ratios[:expected_len]


class DeepSeekV4DFlashBridge(DeepSeekV4Bridge):
    """DeepSeek-V4 bridge variant that builds mega-dflash models."""

    PROVIDER_CLASS = DFlashModelProvider

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> DFlashModelProvider:
        """Create a DFlash provider initialized from the DeepSeek-V4 provider."""
        base_provider = super().provider_bridge(hf_pretrained)
        provider = DFlashModelProvider(
            **{field.name: getattr(base_provider, field.name) for field in fields(base_provider)}
        )

        for hook_name in ("_pre_wrap_hooks", "_post_wrap_hooks"):
            if hasattr(base_provider, hook_name):
                setattr(provider, hook_name, list(getattr(base_provider, hook_name)))

        hf_config = hf_pretrained.config
        provider.dflash_num_layers = _get_num_dflash_layers(hf_pretrained)
        # Keep this in sync with DFlash so MCore validates per-layer DSv4
        # fields such as csa_compress_ratios against the full decoder+DFlash
        # layer count. DFlashModelProvider still disables native MTP
        # construction by passing mtp_block_spec=None.
        provider.mtp_num_layers = provider.dflash_num_layers
        _normalize_csa_compress_ratios(provider, hf_config)
        # Conversion does not need the Apex fused weight-gradient extension, and
        # many lightweight conversion environments do not install it.
        provider.gradient_accumulation_fusion = False
        provider.dflash_num_draft_tokens = _get_dflash_num_draft_tokens(hf_config)
        provider.dflash_target_layer_indices = _get_dflash_target_layer_indices(hf_config)

        provider.dflash_loss_scaling_factor = float(getattr(hf_config, "dflash_loss_scaling_factor", 1.0))
        provider.dflash_use_repeated_layer = bool(getattr(hf_config, "dflash_use_repeated_layer", False))
        provider.dflash_detach_hidden_states = bool(getattr(hf_config, "dflash_detach_hidden_states", False))
        provider.dflash_mask_token_id = _first_config_value(hf_config, "dflash_mask_token_id", "dspark_noise_token_id")
        provider.dflash_freeze_target = bool(getattr(hf_config, "dflash_freeze_target", False))

        provider.dspark_markov_rank = getattr(hf_config, "dspark_markov_rank", None)
        provider.dspark_loss_decay_gamma = getattr(hf_config, "dspark_loss_decay_gamma", 4.0)
        provider.dspark_l1_loss_alpha = float(getattr(hf_config, "dspark_l1_loss_alpha", 0.9))
        provider.dspark_confidence_head_alpha = float(getattr(hf_config, "dspark_confidence_head_alpha", 1.0))
        provider.dspark_confidence_loss_scale = getattr(hf_config, "dspark_confidence_loss_scale", None)

        print_rank_0(
            "DFlash provider config: "
            f"num_layers={provider.dflash_num_layers}, "
            f"target_layer_indices={provider.dflash_target_layer_indices}, "
            f"num_draft_tokens={provider.dflash_num_draft_tokens}, "
            f"dspark_markov_rank={provider.dspark_markov_rank}, "
            f"csa_compress_ratios_len={len(provider.csa_compress_ratios or [])}"
        )

        return provider

    def _dflash_mappings(self) -> list:
        num_dflash_layers = int(_get_num_dflash_layers(self.hf_pretrained) or 0)
        if num_dflash_layers <= 0:
            return []

        head_prefix = f"mtp.{num_dflash_layers - 1}"
        mappings = [
            AutoMapping("dflash.main_proj.weight", "mtp.0.main_proj.weight"),
            AutoMapping("dflash.main_norm.weight", "mtp.0.main_norm.weight"),
            AutoMapping("dflash.final_norm.weight", f"{head_prefix}.norm.weight"),
            ReplicatedMapping("dflash.hc_head_fn", f"{head_prefix}.hc_head_fn"),
            ReplicatedMapping("dflash.hc_head_base", f"{head_prefix}.hc_head_base"),
            ReplicatedMapping("dflash.hc_head_scale", f"{head_prefix}.hc_head_scale"),
            AutoMapping(
                "dflash.markov_embedding.word_embeddings.weight",
                f"{head_prefix}.markov_head.markov_w1.weight",
            ),
            AutoMapping("dflash.markov_lm_head.weight", f"{head_prefix}.markov_head.markov_w2.weight"),
            ReplicatedMapping("dflash.confidence_head.weight", f"{head_prefix}.confidence_head.proj.weight"),
        ]

        for layer_idx in range(num_dflash_layers):
            ck_pfx = f"mtp.{layer_idx}"
            mg_pfx = f"dflash.layers.{layer_idx}"
            mappings += [
                AutoMapping(f"{mg_pfx}.input_layernorm.weight", f"{ck_pfx}.attn_norm.weight"),
                AutoMapping(f"{mg_pfx}.pre_mlp_layernorm.weight", f"{ck_pfx}.ffn_norm.weight"),
                AutoMapping(f"{mg_pfx}.self_attention.linear_q_down_proj.weight", f"{ck_pfx}.attn.wq_a.weight"),
                AutoMapping(f"{mg_pfx}.self_attention.q_layernorm.weight", f"{ck_pfx}.attn.q_norm.weight"),
                AutoMapping(f"{mg_pfx}.self_attention.linear_q_up_proj.weight", f"{ck_pfx}.attn.wq_b.weight"),
                AutoMapping(f"{mg_pfx}.self_attention.linear_kv_proj.weight", f"{ck_pfx}.attn.wkv.weight"),
                AutoMapping(f"{mg_pfx}.self_attention.kv_layernorm.weight", f"{ck_pfx}.attn.kv_norm.weight"),
                ReplicatedMapping(f"{mg_pfx}.self_attention.linear_o_group_proj", f"{ck_pfx}.attn.wo_a.weight"),
                AutoMapping(f"{mg_pfx}.self_attention.linear_proj.weight", f"{ck_pfx}.attn.wo_b.weight"),
                ColumnParallelMapping(f"{mg_pfx}.self_attention.core_attention.attn_sink", f"{ck_pfx}.attn.attn_sink"),
                AutoMapping(f"{mg_pfx}.mlp.router.weight", f"{ck_pfx}.ffn.gate.weight"),
                AutoMapping(f"{mg_pfx}.mlp.router.expert_bias", f"{ck_pfx}.ffn.gate.bias"),
                AutoMapping(f"{mg_pfx}.mlp.router.tid2eid", f"{ck_pfx}.ffn.gate.tid2eid"),
                GatedMLPMapping(
                    megatron_param=f"{mg_pfx}.mlp.experts.linear_fc1.weight*",
                    gate=f"{ck_pfx}.ffn.experts.*.w1.weight",
                    up=f"{ck_pfx}.ffn.experts.*.w3.weight",
                ),
                AutoMapping(f"{mg_pfx}.mlp.experts.linear_fc2.weight*", f"{ck_pfx}.ffn.experts.*.w2.weight"),
                GatedMLPMapping(
                    megatron_param=f"{mg_pfx}.mlp.shared_experts.linear_fc1.weight",
                    gate=f"{ck_pfx}.ffn.shared_experts.w1.weight",
                    up=f"{ck_pfx}.ffn.shared_experts.w3.weight",
                ),
                AutoMapping(
                    f"{mg_pfx}.mlp.shared_experts.linear_fc2.weight",
                    f"{ck_pfx}.ffn.shared_experts.w2.weight",
                ),
                ReplicatedMapping(
                    f"{mg_pfx}.self_attention_hyper_connection.mapping_proj.weight",
                    f"{ck_pfx}.hc_attn_fn",
                ),
                ReplicatedMapping(f"{mg_pfx}.self_attention_hyper_connection.bias", f"{ck_pfx}.hc_attn_base"),
                ReplicatedMapping(f"{mg_pfx}.mlp_hyper_connection.mapping_proj.weight", f"{ck_pfx}.hc_ffn_fn"),
                ReplicatedMapping(f"{mg_pfx}.mlp_hyper_connection.bias", f"{ck_pfx}.hc_ffn_base"),
                _HCAlphaMapping(
                    megatron_pre=f"{mg_pfx}.self_attention_hyper_connection.alpha_pre",
                    megatron_post=f"{mg_pfx}.self_attention_hyper_connection.alpha_post",
                    megatron_res=f"{mg_pfx}.self_attention_hyper_connection.alpha_res",
                    hf_param=f"{ck_pfx}.hc_attn_scale",
                ),
                _HCAlphaMapping(
                    megatron_pre=f"{mg_pfx}.mlp_hyper_connection.alpha_pre",
                    megatron_post=f"{mg_pfx}.mlp_hyper_connection.alpha_post",
                    megatron_res=f"{mg_pfx}.mlp_hyper_connection.alpha_res",
                    hf_param=f"{ck_pfx}.hc_ffn_scale",
                ),
                _HCAlphaSecondaryMapping(
                    f"{mg_pfx}.self_attention_hyper_connection.alpha_post",
                    f"{ck_pfx}.hc_attn_scale",
                    1,
                ),
                _HCAlphaSecondaryMapping(
                    f"{mg_pfx}.self_attention_hyper_connection.alpha_res",
                    f"{ck_pfx}.hc_attn_scale",
                    2,
                ),
                _HCAlphaSecondaryMapping(
                    f"{mg_pfx}.mlp_hyper_connection.alpha_post",
                    f"{ck_pfx}.hc_ffn_scale",
                    1,
                ),
                _HCAlphaSecondaryMapping(
                    f"{mg_pfx}.mlp_hyper_connection.alpha_res",
                    f"{ck_pfx}.hc_ffn_scale",
                    2,
                ),
            ]

        return mappings

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Return DeepSeek-V4 mappings plus DFlash/DSpark head mappings."""
        base_registry = super().mapping_registry()
        return MegatronMappingRegistry(*base_registry.get_all_mappings(), *self._dflash_mappings())
