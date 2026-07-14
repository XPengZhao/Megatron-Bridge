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

"""DFlash provider for DeepSeek-V4 based models.

The implementation intentionally keeps the DFlash model code in the external
``mega-dflash`` package.  This provider only adapts Bridge's provider contract
to construct ``DFlashGPTModel`` while preserving DeepSeek-V4 config fields and
weight-loading hooks.
"""

import inspect
from dataclasses import dataclass, field

from megatron.core.models.backends import LocalSpecProvider
from megatron.core.transformer import ModuleSpec

from megatron.bridge.models import gpt_provider
from megatron.bridge.models.mla_provider import MLAModelProvider


def _load_dflash_components():
    """Import mega-dflash lazily so regular DeepSeek-V4 users are unaffected."""
    try:
        from dflash.core.config import DFlashConfig
        from dflash.core.models.gpt.dflash_gpt_model import DFlashGPTModel
        from dflash_builders import _mtp_spec_to_dflash_spec, get_dflash_spec_for_backend
    except ImportError as exc:
        raise ImportError(
            "DFlashModelProvider requires mega-dflash on PYTHONPATH. "
            "For this repo layout, add 3rdparty/mega-dflash to PYTHONPATH before constructing the model."
        ) from exc

    return DFlashConfig, DFlashGPTModel, _mtp_spec_to_dflash_spec, get_dflash_spec_for_backend


def _te_spec_provider():
    try:
        from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
    except ImportError:
        return LocalSpecProvider()
    return TESpecProvider()


@dataclass
class DFlashModelProvider(MLAModelProvider):
    """Provider that instantiates mega-dflash's ``DFlashGPTModel``."""

    dflash_num_layers: int | None = None
    dflash_num_draft_tokens: int = 2
    dflash_target_layer_indices: list[int] = field(default_factory=list)
    dflash_loss_scaling_factor: float = 1.0
    dflash_use_repeated_layer: bool = False
    dflash_detach_hidden_states: bool = False
    dflash_mask_token_id: int | None = None
    dflash_freeze_target: bool = False

    dspark_markov_rank: int | None = None
    dspark_loss_decay_gamma: float | None = 4.0
    dspark_l1_loss_alpha: float = 0.9
    dspark_confidence_head_alpha: float = 1.0
    dspark_confidence_loss_scale: float | None = None

    def _build_dflash_block_spec(self, transformer_layer_spec: ModuleSpec, vp_stage: int | None) -> ModuleSpec | None:
        if not self.dflash_num_layers:
            return None

        _DFlashConfig, _DFlashGPTModel, mtp_to_dflash_spec, dflash_spec_for_backend = _load_dflash_components()
        del _DFlashConfig, _DFlashGPTModel

        if getattr(self, "experimental_attention_variant", None) is not None:
            from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
                get_transformer_layer_with_experimental_attention_variant_spec,
            )

            layer_specs = get_transformer_layer_with_experimental_attention_variant_spec(config=self)
            dflash_base_layer_spec = layer_specs[-1]
        elif hasattr(transformer_layer_spec, "layer_specs") and len(transformer_layer_spec.layer_specs) == 0:
            from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_layer_specs

            decoder_layer_specs = get_gpt_decoder_layer_specs(
                self,
                use_transformer_engine=True,
                normalization=self.normalization,
                qk_l2_norm=self.qk_l2_norm,
                vp_stage=vp_stage,
            )
            dflash_base_layer_spec = decoder_layer_specs[-1]
        elif hasattr(transformer_layer_spec, "layer_specs"):
            dflash_base_layer_spec = transformer_layer_spec.layer_specs[-1]
        else:
            dflash_base_layer_spec = transformer_layer_spec

        dflash_layer_spec = mtp_to_dflash_spec(dflash_base_layer_spec, self, self.dflash_num_draft_tokens)
        backend = _te_spec_provider() if self.transformer_impl != "local" else LocalSpecProvider()
        return dflash_spec_for_backend(dflash_layer_spec, backend)

    def _build_dflash_config(self):
        DFlashConfig, _DFlashGPTModel, _mtp_to_dflash_spec, _dflash_spec_for_backend = _load_dflash_components()
        del _DFlashGPTModel, _mtp_to_dflash_spec, _dflash_spec_for_backend

        return DFlashConfig(
            num_layers=self.dflash_num_layers or 0,
            num_draft_tokens=self.dflash_num_draft_tokens,
            target_layer_indices=list(self.dflash_target_layer_indices),
            loss_scaling_factor=self.dflash_loss_scaling_factor,
            use_repeated_layer=self.dflash_use_repeated_layer,
            detach_hidden_states=self.dflash_detach_hidden_states,
            mask_token_id=self.dflash_mask_token_id,
            freeze_target=self.dflash_freeze_target,
            dspark_markov_rank=self.dspark_markov_rank,
            dspark_loss_decay_gamma=self.dspark_loss_decay_gamma,
            dspark_l1_loss_alpha=self.dspark_l1_loss_alpha,
            dspark_confidence_head_alpha=self.dspark_confidence_head_alpha,
            dspark_confidence_loss_scale=self.dspark_confidence_loss_scale,
        )

    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        """Instantiate ``DFlashGPTModel`` through Bridge's standard GPT provider path."""
        transformer_layer_spec = self.transformer_layer_spec
        if not isinstance(transformer_layer_spec, ModuleSpec):
            if "vp_stage" in inspect.signature(transformer_layer_spec).parameters:
                transformer_layer_spec = transformer_layer_spec(self, vp_stage=vp_stage)
            else:
                transformer_layer_spec = transformer_layer_spec(self)

        _DFlashConfig, DFlashGPTModel, _mtp_to_dflash_spec, _dflash_spec_for_backend = _load_dflash_components()
        del _DFlashConfig, _mtp_to_dflash_spec, _dflash_spec_for_backend

        dflash_config = self._build_dflash_config()
        dflash_block_spec = self._build_dflash_block_spec(transformer_layer_spec, vp_stage)

        original_model_cls = gpt_provider.MCoreGPTModel
        original_mtp_block_spec = gpt_provider.mtp_block_spec

        def dflash_model_factory(config, **kwargs):
            kwargs["mtp_block_spec"] = None
            return DFlashGPTModel(
                config=config,
                dflash_config=dflash_config,
                dflash_block_spec=dflash_block_spec,
                **kwargs,
            )

        gpt_provider.MCoreGPTModel = dflash_model_factory
        gpt_provider.mtp_block_spec = lambda config, vp_stage=None: None
        try:
            return super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
        finally:
            gpt_provider.MCoreGPTModel = original_model_cls
            gpt_provider.mtp_block_spec = original_mtp_block_spec
