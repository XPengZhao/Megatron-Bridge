#!/usr/bin/env python3
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

"""Convert DeepSeek-V4 DFlash/DSpark checkpoints between HF and Megatron."""

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import torch

from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.models.decorators import torchrun_main
from megatron.bridge.models.deepseek.dflash_bridge import DeepSeekV4DFlashBridge
from megatron.bridge.models.deepseek.dflash_provider import DFlashModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.hf_pretrained.utils import is_safe_repo
from megatron.bridge.utils.common_utils import print_rank_0


DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class DeepSeekV4DFlashAutoBridge(AutoBridge):
    """AutoBridge wrapper that forces the DeepSeek-V4 DFlash bridge."""

    @property
    def _model_bridge(self):
        bridge = DeepSeekV4DFlashBridge()
        bridge.hf_pretrained = self.hf_pretrained
        bridge.hf_config = self.hf_pretrained.config
        bridge.export_weight_dtype = self.export_weight_dtype
        return bridge


def _parse_dtype(name: str) -> torch.dtype:
    if name not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype '{name}'. Choose from {list(DTYPE_MAP)}.")
    return DTYPE_MAP[name]


def _check_distributed() -> None:
    if os.environ.get("WORLD_SIZE") is None:
        print_rank_0("This script must be launched with torchrun or srun. Example:")
        print_rank_0(f"  torchrun --nproc_per_node <gpus> {sys.argv[0]} --hf-model <path> --megatron-path <path>")
        sys.exit(1)


def _ensure_distributed_initialized(timeout_minutes: int | None) -> None:
    _check_distributed()
    if timeout_minutes is None or torch.distributed.is_initialized():
        return

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    torch.distributed.init_process_group(
        "nccl",
        timeout=datetime.timedelta(minutes=timeout_minutes),
    )


def _create_bridge(
    hf_model: str,
    *,
    dtype: torch.dtype,
    trust_remote_code: bool,
) -> DeepSeekV4DFlashAutoBridge:
    hf_pretrained = PreTrainedCausalLM.from_pretrained(
        hf_model,
        trust_remote_code=is_safe_repo(trust_remote_code=trust_remote_code, hf_path=hf_model),
        torch_dtype=dtype,
    )
    return DeepSeekV4DFlashAutoBridge(hf_pretrained)


def _get_source_config(hf_model: str, normalized_config: dict[str, object]) -> dict[str, object]:
    """Read the raw local config so custom DSv4 fields survive HF normalization."""
    source_config_path = Path(hf_model) / "config.json"
    if not source_config_path.is_file():
        return normalized_config

    with source_config_path.open() as config_file:
        return json.load(config_file)


def _fix_exported_config(hf_path: str, source_config: dict[str, object]) -> None:
    """Keep the exported DSv4 config compatible with vLLM."""
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return

    config_path = Path(hf_path) / "config.json"
    with config_path.open() as config_file:
        exported_config = json.load(config_file)

    # Training does not change the model architecture. Restore raw source
    # fields that Transformers may discard while normalizing and resaving.
    exported_config.update(source_config)

    # Hash routing is fully described by num_hash_layers. Transformers' generic
    # layer-type validator does not recognize the redundant "hash_moe" value.
    exported_config.pop("mlp_layer_types", None)

    with config_path.open("w") as config_file:
        json.dump(exported_config, config_file, indent=2, sort_keys=True)
        config_file.write("\n")


def _prepare_model_provider(
    bridge: DeepSeekV4DFlashAutoBridge,
    *,
    load_weights: bool,
    tp: int,
    pp: int,
    ep: int,
    etp: int,
    dtype: torch.dtype,
) -> tuple[DFlashModelProvider, list[list[str]] | None]:
    model_provider = bridge.to_megatron_provider(load_weights=load_weights)
    model_provider.tensor_model_parallel_size = tp
    model_provider.pipeline_model_parallel_size = pp
    model_provider.expert_model_parallel_size = ep
    model_provider.expert_tensor_parallel_size = etp
    model_provider.pipeline_dtype = dtype
    model_provider.params_dtype = dtype

    if pp > 1:
        num_layers = bridge.hf_pretrained.config.num_hidden_layers
        num_dflash_layers = model_provider.dflash_num_layers or 0
        model_provider.pipeline_model_parallel_layout = bridge._model_bridge.generate_pipeline_layout(
            num_layers,
            pp,
            num_dflash_layers,
        )
        print_rank_0(
            f"  Auto-generated pipeline layout for PP={pp} "
            f"({num_layers} layers, {num_dflash_layers} DFlash layers)"
        )

    resolved_pp_layout = model_provider.pipeline_model_parallel_layout if pp > 1 else None
    model_provider.finalize()
    model_provider.initialize_model_parallel(seed=0)
    return model_provider, resolved_pp_layout if isinstance(resolved_pp_layout, list) else None


@torchrun_main
def import_hf_to_megatron(
    hf_model: str,
    megatron_path: str,
    tp: int = 1,
    pp: int = 1,
    ep: int = 1,
    etp: int = 1,
    torch_dtype: str = "bfloat16",
    trust_remote_code: bool = False,
    distributed_timeout_minutes: int | None = None,
) -> None:
    """Import a DeepSeek-V4 DFlash HF checkpoint and save a Megatron checkpoint."""
    _ensure_distributed_initialized(distributed_timeout_minutes)
    dtype = _parse_dtype(torch_dtype)

    print_rank_0(f"Importing DeepSeek-V4 DFlash: {hf_model} -> {megatron_path}")
    print_rank_0(f"  TP={tp}  PP={pp}  EP={ep}  ETP={etp}  dtype={torch_dtype}")

    bridge = _create_bridge(hf_model, dtype=dtype, trust_remote_code=trust_remote_code)
    model_provider, _ = _prepare_model_provider(
        bridge,
        load_weights=True,
        tp=tp,
        pp=pp,
        ep=ep,
        etp=etp,
        dtype=dtype,
    )
    megatron_model = model_provider.provide_distributed_model(wrap_with_ddp=False)

    hf_tokenizer_kwargs = {}
    if hasattr(bridge._model_bridge, "get_hf_tokenizer_kwargs"):
        hf_tokenizer_kwargs = bridge._model_bridge.get_hf_tokenizer_kwargs() or {}
    if trust_remote_code:
        hf_tokenizer_kwargs["trust_remote_code"] = True

    print_rank_0(f"Saving Megatron checkpoint to: {megatron_path}")
    bridge.save_megatron_model(
        megatron_model,
        megatron_path,
        hf_tokenizer_path=hf_model,
        hf_tokenizer_kwargs=hf_tokenizer_kwargs,
    )
    print_rank_0(f"Import complete: {megatron_path}")


@torchrun_main
def export_megatron_to_hf(
    hf_model: str,
    megatron_path: str,
    hf_path: str,
    tp: int = 1,
    pp: int = 1,
    ep: int = 1,
    etp: int = 1,
    torch_dtype: str = "bfloat16",
    trust_remote_code: bool = False,
    strict: bool = True,
    show_progress: bool = True,
    distributed_save: bool = False,
    save_every_n_ranks: int = 1,
    distributed_timeout_minutes: int | None = None,
) -> None:
    """Export a distributed Megatron DFlash checkpoint to HF DSpark format."""
    _ensure_distributed_initialized(distributed_timeout_minutes)
    dtype = _parse_dtype(torch_dtype)

    print_rank_0(f"Exporting DeepSeek-V4 DFlash: {megatron_path} -> {hf_path}")
    print_rank_0(f"  TP={tp}  PP={pp}  EP={ep}  ETP={etp}  dtype={torch_dtype}")
    print_rank_0(f"  distributed_save={distributed_save}  save_every_n_ranks={save_every_n_ranks}")

    bridge = _create_bridge(hf_model, dtype=dtype, trust_remote_code=trust_remote_code)
    source_config = _get_source_config(hf_model, bridge.hf_pretrained.config.to_dict())
    _, resolved_pp_layout = _prepare_model_provider(
        bridge,
        load_weights=False,
        tp=tp,
        pp=pp,
        ep=ep,
        etp=etp,
        dtype=dtype,
    )

    mp_overrides = {
        "tensor_model_parallel_size": tp,
        "pipeline_model_parallel_size": pp,
        "expert_model_parallel_size": ep,
        "expert_tensor_parallel_size": etp,
        "pipeline_dtype": dtype,
        "params_dtype": dtype,
    }
    if resolved_pp_layout is not None:
        mp_overrides["pipeline_model_parallel_layout"] = resolved_pp_layout

    print_rank_0(f"Loading Megatron checkpoint from: {megatron_path}")
    megatron_model = bridge.load_megatron_model(
        megatron_path,
        mp_overrides=mp_overrides,
        wrap_with_ddp=False,
    )

    print_rank_0(f"Saving HuggingFace checkpoint to: {hf_path}")
    bridge.save_hf_pretrained(
        megatron_model,
        hf_path,
        source_path=hf_model,
        strict=strict,
        show_progress=show_progress,
        distributed_save=distributed_save,
        save_every_n_ranks=save_every_n_ranks,
    )
    _fix_exported_config(hf_path, source_config)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    print_rank_0(f"Export complete: {hf_path}")


@torchrun_main
def roundtrip_hf_to_hf(
    hf_model: str,
    hf_path: str,
    tp: int = 1,
    pp: int = 1,
    ep: int = 1,
    etp: int = 1,
    torch_dtype: str = "bfloat16",
    trust_remote_code: bool = False,
    strict: bool = True,
    show_progress: bool = True,
    distributed_save: bool = False,
    save_every_n_ranks: int = 1,
    distributed_timeout_minutes: int | None = None,
) -> None:
    """Round-trip HF weights through an in-memory DFlash model without DistCP."""
    _ensure_distributed_initialized(distributed_timeout_minutes)
    dtype = _parse_dtype(torch_dtype)

    print_rank_0(f"Round-tripping DeepSeek-V4 DFlash: {hf_model} -> {hf_path}")
    print_rank_0(f"  TP={tp}  PP={pp}  EP={ep}  ETP={etp}  dtype={torch_dtype}")

    bridge = _create_bridge(hf_model, dtype=dtype, trust_remote_code=trust_remote_code)
    source_config = _get_source_config(hf_model, bridge.hf_pretrained.config.to_dict())
    model_provider, _ = _prepare_model_provider(
        bridge,
        load_weights=True,
        tp=tp,
        pp=pp,
        ep=ep,
        etp=etp,
        dtype=dtype,
    )
    megatron_model = model_provider.provide_distributed_model(wrap_with_ddp=False)

    bridge.save_hf_pretrained(
        megatron_model,
        hf_path,
        source_path=hf_model,
        strict=strict,
        show_progress=show_progress,
        distributed_save=distributed_save,
        save_every_n_ranks=save_every_n_ranks,
    )
    _fix_exported_config(hf_path, source_config)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    print_rank_0(f"Round-trip complete: {hf_path}")


def main() -> None:
    """Parse CLI arguments and dispatch a DeepSeek-V4 DFlash conversion."""
    parser = argparse.ArgumentParser(
        description="Convert DeepSeek-V4 DFlash/DSpark checkpoints between HuggingFace and Megatron formats"
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("import", "export", "roundtrip"),
        default="import",
        help="Conversion direction (default: import for backward compatibility)",
    )
    parser.add_argument("--hf-model", required=True, help="HuggingFace model ID or local path")
    parser.add_argument("--megatron-path", help="Megatron checkpoint directory")
    parser.add_argument("--hf-path", help="Directory to save the exported HuggingFace checkpoint")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism size")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallelism size")
    parser.add_argument("--ep", type=int, default=1, help="Expert parallelism size")
    parser.add_argument("--etp", type=int, default=1, help="Expert tensor parallelism size")
    parser.add_argument(
        "--torch-dtype",
        choices=list(DTYPE_MAP),
        default="bfloat16",
        help="Model precision (default: bfloat16)",
    )
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow custom model code execution")
    parser.add_argument("--no-progress", action="store_true", help="Disable the export progress bar")
    parser.add_argument("--not-strict", action="store_true", help="Allow source and target keys to differ on export")
    parser.add_argument(
        "--distributed-save",
        action="store_true",
        help="Save HF shards from multiple ranks to reduce rank-0 memory pressure",
    )
    parser.add_argument(
        "--save-every-n-ranks",
        type=int,
        default=1,
        help="Only every N-th rank writes HF shards when distributed saving is enabled",
    )
    parser.add_argument(
        "--distributed-timeout-minutes",
        type=int,
        default=None,
        help="Initialize the distributed process group with this timeout before model setup",
    )
    args = parser.parse_args()

    if args.command == "import":
        if args.megatron_path is None:
            parser.error("--megatron-path is required for import")
        import_hf_to_megatron(
            hf_model=args.hf_model,
            megatron_path=args.megatron_path,
            tp=args.tp,
            pp=args.pp,
            ep=args.ep,
            etp=args.etp,
            torch_dtype=args.torch_dtype,
            trust_remote_code=args.trust_remote_code,
            distributed_timeout_minutes=args.distributed_timeout_minutes,
        )
        return

    if args.hf_path is None:
        parser.error(f"--hf-path is required for {args.command}")
    if args.command == "roundtrip":
        roundtrip_hf_to_hf(
            hf_model=args.hf_model,
            hf_path=args.hf_path,
            tp=args.tp,
            pp=args.pp,
            ep=args.ep,
            etp=args.etp,
            torch_dtype=args.torch_dtype,
            trust_remote_code=args.trust_remote_code,
            strict=not args.not_strict,
            show_progress=not args.no_progress,
            distributed_save=args.distributed_save,
            save_every_n_ranks=args.save_every_n_ranks,
            distributed_timeout_minutes=args.distributed_timeout_minutes,
        )
        return

    if args.megatron_path is None:
        parser.error("--megatron-path is required for export")
    export_megatron_to_hf(
        hf_model=args.hf_model,
        megatron_path=args.megatron_path,
        hf_path=args.hf_path,
        tp=args.tp,
        pp=args.pp,
        ep=args.ep,
        etp=args.etp,
        torch_dtype=args.torch_dtype,
        trust_remote_code=args.trust_remote_code,
        strict=not args.not_strict,
        show_progress=not args.no_progress,
        distributed_save=args.distributed_save,
        save_every_n_ranks=args.save_every_n_ranks,
        distributed_timeout_minutes=args.distributed_timeout_minutes,
    )


if __name__ == "__main__":
    main()
