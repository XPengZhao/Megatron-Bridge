# DSpark / DFlash 权重转换说明

## 目标

把 DeepSeek-V4-Flash-DSpark 的 HF checkpoint 转成 mega-dflash 可以加载的 Megatron distributed checkpoint。

当前做法：

1. 复用 `DeepSeekV4Bridge` 转换 DeepSeek-V4 主干权重。
2. 新增 `DeepSeekV4DFlashBridge`，只补 DSpark / DFlash 相关的 `mtp.* -> dflash.*` 映射。
3. 新增 `DFlashModelProvider`，在 Bridge 构造模型时实例化 mega-dflash 的 `DFlashGPTModel`。

## HF 转 Megatron

```bash
export PYTHONPATH=/public/workspace/dspark/megatron-bridge/src:\
/public/workspace/dspark/megatron-bridge/3rdparty/Megatron-LM:\
/public/workspace/dspark/mega-dflash:\
$PYTHONPATH

uv run python -m torch.distributed.run \
  --nproc_per_node=8 \
  megatron-bridge/examples/conversion/convert_deepseek_v4_dflash_multi_gpu.py import \
  --hf-model /public/llm_models/DeepSeek/DeepSeek-V4-Flash-DSpark \
  --megatron-path /public/workspace/dspark/checkpoints/deepseek_v4_flash_dflash_tp1_ep8_v3 \
  --tp 1 \
  --pp 1 \
  --ep 8 \
  --etp 1 \
  --torch-dtype bfloat16 \
  --trust-remote-code \
  --distributed-timeout-minutes 60 \
  2>&1 | tee convert.log
```

不写方向时默认执行 `import`，建议显式指定转换方向。

## Megatron 转 HF

反向转换使用原始 HF 目录作为配置、tokenizer 和权重分片模板：

```bash
python -m torch.distributed.run \
  --nproc_per_node=8 \
  megatron-bridge/examples/conversion/convert_deepseek_v4_dflash_multi_gpu.py export \
  --hf-model /public/llm_models/DeepSeek/DeepSeek-V4-Flash-DSpark \
  --megatron-path /public/workspace/dspark/checkpoints/deepseek_v4_flash_dflash_tp1_ep8_v3 \
  --hf-path /public/workspace/dspark/checkpoints/deepseek_v4_flash_dflash_hf \
  --tp 1 \
  --pp 1 \
  --ep 8 \
  --etp 1 \
  --torch-dtype bfloat16 \
  --trust-remote-code \
  --distributed-save \
  --distributed-timeout-minutes 60 \
  2>&1 | tee export.log
```

导出会把 Megatron 中的 BF16 权重按原 HF checkpoint 的 scale 布局重新量化为 FP8/MXFP4，并保留 DSpark 配置和模型代码。首次验证建议保持严格模式；只有排查 key 差异时才加 `--not-strict`。

## 相关新增和修改文件

```text
src/megatron/bridge/models/deepseek/dflash_bridge.py
src/megatron/bridge/models/deepseek/dflash_provider.py
examples/conversion/convert_deepseek_v4_dflash_multi_gpu.py
3rdparty/mega-dflash/dflash/core/transformer/dflash_block.py
```

## Provider 做了什么

`DFlashModelProvider` 继承 DSv4 的 `MLAModelProvider` 配置，然后在 `provide()` 里构造 mega-dflash 的 `DFlashGPTModel`。

关键点：

```text
DeepSeekV4 HF config
  -> DeepSeekV4Bridge.provider_bridge()
  -> DFlashModelProvider
  -> DFlashGPTModel
```

当前从 HF config 读取的 DSpark / DFlash 字段包括：

```text
dspark_target_layer_ids -> dflash_target_layer_indices
dspark_markov_rank     -> DFlashConfig.dspark_markov_rank
dspark_block_size      -> DFlashConfig.num_draft_tokens
compress_ratios        -> csa_compress_ratios
```

对当前 DSpark checkpoint，期望看到类似：

```text
DFlash provider config:
num_layers=3,
target_layer_indices=[40, 41, 42],
num_draft_tokens=5,
dspark_markov_rank=256,
csa_compress_ratios_len=46
```

## 主干权重

DeepSeek-V4 主干不重新写映射，直接复用 `DeepSeekV4Bridge.mapping_registry()`。

也就是说这些 DSv4 权重仍走原来的 DeepSeek-V4 转换逻辑：

```text
embedding
decoder.layers.*.self_attention.*
decoder.layers.*.mlp.*
decoder.layers.*.self_attention_hyper_connection.*
decoder.layers.*.mlp_hyper_connection.*
final_layernorm
output_layer
```

## DFlash 全局权重映射

```text
HF checkpoint                         Megatron / mega-dflash
------------------------------------------------------------
mtp.0.main_proj.weight             -> dflash.main_proj.weight
mtp.0.main_norm.weight             -> dflash.main_norm.weight
mtp.{last}.norm.weight             -> dflash.final_norm.weight
mtp.{last}.hc_head_fn              -> dflash.hc_head_fn
mtp.{last}.hc_head_base            -> dflash.hc_head_base
mtp.{last}.hc_head_scale           -> dflash.hc_head_scale
```

`{last}` 表示 `num_dflash_layers - 1`。当前 DSpark checkpoint 里 `num_dflash_layers=3`，所以 `{last}=2`。

## DSpark 头映射

```text
HF checkpoint                                      Megatron / mega-dflash
-------------------------------------------------------------------------
mtp.{last}.markov_head.markov_w1.weight        -> dflash.markov_embedding.word_embeddings.weight
mtp.{last}.markov_head.markov_w2.weight        -> dflash.markov_lm_head.weight
mtp.{last}.confidence_head.proj.weight         -> dflash.confidence_head.weight
```

这几个是 DSpark 头的关键权重。转换后 checkpoint metadata 里应能看到：

```text
dflash.markov_embedding.word_embeddings.weight
dflash.markov_lm_head.weight
dflash.confidence_head.weight
```

## 每个 DFlash layer 映射

对每个 `i in [0, num_dflash_layers)`：

```text
HF checkpoint                                  Megatron / mega-dflash
-------------------------------------------------------------------------
mtp.{i}.attn_norm.weight                    -> dflash.layers.{i}.input_layernorm.weight
mtp.{i}.ffn_norm.weight                     -> dflash.layers.{i}.pre_mlp_layernorm.weight

mtp.{i}.attn.wq_a.weight                    -> dflash.layers.{i}.self_attention.linear_q_down_proj.weight
mtp.{i}.attn.q_norm.weight                  -> dflash.layers.{i}.self_attention.q_layernorm.weight
mtp.{i}.attn.wq_b.weight                    -> dflash.layers.{i}.self_attention.linear_q_up_proj.weight
mtp.{i}.attn.wkv.weight                     -> dflash.layers.{i}.self_attention.linear_kv_proj.weight
mtp.{i}.attn.kv_norm.weight                 -> dflash.layers.{i}.self_attention.kv_layernorm.weight
mtp.{i}.attn.wo_a.weight                    -> dflash.layers.{i}.self_attention.linear_o_group_proj
mtp.{i}.attn.wo_b.weight                    -> dflash.layers.{i}.self_attention.linear_proj.weight
mtp.{i}.attn.attn_sink                      -> dflash.layers.{i}.self_attention.core_attention.attn_sink

mtp.{i}.ffn.gate.weight                     -> dflash.layers.{i}.mlp.router.weight
mtp.{i}.ffn.gate.bias                       -> dflash.layers.{i}.mlp.router.expert_bias
mtp.{i}.ffn.gate.tid2eid                    -> dflash.layers.{i}.mlp.router.tid2eid
```

MoE expert：

```text
mtp.{i}.ffn.experts.*.w1.weight \
mtp.{i}.ffn.experts.*.w3.weight              -> dflash.layers.{i}.mlp.experts.linear_fc1.weight*

mtp.{i}.ffn.experts.*.w2.weight              -> dflash.layers.{i}.mlp.experts.linear_fc2.weight*
```

Shared expert：

```text
mtp.{i}.ffn.shared_experts.w1.weight \
mtp.{i}.ffn.shared_experts.w3.weight         -> dflash.layers.{i}.mlp.shared_experts.linear_fc1.weight

mtp.{i}.ffn.shared_experts.w2.weight         -> dflash.layers.{i}.mlp.shared_experts.linear_fc2.weight
```

HyperConnection：

```text
mtp.{i}.hc_attn_fn                          -> dflash.layers.{i}.self_attention_hyper_connection.mapping_proj.weight
mtp.{i}.hc_attn_base                        -> dflash.layers.{i}.self_attention_hyper_connection.bias
mtp.{i}.hc_attn_scale                       -> dflash.layers.{i}.self_attention_hyper_connection.alpha_pre/post/res

mtp.{i}.hc_ffn_fn                           -> dflash.layers.{i}.mlp_hyper_connection.mapping_proj.weight
mtp.{i}.hc_ffn_base                         -> dflash.layers.{i}.mlp_hyper_connection.bias
mtp.{i}.hc_ffn_scale                        -> dflash.layers.{i}.mlp_hyper_connection.alpha_pre/post/res
```

## DistCP expert 分片

`DFlashBlock.layers` 是 `nn.ModuleList`。只使用 `MegatronModule` 的默认
`sharded_state_dict()` 会把 ModuleList 内的 expert 当作普通 tensor 保存，丢失 EP
全局 expert 轴。`DFlashBlock` 必须逐层调用 `layer.sharded_state_dict()`。

EP=8、256 experts 时，错误 metadata 形态是：

```text
dflash.layers.0.mlp.experts.linear_fc1.weight0
size=[4096, 4096]
```

修复后应与 decoder 的 grouped expert 一样包含全局 expert 轴：

```text
dflash.layers.0.mlp.experts.experts.linear_fc1.weight
size=[256, 4096, 4096]
```

旧 checkpoint 已经丢失 expert 分片信息，修改代码后必须从原始 HF 权重重新转换。

## 当前验证结论

修复 DistCP 分片并重新转换后，以下有效权重的 round-trip 误差为零：

```text
layers.0.attn.wq_a.weight
layers.0.ffn.experts.0.w1.weight
mtp.0.main_proj.weight
mtp.0/1/2.attn.wq_a.weight
mtp.0.ffn.experts.0.w1.weight
```

HC 参数原始 dtype 为 FP32，经过 Megatron BF16 后相对误差约为 `1e-3`，属于 dtype
舍入误差。重新加载后的首步 loss 为：

```text
lm=1.317, mtp_1=1.338, mtp_2=1.791, mtp_3=2.347, mtp_4=2.801, mtp_5=3.148
```

该结果说明主干和 DFlash 权重已正确加载，后续 draft token 的 loss 平滑递增。

## 反向转换验证

```text
检查 export.log 中没有 No mapping found、missing、unexpected 或 could not be saved
检查导出 config.json 保留 dspark_block_size=5、dspark_target_layer_ids 和 dspark_markov_rank
检查 model.safetensors.index.json 中存在 mtp.0/1/2、markov_head 和 confidence_head
检查主体和 DFlash attention、main_proj、routed expert 的有效权重 rel_l2=0
```
