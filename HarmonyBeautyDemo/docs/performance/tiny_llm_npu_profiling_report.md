# 极小语言模型手机 NPU Profiling 验证报告

## 1. 结论摘要

本次实验已经打通以下完整链路：

```text
Tiny Transformer
  -> MNN USER_0 整图适配
  -> HiAI BuildIRModel 生成 OM
  -> NNRT 加载 Profiling OM
  -> NPU 连续推理
  -> 导出模型级和算子级 CSV
```

核心结论：

- 手机上可以对具备 Transformer 结构的 OM 开启 CANN Kit Profiling。
- 能获得模型加载、推理、卸载耗时，以及算子名称、开始时间、耗时和执行设备。
- Embedding、Attention、FFN、LM Head 均被识别，采集到的模型算子全部运行在 `npu_aicore`。
- 50 次 NNRT 推理平均耗时为 **534.7 us**，P50 为 **531 us**，P95 为 **612 us**。
- 当前能力足以分析“哪个算子慢、算子是否落在 NPU、算子执行顺序”，但还不能直接回答 Cube/Vector 利用率、DDR/L2 流量和算子内部流水阻塞。

## 2. 实验目的

此前手机侧只能通过 trace、频率和进程资源观察推理整体行为，缺少算子层证据。本实验构建一个参数量极小、结构可控的 Transformer，用于确认手机 CANN Kit 是否能够输出大语言模型典型算子的运行时间。

该模型是固定随机权重的 Profiling fixture，目标是验证执行和维测链路，不用于评价自然语言生成质量。

## 3. 模型与环境

### 3.1 模型规格

| 参数 | 配置 |
|---|---:|
| Sequence length | 4 |
| Hidden size | 32 |
| Attention heads | 1 |
| FFN intermediate size | 64 |
| Vocabulary size | 64 |
| Transformer block | 1 |
| 权重 | 固定种子随机 FP32 |

模型流程：

```text
input_ids [1,4]
  -> Token Embedding / GatherV2
  -> LayerNorm
  -> Q/K/V Projection
  -> QK^T
  -> Scale + Causal Mask
  -> Softmax
  -> Attention * V
  -> Output Projection + Residual
  -> LayerNorm
  -> FFN Up + ReLU + FFN Down + Residual
  -> LayerNorm
  -> LM Head
  -> logits [1,4,64]
```

### 3.2 软件链路

| 层级 | 实现 |
|---|---|
| 模型生成 | MNN Express |
| 三方框架 NPU 后端 | MNN `MNN_FORWARD_USER_0` |
| 在线整图编译 | `HiaiIrBuild::BuildIRModel` |
| 离线模型 | OM |
| 加载与执行 | OpenHarmony NNRT |
| Profiling 开关 | `HMS_HiAIOptions_SetOmOptions(..., HIAI_OM_TYPE_PROFILING, ...)` |
| 执行设备 | `HIAI_F` / NPU |

## 4. 实现过程

1. 使用 MNN Express 在应用内生成 Tiny Transformer MNN Buffer。
2. 创建 `MNN_FORWARD_USER_0` Session，将整图转换为 HiAI Graph。
3. 调用 `BuildIRModel`，导出 `mnn_user0_outputs_1.om`。
4. 首次通过 MNN-U0 执行固定输入 `[1, 7, 11, 23]`，输出 next token `54`。
5. 使用 NNRT 重新加载该 OM，并设置 `HIAI_OM_TYPE_PROFILING`。
6. 固定形状连续执行 50 次，输出 `_model.csv` 和 `_op.csv`。

最初实现使用 MNN `Gather`，真机日志显示 USER_0 未注册该算子。将 Embedding 改为语义等价的 `GatherV2(axis=0)` 后，整图构建、推理和 Profiling 全部成功。这也说明算子类型即使语义接近，仍需匹配后端已注册的具体 IR 映射。

## 5. 真机结果

### 5.1 模型级结果

| 指标 | 结果 |
|---|---:|
| MNN Buffer | 53,796 bytes |
| USER_0 导出 OM | 282,325 bytes |
| OM 加载时间 | 8,618 us |
| Profiling 运行次数 | 50 |
| 推理成功次数 | 50 |
| 平均推理时间 | **534.7 us** |
| P50 | 531 us |
| P90 | 590 us |
| P95 | 612 us |
| P99 | 686 us |
| 最小值 | 447 us |
| 最大值 | 686 us |
| 标准差 | 49.8 us |
| OM 卸载时间 | 3,018 us |

50 次推理无失败，最大值约为平均值的 1.28 倍。由于模型极小，调度、同步和固定开销在端到端耗时中的占比会明显高于真实大模型。

### 5.2 算子级结果

| 算子类型 | 每轮数量 | 50 轮样本数 | 平均单算子耗时 |
|---|---:|---:|---:|
| BatchMatMulV2 | 9 | 450 | **16.7 us** |
| GatherV2D | 1 | 50 | **8.9 us** |
| LayerNorm | 3 | 150 | 2.2 us |
| TransData | 6 | 300 | 2.0 us |
| Reshape | 7 | 350 | 2.0 us |
| Softmax | 1 | 50 | 2.0 us |
| Mul | 1 | 50 | 2.0 us |
| Add | 3 | 150 | 1.6 us |
| Activation | 1 | 50 | 1.3 us |

主要矩阵算子均被保留并可单独识别：

- `q_proj`、`k_proj`、`v_proj`
- `QK^T` 与 `Attention * V`
- Attention output projection
- FFN up projection 与 down projection
- `logits` / LM Head

所有记录的 `device type` 均为 `npu_aicore`，本次没有观察到 CPU 算子回退。

### 5.3 时间解释

`_op.csv` 中各算子记录按一轮简单求和约为 202.5 us，而 `_model.csv` 的端到端均值为 534.7 us。两者差异不能直接等同于框架开销，原因包括：

- Q/K/V 等算子可能存在并行或时间重叠，算子耗时不一定可以串行相加；
- 模型级时间包含提交、同步、输入输出 Tensor 和运行时调度；
- CSV 时间粒度和统计边界可能与 NNRT inference 区间不同。

后续应按 `start time` 重建单轮算子时间线，分析并行区间和未覆盖区间，再判断固定开销来源。

## 6. 当前能回答的问题

| 问题 | 本次是否可回答 | 证据 |
|---|---|---|
| Transformer 是否成功整图编译 | 是 | OM 成功导出 |
| 是否真实运行在 NPU | 是 | `device type=npu_aicore` |
| 包含哪些运行时算子 | 是 | `_op.csv` 算子名称 |
| 单算子运行时间 | 是 | `_op.csv total time(us)` |
| 模型加载、推理、卸载耗时 | 是 | `_model.csv` |
| 是否存在明显 CPU fallback | 本次可回答 | 未出现非 NPU device type |
| Cube/Vector 单元利用率 | 否 | 手机 CSV 未提供 |
| DDR/L2 访问字节和命中率 | 否 | 手机 CSV 未提供 |
| 算子内部 Stall 与流水效率 | 否 | 需要更底层 PMU/开发板工具 |
| KV Cache 行为 | 否 | 本模型未实现增量 KV Cache |

## 7. 限制与风险

- 模型为单层极小固定 Shape，不能代表 3B/MoE 模型的绝对性能。
- 权重未经训练，输出 token 只用于检查数据链路可重复。
- NNRT Profiling 重放阶段当前填充零输入；算子拓扑和耗时有效，但没有再次校验重放输出与 MNN 首次输出的一致性。
- 极小模型容易被框架固定开销主导，不适合直接推导大模型算力利用率。
- 目前只验证 FP32 路径，尚未覆盖 FP16、INT8/INT4、KV Cache 和动态 Shape。

## 8. 后续建议

1. 增加 `sequence=4/16/64/128` 分档，观察 Attention 随序列长度的变化。
2. 增加 `hidden=32/128/512` 分档，区分固定调度开销与矩阵计算开销。
3. 增加 FP16 或量化权重版本，对比 OM 大小、加载时间和 MatMul 耗时。
4. 重建单轮算子时间线，验证 Q/K/V 是否并行以及端到端未覆盖区间。
5. 增加 Prefill 与单 Token Decode 两种图，建立更接近真实 LLM 的 Profiling 基线。
6. 若要研究访存效率，将同一模型迁移到昇腾开发板，补充 Cube/Vector 利用率、DDR/L2 与流水 Stall 指标。

## 9. 代码与证据

- 模型构建与执行：`entry/src/main/cpp/tiny_llm_profiler.cpp`
- NNRT Profiling：`entry/src/main/cpp/cann_profile_self_test.cpp`
- 应用入口：`entry/src/main/ets/pages/Index.ets`
- 原始模型级数据：`docs/data/tiny_llm_profile_20260907/model.csv`
- 原始算子级数据：`docs/data/tiny_llm_profile_20260907/op.csv`

![Tiny LLM Profiling 真机结果](../assets/tiny_llm_profiling_result.jpeg)
