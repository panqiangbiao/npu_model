# Tiny LLM NPU Profiling Demo

## 目的

用一个可控、极小但包含 Transformer 关键算子的模型，验证手机端以下完整链路：

```text
MNN Express 构建模型
  -> MNN USER_0 映射 HiAI 整图
  -> BuildIRModel 生成 OM
  -> NNRT 加载 OM
  -> HIAI_OM_TYPE_PROFILING
  -> 输出 model.csv / op.csv
```

该模型是固定随机权重的算子 Profiling fixture，用于验证执行链路和采集算子耗时，不用于评价语言生成质量。

## 模型结构

固定规格：`sequence=4`、`hidden=32`、`heads=1`、`FFN=64`、`vocab=64`。

```text
input_ids
  -> token embedding (GatherV2)
  -> LayerNorm
  -> Q/K/V projection
  -> QK^T + causal mask + Softmax
  -> Attention * V + output projection + residual
  -> LayerNorm
  -> FFN up + ReLU + FFN down + residual
  -> LayerNorm
  -> LM Head
  -> logits
```

核心代码：

- `entry/src/main/cpp/tiny_llm_profiler.cpp`：构建 MNN 模型、触发 USER_0 构图、读取导出的 OM、调用 NNRT Profiling。
- `entry/src/main/cpp/cann_profile_self_test.cpp`：设置 `HIAI_OM_TYPE_PROFILING`，创建 NNRT Executor 并重复执行。
- `entry/src/main/ets/pages/Index.ets`：`Tiny LLM Profiling` 按钮及结果显示。

## 操作

1. 编译并安装应用。
2. 启动“美颜实验室”。
3. 点击 `Tiny LLM Profiling`。
4. 查看界面结果或过滤日志关键字 `PQB:TINY_LLM`、`PQB:CANN_PROFILE`。
5. CSV 位于应用沙箱的 `files/tiny_llm_profile` 目录。

## 2026-09-07 真机结果

- MNN：53,796 bytes。
- USER_0 导出 OM：282,325 bytes。
- NNRT Profiling：50 次全部成功。
- NNRT inference 平均：534.7 us，最小 447 us，最大 686 us。
- 生成结果：固定输入 `[1, 7, 11, 23]` 得到 next token `54`。
- `_op.csv` 中所有模型算子的 `device type` 均为 `npu_aicore`。

主要算子平均耗时：

| 算子类型 | 样本数 | 平均耗时 |
|---|---:|---:|
| BatchMatMulV2 | 450 | 16.7 us |
| GatherV2D | 50 | 8.9 us |
| LayerNorm | 150 | 2.2 us |
| Softmax | 50 | 2.0 us |
| Reshape | 350 | 2.0 us |
| TransData | 300 | 2.0 us |
| Add | 150 | 1.6 us |
| Activation | 50 | 1.3 us |

本结果说明手机 CANN Kit 的 OM Profiling 路径能够返回 Transformer 算子名称、开始时间、单算子耗时和执行设备。它仍不等价于昇腾服务器 `msprof`：当前 CSV 没有 Cube/Vector 利用率、L2/DDR 字节、流水阻塞和算子内部核级指标。
