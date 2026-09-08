# TinyGPT2 CPU/NPU Profiling 对比报告

## 1. 目的与结论

在当前 HarmonyBeautyDemo 中，使用同一 TinyGPT2 固定 Shape 图和同一输入，分别验证 MNN CPU、MNN USER_0/NPU，并将 USER_0 导出的 OM 通过 NNRT 重新加载，开启 CANN Profiling。

本轮结论：

- 三条执行路径均运行成功，NNRT Profiling 成功产出模型级和算子级记录。
- 当前极小模型上，CPU 稳定态均值为 **0.648 ms**，MNN USER_0 为 **1.404 ms**；NPU 比 CPU 慢约 **2.17 倍**。
- 这不代表真实 LLM 上 CPU 优于 NPU。该模型 hidden size 仅为 2，计算量过小，NPU 提交、同步和格式转换等固定成本占主导。
- USER_0 输出 token 与 CPU 不一致。按本次要求暂不修复精度，因此本报告只说明路径与性能现状，不把 NPU 结果视为可发布的正确实现。

## 2. 测试对象与口径

| 项目 | 配置 |
|---|---|
| 开源模型 | `sshleifer/tiny-gpt2` |
| 图 | 固定 Prefill，sequence length=16，hidden size=2 |
| MNN 文件 | 635,484 bytes |
| USER_0 导出 OM | 2,260,001 bytes |
| 输入 | 同一份 `float32[1,16,2]` embeddings，128 bytes |
| MNN 测试 | CPU/U0 各预热 5 次，测量 `runSession` 50 次 |
| setup | `createFromBuffer + createSession`，与稳定态推理分开统计 |
| NNRT Profiling | `HIAI_F + HIAI_OM_TYPE_PROFILING`，相同输入运行 50 次 |

## 3. 运行流程

```text
CPU
MNN模型 -> createFromBuffer -> CPU Session -> 预热5次 -> runSession x50 -> 读取输出

NPU性能
MNN模型 -> USER_0 Session/BuildIRModel -> 导出OM -> 预热5次 -> runSession x50 -> 读取输出

NPU算子Profiling
导出OM -> NNRT离线模型Compilation -> HIAI_F/NPU -> 开启Profiling
      -> 复制同一输入 -> RunSync x50 -> model.csv + op.csv + 原始.prof
```

## 4. 性能结果

| 路径 | setup | 均值 | P50 | P95 | 最小 | 最大 |
|---|---:|---:|---:|---:|---:|---:|
| MNN CPU | 3.47 ms | **0.648 ms** | 0.616 ms | 0.719 ms | 0.598 ms | 1.632 ms |
| MNN USER_0 | 116.28 ms | **1.404 ms** | 1.350 ms | 1.768 ms | 1.208 ms | 1.906 ms |
| NNRT + CANN Profiling | 20.43 ms load | **1.546 ms** | 1.504 ms | 1.995 ms | 1.247 ms | 2.120 ms |

说明：NNRT 行来自 Profiling 模式，包含采集开销，不能代替非 Profiling 稳态性能；因此 CPU/U0 的公平性能比较使用前两行。USER_0 setup 包含在线建图/编译与 OM 导出，首次运行成本约比 CPU setup 高 33.5 倍。

## 5. NPU 算子画像

CANN 共记录 **4,300 条算子事件**，即每轮 86 条；执行设备全部为 `npu_aicore`，未观察到 CPU 算子 fallback。

| 算子类型 | 50轮样本数 | 累计时间 | 单次算子均值 |
|---|---:|---:|---:|
| MatMulV2 | 450 | 29.300 ms | 65.11 us |
| Reshape | 600 | 6.221 ms | 10.37 us |
| BatchMatMulV2 | 200 | 3.989 ms | 19.94 us |
| TransData | 700 | 3.483 ms | 4.98 us |
| Mul | 600 | 1.364 ms | 2.27 us |
| LayerNorm | 250 | 1.329 ms | 5.32 us |
| Add | 500 | 1.187 ms | 2.37 us |
| Permute | 400 | 1.048 ms | 2.62 us |

MatMulV2 与 BatchMatMulV2 合计约占算子累计时间的 **66.4%**。同时存在每轮 14 个 TransData，以及较多 Reshape/Permute；对于这种极小 Shape，它们的固定调度与数据布局成本不可忽略。算子累计时间可能受并行和 Profiling 记账方式影响，不能直接从模型端到端时间中相减推导框架开销。

## 6. 精度状态与边界

| 项目 | CPU | USER_0/NPU |
|---|---:|---:|
| 输出 token | 5087 | 16708 |
| PyTorch 参考 token | 5087 | 5087 |
| 当前判定 | 匹配参考 | **MISMATCH** |

中间输出最大绝对误差约为 `1.077/1.076/1.077`。本轮没有修改精度逻辑。性能数据可用于验证接入、建图、执行与 Profiling 链路，但不能作为该 USER_0 图已完成正确性适配的证明。

## 7. 可回答与不可回答

本次已经能回答：CPU/U0 的 setup 与稳定态耗时、NPU 是否命中、是否发生算子级 CPU fallback、NPU 算子类型与耗时分布。

本次不能回答：NPU Cube/Vector 利用率、DDR 带宽、片上缓存命中率、算子 stall 原因和能耗。手机 CANN Kit 当前导出的 CSV 没有这些昇腾开发环境中的微观计数器。

原始数据见 [`../data/tiny_gpt2_cpu_npu_20260908_120500/`](../data/tiny_gpt2_cpu_npu_20260908_120500/)。
