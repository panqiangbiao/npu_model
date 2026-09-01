# 原生 NPU 与 MNN USER_0 固定输入性能分析

## 1. 测试目的

区分以下两条华为 NPU 接入路径的真实差异：

```text
原生 NPU：.ms -> MindSpore Lite -> NNRT -> HiAI/驱动 -> NPU
MNN-U0： .mnn -> MNN USER_0/HiAI Backend -> GE IR -> BuildIRModel -> Process -> NPU
```

测试使用同一份固定人脸输入和同一份固定 68 点输入，每 500 ms 执行一轮，排除摄像头画面变化、ROI 变化和检测节流的影响。

## 2. 模型与正确性

| 项目 | 原生 NPU | MNN-U0 |
|---|---:|---:|
| 人脸模型文件 | `version-slim-320-nofusion.ms`，1,060,824 B | `gpu_face_detector.mnn`，1,041,656 B |
| 68 点模型文件 | `landmarks_68_pfld_nofusion.ms`，2,926,064 B | `gpu_landmarks_68.mnn`，2,914,444 B |
| 人脸最佳候选 | 4251 | 4251 |
| 人脸置信度 | 0.9980 | 0.9980 |
| 人脸框 IoU | 0.954 | 0.954 |
| 分数对范围 | 0.9969～1.0022 | 0.9969～1.0022 |
| 68 点最大绝对误差 | 0.00168 | 0.00168 |
| 连续运行漂移 | 0 | 0 |

两种格式不是同一个二进制，但模型大小、输入、输出和结果一致，可以排除“使用不同规模模型”或“降低精度换速度”造成的假差异。

## 3. 同版本 Trace 结果

每条路径均从 30 秒采集中取得 49 组完整样本。由于系统 trace buffer 只保留约 24.8 秒有效窗口，以下统计均使用实际完整样本。

| 阶段 | 原生 NPU 均值 | MNN-U0 均值 | 差异 |
|---|---:|---:|---:|
| 人脸完整调用 | 14.138 ms | 15.094 ms | U0 慢 6.8% |
| 68 点完整调用 | 5.219 ms | 2.820 ms | U0 快 46.0% |
| 单轮总耗时 | 31.461 ms | 28.682 ms | U0 快 8.8% |
| 单轮 P50 | 32.522 ms | 31.032 ms | U0 快 4.6% |
| 单轮 P90 | 35.730 ms | 32.195 ms | U0 快 9.9% |
| 单轮 P95 | 36.066 ms | 33.332 ms | U0 快 7.6% |

结论：U0 并非所有模型都更快。人脸模型最终略慢，整轮收益主要来自 68 点模型。

## 4. 人脸模型拆分

| 阶段 | 原生 NPU | MNN-U0 |
|---|---:|---:|
| 输入上传 | 4.431 ms | 2.469 ms |
| `predict` / `RunSession` | 8.865 ms | 8.301 ms |
| 输出读取/下载 | 0.525 ms | 0.640 ms |
| U0 C++ 内部总计 | - | 11.476 ms |
| ArkTS/NAPI/结果封装等剩余开销 | 0.317 ms | 3.618 ms |
| 完整调用 | 14.138 ms | 15.094 ms |

U0 的模型执行和输入上传本来更快，但跨 ArkTS/NAPI 返回完整输出的额外开销抵消了优势。人脸模型每次返回：

```text
scores: 4420 x 2
boxes:  4420 x 4
合计：  26520 float，约 106 KB/次
```

当前 C++ 创建输出 `vector`，NAPI 再构造 ArrayBuffer，ArkTS 随后再次包装为 `Float32Array` 并查找最佳候选。这是 U0 人脸路径最明确的应用侧优化点。

## 5. 68 点模型拆分

| 阶段 | 原生 NPU | MNN-U0 |
|---|---:|---:|
| 输入上传 | 0.802 ms | 0.330 ms |
| `predict` / `RunSession` | 4.156 ms | 2.023 ms |
| 输出读取/下载 | 0.071 ms | 0.039 ms |
| U0 C++ 内部总计 | - | 2.433 ms |
| 完整调用 | 5.219 ms | 2.820 ms |

68 点输出只有 136 个 float，NAPI 搬运影响很小。U0 的主要优势来自调用/执行阶段，`RunSession` 比 MindSpore Lite `predict` 快约 51%。

这个数据说明差异位于“框架调度 + 编译后图执行”的组合路径中。现有 trace 还不能把平台 `predict` 内部的 NNRT 调度开销与纯 NPU kernel 时间完全分开，因此不能仅凭该结果断言是硬件算子本身更快。

## 6. 负载与频率

| 指标 | 原生 NPU | MNN-U0 |
|---|---:|---:|
| 应用进程平均 CPU | 21.99% | 20.09% |
| 系统 CPU capacity | 31.334% | 31.430% |
| DDR cluster0 平均频率 | 643,016 | 639,754 |
| DDR cluster1 平均频率 | 501,023 | 526,778 |
| DDR cluster2 平均频率 | 418,719 | 422,799 |
| GPU load | 25.6 | 25.2 |

系统总负载和 GPU 基本相同。U0 应用 CPU 略低，但 DDR cluster1 略高，因此 U0 的时延收益不能解释为“频率更高”或“系统更空闲”。当前 trace 未提供可用的 NPU completion IRQ/频率统计，不能据此比较 NPU 硬件利用率。

## 7. 当前性能差异的判断

1. **人脸模型：U0 的 NPU 执行并不差，差在大输出跨 NAPI 搬运。**
2. **68 点模型：U0 的直接 HiAI/GE 路径明显更快。** 可能包含图优化、运行时派发和 NNRT/MindSpore Lite 封装层差异，尚不能只归因于某一个算子。
3. **整轮 U0 快约 9%，不是此前实时界面单帧显示的两倍差距。** 之前 `43 ms/22 ms` 的单帧数据混入了画面、调度和采样波动，不能代表固定模型调用性能。
4. **两条路径准确率相同。** 当前没有发现 U0 通过错误 layout、错误 Softmax 或精度降级换取性能。

## 8. 优化优先级

1. 在 U0 C++ 层直接完成 4420 个候选框解码，只向 ArkTS 返回 `score + box + index`，避免每次返回约 106 KB 原始输出。
2. 复用 MNN host input/output Tensor 和输出容器，减少每次构造、`memcpy` 与 vector 分配。
3. 在 `NPUBackend::process()` 中给 `AiModelMngerClient::Process()` 增加独立 trace，区分 MNN 调度和 HiAI 同步执行。
4. 若平台允许，给 NNRT/MindSpore Lite `predict` 增加同层级内部标记，才能公平比较纯 `Process`/driver 时间。
5. 正式结论采用交替顺序的多轮采集，记录温度和 NPU/DDR 投票，避免热状态影响。

## 9. 证据文件

- `artifacts/u0_vs_npu_static/npu_deep.trace`
- `artifacts/u0_vs_npu_static/u0_deep.trace`
- `artifacts/u0_vs_npu_static/npu_deep_spans.json`
- `artifacts/u0_vs_npu_static/u0_deep_spans.json`
- `artifacts/u0_vs_npu_static/npu_deep_summary.json`
- `artifacts/u0_vs_npu_static/u0_deep_summary.json`
- `tools/analyze_static_backend_trace.py`
