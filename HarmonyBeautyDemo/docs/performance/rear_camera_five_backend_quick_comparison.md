# 后摄固定人脸五种推理后端正式对比

## 1. 结论

2026-09-01 在 HarmonyBeautyDemo 中重新使用后摄拍摄另一台手机显示的同一张固定正脸图片。五个后端均先预热 10 秒，再采集 30 秒 App Trace。统计时只保留同一个 trace ID 下同时包含人脸推理和 68 点推理的完整帧，不使用界面单帧截图值。

| 推理后端 | 完整样本 | 平均耗时 | P50 | P90 | P95 | 相对 NPU 均值 | 置信度范围 | 关键点 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 原生 NPU | 108 | **50.25 ms** | **50.47 ms** | **66.67 ms** | **68.95 ms** | **1.00 倍** | 99.8% | 68 |
| MNN-U0 | 100 | 53.35 ms | 53.04 ms | 71.60 ms | 72.85 ms | 1.06 倍 | 99.8% | 68 |
| CPU | 106 | 58.86 ms | 59.16 ms | 73.20 ms | 76.40 ms | 1.17 倍 | 100% | 68 |
| MNN-U1 | 98 | 84.40 ms | 86.45 ms | 105.66 ms | 120.21 ms | 1.68 倍 | 100% | 68 |
| GPU | 110 | 86.28 ms | 83.67 ms | 109.95 ms | 116.46 ms | 1.72 倍 | 100% | 68 |

首轮按完整链路均值排序：

```text
原生 NPU < MNN-U0 < CPU < MNN-U1 < GPU
```

首轮原生 NPU 与 MNN-U0 相差 `3.11 ms`。后续交换顺序并同时采集频点后，U0 为 `45.47 ms`，NPU 为 `48.79 ms`，快慢关系反转。两轮加权均值分别为 NPU `49.64 ms`、U0 `49.90 ms`，只差 `0.26 ms / 0.5%`。因此不能把首轮的 `6.2%` 当成稳定的后端性能差距，两条路径在当前完整应用链路中基本持平。

旧报告中的 `22 ms / 43 ms` 来自两个界面单帧截图，不是均值，已从性能结论中撤销。

## 2. 测试口径

| 项目 | 配置 |
|---|---|
| 设备与应用 | 同一台手机、同一进程、同一版本 HarmonyBeautyDemo |
| 相机 | 后摄，`1440 x 1080` |
| 输入画面 | 另一台手机显示的同一张固定正脸图片 |
| 公共输出 | 1 张人脸、68 个关键点 |
| 美颜配置 | 实时滤镜开启、亮白预设，采集中不调整 |
| 静态 Tensor 模式 | 关闭，使用真实后摄输入 |
| 后端顺序 | NPU、MNN-U0、MNN-U1、CPU、GPU |
| 预热 | 每个后端 10 秒 |
| 采集 | 每个后端 30 秒，`hitrace app` |
| 有效帧规则 | `Frame/Total`、`Face/Infer`、`Landmarks/Infer` 三段 trace ID 完整匹配 |
| 统计 | 均值、P50、P90、P95、最小值、最大值 |

每个 30 秒窗口均记录到 166 次取帧尝试。由于相机生产者与推理消费者节奏不同，部分尝试没有取得新帧，只产生很短的 `Frame/Total`。这些空记录全部排除，否则会人为降低平均耗时。

## 3. 阶段拆分

### 3.1 平均耗时

| 后端 | 人脸推理 | SSD 解码 | 关键点预处理 | 68 点推理 | 其他/调度 | 完整链路 |
|---|---:|---:|---:|---:|---:|---:|
| 原生 NPU | 6.44 ms | 0.81 ms | 31.21 ms | 3.13 ms | 8.67 ms | **50.25 ms** |
| MNN-U0 | 6.93 ms | 0.98 ms | 34.36 ms | **2.03 ms** | 9.05 ms | 53.35 ms |
| CPU | 14.99 ms | 0.51 ms | **25.39 ms** | 9.76 ms | 8.21 ms | 58.86 ms |
| MNN-U1 | 26.72 ms | 0.74 ms | 28.99 ms | 18.48 ms | 9.47 ms | 84.40 ms |
| GPU | 32.85 ms | 0.55 ms | 26.26 ms | 18.64 ms | 7.98 ms | 86.28 ms |

“其他/调度”由完整链路减去已标记阶段得到，包含 ArkTS/NAPI 封装、阶段间调度、对象与 ArrayBuffer 处理、结果提交等小段开销。

### 3.2 NPU 与 U0

模型推理阶段相加：

```text
原生 NPU：6.44 + 3.13 = 9.57 ms
MNN-U0： 6.93 + 2.03 = 8.96 ms
```

U0 的两次模型推理合计反而比原生 NPU 快约 `0.60 ms`。完整链路中 NPU 最终快 `3.11 ms`，主要来自本轮 NPU 窗口的关键点 ROI 预处理比 U0 少 `3.15 ms`。

关键点预处理是公共 CPU 路径，不属于 NPU 或 MNN 模型执行。因此，首轮数据不能说明“原生 NPU 的模型执行比 U0 快”。

### 3.3 交换顺序复测

为排除固定采集顺序的影响，在相同后摄、固定人脸和应用配置下，按 `MNN-U0 -> NPU` 顺序重新预热和采集约 22 秒，并同时开启 `app + freq` trace：

| 后端 | 完整帧 | 完整链路均值 | P50 | P90 | 人脸推理 | ROI 预处理 | 68 点推理 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MNN-U0 | 78 | **45.47 ms** | **42.20 ms** | **58.88 ms** | 7.11 ms | **28.67 ms** | **1.82 ms** |
| 原生 NPU | 78 | 48.79 ms | 51.24 ms | 64.37 ms | **6.11 ms** | 31.11 ms | 3.49 ms |

这一次 U0 完整链路反而快 `3.32 ms`。变化最大的仍是公共 ROI 预处理：首轮 U0 比 NPU 慢 `3.15 ms`，复测 U0 又比 NPU 快 `2.44 ms`。这说明公共 CPU 阶段存在明显窗口波动，会掩盖约 1 ms 量级的后端差异。

两轮合并后共有 NPU 186 帧、U0 178 帧：

| 指标 | 原生 NPU | MNN-U0 | U0 - NPU |
|---|---:|---:|---:|
| 完整链路加权均值 | 49.64 ms | 49.90 ms | +0.26 ms |
| 人脸模型推理 | **6.30 ms** | 7.01 ms | +0.71 ms |
| 68 点模型推理 | 3.28 ms | **1.94 ms** | -1.34 ms |
| 两次模型推理合计 | 9.58 ms | **8.95 ms** | -0.63 ms |
| 公共 ROI 预处理 | **31.16 ms** | 31.86 ms | +0.70 ms |

当前最稳妥的结论是：

- 端到端 NPU 与 U0 基本持平，`0.5%` 差值低于公共预处理的窗口波动；
- 原生 NPU 的人脸模型快约 `0.71 ms`；
- U0 的 68 点模型快约 `1.34 ms`，两次模型合计快约 `0.63 ms`；
- 真正占主导的是约 `31 ms` 的 ROI 缩放、裁剪和归一化，而不是 9 ms 左右的模型执行。

### 3.4 频点对比

交换顺序复测中，两路 CPU 和 DDR 平均频点接近：

| 频点 | MNN-U0 | 原生 NPU | 差异 |
|---|---:|---:|---:|
| CPU 0-3 | 1.417 GHz | 1.422 GHz | +0.3% |
| CPU 4-9 | 0.988 GHz | 0.997 GHz | +0.9% |
| CPU 10-11 | 2.315 GHz | 2.330 GHz | +0.6% |
| DDR cluster0 | 718.6 MHz | 719.3 MHz | +0.1% |
| DDR cluster1 | 801.9 MHz | 828.8 MHz | +3.4% |
| DDR cluster2 | 470.6 MHz | 467.1 MHz | -0.7% |

没有发现某条路径获得系统性更高频率的证据。当前 trace 未记录可用的 NPU 完成中断或 NPU 核心频率，因此这里不能比较 NPU 核心占用率和能耗。

### 3.5 U0 路径内部开销

首轮 U0 trace 还能拆出框架内部阶段：

| U0 内部阶段 | 人脸模型 | 68 点模型 | 合计 |
|---|---:|---:|---:|
| `RunSession` | 5.58 ms | 1.80 ms | 7.38 ms |
| Tensor 上传与下载 | 0.48 ms | 0.08 ms | 0.56 ms |
| NAPI/框架边界残差 | 0.83 ms | 0.12 ms | 0.95 ms |

U0 的显式 Tensor 搬运约 `0.56 ms`，只占完整链路约 `1%`，不是当前 45～50 ms 总耗时的主要瓶颈。优先级最高的是公共 ROI 预处理，其次才是把人脸候选解码下沉到 C++、减少 NAPI 大数组返回和复用 Tensor 缓冲区。

### 3.6 CPU

CPU 的完整链路比 NPU 慢约 `17.1%`，但没有达到旧单帧表中的 `3.05 倍`。其关键点预处理阶段本轮最低，部分抵消了 CPU 模型推理较慢的差距。CPU 仍适合作为兼容回退和正确性基准，不适合作为持续低功耗默认路径。

### 3.7 U1 与 GPU

U1 和 GPU 的平均完整链路分别为 `84.40 ms` 和 `86.28 ms`。两者的主要瓶颈都是模型执行，而不是 SSD/关键点解码：

- U1 人脸与关键点推理合计 `45.20 ms`；
- GPU 人脸与关键点推理合计 `51.49 ms`；
- NPU 对应阶段只有 `9.57 ms`。

U1 的 CPU 主图、HiAI 委托和同步边界开销较大。当前 GPU 路径包含 OpenCL 上传、同步和回读，也没有形成相机纹理到 GPU 推理的零拷贝连续链路。

## 4. 输出完整性

Trace 中每个完整样本都有 `State/LandmarkCount=68`：

| 后端 | 人脸置信度最小值 | 人脸置信度最大值 | 关键点计数 |
|---|---:|---:|---:|
| 原生 NPU | 99.8% | 99.8% | 68 |
| MNN-U0 | 99.8% | 99.8% | 68 |
| CPU | 100% | 100% | 68 |
| MNN-U1 | 100% | 100% | 68 |
| GPU | 100% | 100% | 68 |

这能证明五条路径在整个采集窗口内都持续输出 1 张脸和 68 点，但置信度不是准确率。数值一致性仍以固定 Tensor 测试中的 IoU 和关键点误差为准。

## 5. 当前建议

1. **默认后端可继续使用原生 NPU，但理由是接口更直接、可控性更强，不是已证明它稳定快 6%。** 两轮合并后原生 NPU 与 U0 端到端只差约 0.5%。
2. **MNN-U0 已是性能可用的 NPU 接入路径。** 它的纯模型推理合计略快，当前没有证据表明 MNN 框架本身造成明显性能损失。
3. **第一优化目标是公共 ROI 预处理。** 将裁剪、resize、颜色转换和归一化合并到 native SIMD、GPU shader 或 NPU 前处理，并复用输入缓冲，收益上限明显高于继续抠 0.5～1 ms 的后端差异。
4. **CPU 保留为回退。** 当前均值比首轮 NPU 慢约 17%，但明显优于 U1 和 GPU。
5. **U1 只作为兼容路径。** 需要减少 CPU/NPU 切分和同步边界。
6. **GPU 暂不作为默认推理后端。** 优化重点是纹理零拷贝、异步流水和避免输出同步回读。

## 6. 限制

- 已完成一次反向顺序复测，但每种顺序仍只有一个时间窗口；正式功耗实验还应进行多轮 ABBA 交替并记录温度。
- 复测已同时采集 CPU、DDR 和 GPU 频点，但 trace 中没有可用的 NPU 核心频率和完成中断，不能据此给出 NPU 利用率或功耗结论。
- 关键点 ROI 预处理占完整链路的 `25～34 ms`，且不同窗口有数毫秒波动。继续比较 NPU 与 U0 时，需要把公共预处理结果缓存为同一输入，或在交替测试中单独归一化。

## 7. 可复算证据

原始 Trace：

- `artifacts/formal_rear_five_backend/rear_npu_formal.trace`
- `artifacts/formal_rear_five_backend/rear_u0_formal.trace`
- `artifacts/formal_rear_five_backend/rear_u1_formal.trace`
- `artifacts/formal_rear_five_backend/rear_cpu_formal.trace`
- `artifacts/formal_rear_five_backend/rear_gpu_formal.trace`
- `artifacts/formal_rear_five_backend/rear_u0_freq_valid.trace`
- `artifacts/formal_rear_five_backend/rear_npu_freq_valid.trace`

解析结果：

- `artifacts/formal_rear_five_backend/rear_npu_complete.json`
- `artifacts/formal_rear_five_backend/rear_u0_complete.json`
- `artifacts/formal_rear_five_backend/rear_u1_complete.json`
- `artifacts/formal_rear_five_backend/rear_cpu_complete.json`
- `artifacts/formal_rear_five_backend/rear_gpu_complete.json`
- `artifacts/formal_rear_five_backend/rear_u0_freq_valid.json`
- `artifacts/formal_rear_five_backend/rear_npu_freq_valid.json`
- `artifacts/formal_rear_five_backend/rear_u0_freq_summary.json`
- `artifacts/formal_rear_five_backend/rear_npu_freq_summary.json`

解析脚本：`tools/analyze_rear_backend_trace.py`。
