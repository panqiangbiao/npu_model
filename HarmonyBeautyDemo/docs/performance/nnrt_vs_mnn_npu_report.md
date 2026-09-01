# 原生 NNRT 与 MNN-NPU 性能对比

## 0. USER_1 固定静态图 A/B（非目标路径基线）

> 本节不是目标 USER_0 整图 MNN-NPU 数据。它仅记录 USER_1 逐卷积委托基线，不可用于回答
> “USER_0 对原生 NPU”的性能差异；USER_0 正确性通过前暂停目标性能对比。

本轮使用另一台手机显示同一张静态正脸图片，测试手机保持前摄、画面位置、亮白预设、实时美颜和
68 点标记不变，仅切换推理后端。两条路径的人脸框与关键点均正确，置信度均稳定为 100%。Trace
保留 NPU 41 个、MNN-NPU 42 个按 `traceId` 配对的完整有效帧。

| 指标 | 原生 NPU | MNN-NPU USER_1 | MNN 相对变化 |
|---|---:|---:|---:|
| 完整有效帧平均 | **51.57 ms** | 102.66 ms | **+99.1%** |
| 完整有效帧 P50 | **53.54 ms** | 104.07 ms | +94.4% |
| 完整有效帧 P95 | **63.06 ms** | 128.75 ms | **+104.2%** |
| 完整有效帧最大值 | **66.03 ms** | 199.55 ms | 3.02 倍 |
| 人脸推理平均 | **6.44 ms** | 40.59 ms | **6.31 倍** |
| 人脸推理 P95 | **9.68 ms** | 56.68 ms | 5.86 倍 |
| 关键点 ROI 预处理 | 33.02 ms | **29.84 ms** | -9.6% |
| 关键点推理平均 | **3.50 ms** | 24.23 ms | **6.93 倍** |
| 关键点推理 P95 | **6.54 ms** | 34.37 ms | 5.26 倍 |
| 应用平均 CPU 占用 | **58.54%** | 68.34% | +16.7% |
| NPU 完成中断/秒 | **70.18** | 261.12 | **3.72 倍** |

MNN-NPU 的人脸 `RunSession` 平均 39.44 ms、关键点 `RunSession` 平均 24.09 ms；对应上传和下载
总计约 1.20 ms。固定输入进一步确认差距不来自图像内容、ROI 变化或应用侧 Tensor 拷贝，而来自
USER_1 的逐卷积 HiAI 委托方式：多个离散 NPU 调用增加提交、同步和 CPU/NPU 边界开销，并失去整图
融合机会。MNN-NPU 还出现一个 199.55 ms 完整帧长尾，主要对应人脸推理的 123.87 ms 最大值。

本轮文件：

```text
artifacts/backend_compare/recompare_check.jpeg
artifacts/backend_compare/mnn_static_valid_pre.jpeg
artifacts/backend_compare/npu_static_valid.trace
artifacts/backend_compare/npu_static_valid_hilog.txt
artifacts/backend_compare/npu_static_valid_summary.json
artifacts/backend_compare/mnn_static_valid.trace
artifacts/backend_compare/mnn_static_valid_hilog.txt
artifacts/backend_compare/mnn_static_valid_summary.json
```

## 0. 2026-08-29 同场复测（当前构建）

本轮在 MNN-NPU 正确性自检通过后重新采集。两组均使用前摄、亮白预设、实时美颜和 68 点标记，
同一现场连续采集 20 秒；原生 NPU 与 MNN-NPU 均持续识别人脸并执行关键点链路。MNN-NPU 使用
`MNN_FORWARD_USER_1`，即 CPU + 逐卷积 HiAI 委托，不代表 USER_0 整图后端。

| 指标 | 原生 NPU | MNN-NPU USER_1 | MNN 相对变化 |
|---|---:|---:|---:|
| 完整有效帧平均 | **51.24 ms** | 105.10 ms | **+105.1%** |
| 完整有效帧 P95 | **69.96 ms** | 127.85 ms | +82.7% |
| 人脸推理平均 | **9.57 ms** | 39.84 ms | **4.16 倍** |
| 人脸推理 P95 | **15.01 ms** | 56.46 ms | 3.76 倍 |
| 关键点 ROI 预处理 | **31.12 ms** | 31.92 ms | +2.6% |
| 关键点推理平均 | **3.33 ms** | 24.27 ms | **7.28 倍** |
| 关键点推理 P95 | **4.53 ms** | 29.24 ms | 6.46 倍 |
| 应用平均 CPU 占用 | **50.87%** | 70.40% | +38.4% |
| NPU 完成中断/秒 | **74.35** | 261.31 | **3.51 倍** |

完整有效帧按同一 `traceId` 同时出现 `Frame/Total` 与 `Landmarks/Infer` 配对统计，排除了没有新帧、
无人脸等快速返回事件。环形 trace 最终保留约 5.4 至 5.7 秒，包含 NPU 13 个、MNN-NPU 20 个
完整有效帧；阶段统计各包含约 20 次人脸推理。

### 本轮判断

1. 两条路径的 ROI 预处理几乎相同，约 31 ms，不是框架差距来源。
2. MNN-NPU 的 `RunSession` 占主导：人脸约 38.67 ms，关键点约 24.10 ms；上传与下载合计不足
   1.3 ms，应用侧 Tensor 拷贝不是主要瓶颈。
3. USER_1 将卷积拆成多个独立 HiAI 调用，其 NPU 完成中断频率达到原生路径的 3.51 倍，同时应用
   CPU 占用和调度次数也更高。当前主要损失来自逐算子图提交、CPU/NPU 边界和无法进行整图融合。
4. 因 USER_0 整图路径输出不正确，本轮不能比较“整图 MNN-NPU 与 NNRT”；可用结论是：当前准确
   的 USER_1 混合路径明显慢于原生 NPU，但能够作为兼容性接入方案。

本轮原始数据：

```text
artifacts/backend_compare/npu_user1_pair.trace
artifacts/backend_compare/npu_user1_pair_hilog.txt
artifacts/backend_compare/npu_user1_pair_summary.json
artifacts/backend_compare/mnn_user1_pair.trace
artifacts/backend_compare/mnn_user1_pair_hilog.txt
artifacts/backend_compare/mnn_user1_pair_summary.json
artifacts/backend_compare/pre_capture.jpeg
artifacts/backend_compare/mnn_pair_pre.jpeg
```

> **代码审查更新：** 当前 Demo 选择的 `MNN_FORWARD_USER_1` 实际是 CPU + 逐卷积 HiAI 委托后端，
> 不是整图 MNN-NPU。本文测量值有效，但不能作为整图 MNN 与 NNRT 的最终对比。详见
> [mnn_npu_code_path_audit.md](mnn_npu_code_path_audit.md)。

## 1. 对比对象

本文中的“原生 NNRT”是 Demo 现有的 MindSpore Lite `target: ['nnrt']` 路径，不是应用直接调用
`OH_NN*` C API：

```text
.ms -> MindSpore Lite -> HarmonyOS NNRT -> 厂商 NPU 驱动
```

MNN-NPU 使用 mobiInfer 的 MNN 3.6 HiAI 后端：

```text
.mnn -> MNN Interpreter -> MNN HiAI op 映射 -> BuildIRModel/OM
     -> AiModelMngerClient::Process -> NPU
```

两条路径使用相同 ONNX 来源的人脸检测和 68 点关键点模型，但转换后的 `.ms` 与 `.mnn` 是不同产物。

## 2. 结果

### 2.1 新采集的 NNRT 完整链路

本次在当前安装包、清晰正脸、置信度约 `0.92~0.998` 的条件下重新采集。Trace 中包含 37 次人脸
检测和 37/38 次关键点推理，不再是无脸时的半条链路。

| NNRT 阶段 | 平均 | P95 | 最大 |
|---|---:|---:|---:|
| 完整链路（hilog） | **33.17 ms** | 36.00 ms | 36.00 ms |
| 人脸推理 | 4.56 ms | 6.40 ms | 6.95 ms |
| 人脸 SSD 解码 | 0.37 ms | 0.43 ms | 0.45 ms |
| 关键点 ROI 预处理 | 24.56 ms | 26.84 ms | 27.41 ms |
| 关键点推理 | 2.84 ms | 4.84 ms | 5.95 ms |
| 关键点解码 | 0.07 ms | 0.08 ms | 0.10 ms |

NNRT 链路的最大瓶颈已经不是 NPU 推理，而是 CPU 上的关键点 ROI 预处理，占完整链路约 `74%`。

### 2.2 同场 MNN-NPU 完整链路

重新调整为 MNN 可稳定识别的正脸姿态后采集，置信度为 `0.893~1.000`，Trace 中包含 34 次完整的
人脸和关键点推理。两组均来自当前安装包和同一现场：

| 指标 | NNRT/NPU | MNN-NPU |
|---|---:|---:|
| 完整链路（hilog） | **33.17 ms** | 55.50 ms |
| 完整链路 P95 | **36.00 ms** | 57.75 ms |
| 人脸推理平均 | **4.56 ms** | 17.97 ms |
| 人脸推理 P95 | **6.40 ms** | 22.89 ms |
| 关键点 ROI 预处理 | 24.56 ms | **22.46 ms** |
| 关键点推理平均 | **2.84 ms** | 18.00 ms |
| 关键点推理 P95 | **4.84 ms** | 20.49 ms |

MNN-NPU 完整链路比 NNRT 多 `22.33 ms`，慢 `67.3%`。人脸推理是 NNRT 的 `3.94 倍`，关键点
推理是 `6.35 倍`。MNN 的 ROI 预处理反而快约 `2.10 ms`，因此性能差距明确来自两个 NPU 图。

两组是同一现场连续采集但不是同一帧固定输入；比例用于定位数量级，最终仍需固定输入复测。

### 2.3 MNN 内部分段

| MNN 阶段 | 平均 | P95 |
|---|---:|---:|
| 人脸 Tensor 上传 | 0.12 ms | 0.20 ms |
| 人脸 `RunSession` | 17.39 ms | 22.24 ms |
| 人脸输出回读 | 0.45 ms | 0.59 ms |
| 关键点 Tensor 上传 | 0.04 ms | 0.12 ms |
| 关键点 `RunSession` | 17.85 ms | 20.36 ms |
| 关键点输出回读 | 0.05 ms | 0.07 ms |

两次 MNN 推理中，上传和下载总计约 `0.66 ms`，而 `RunSession` 合计约 `35.24 ms`。差距不是应用侧
Tensor 搬运造成的。

## 3. 为什么 MNN-NPU 更慢

### 3.1 两条路径生成的 NPU 图并不相同

NNRT 路径加载 `.ms` 后，由 MindSpore Lite 与 NNRT 设备后端完成图转换和编译。MNN 路径先把
`.mnn` 中的算子逐个映射为 HiAI GE Operator，再在线执行 `BuildIRModel` 生成并加载 OM。

即使源 ONNX 相同，算子融合、常量折叠、布局转换、卷积算法和精度选择都可能不同。当前稳定态
`RunSession` 已经比 NNRT `predict()` 明显更慢，说明 MNN 生成图的执行效率是首要调查对象。

### 3.2 精度与布局策略可能不同

当前 MNN Session 明确配置 `BackendConfig::Precision_High`。NNRT 配置的是
`PERFORMANCE_HIGH` 和 `PRIORITY_HIGH`，它们是性能与调度偏好，不等同于强制高精度。

MNN HiAI 后端还包含 NCHW、NHWC、NC4HW4 转换逻辑。虽然 trace 显示应用侧 Tensor 拷贝很小，
图内部是否插入额外 Transpose/FormatConvert、是否使用 FP32，仍需导出两份 OM 或编译日志确认。

### 3.3 MNN 使用的是自定义 HiAI 适配层

mobiInfer 后端最终调用旧式 `AiModelMngerClient::Process`，中间承担 MNN Tensor 到 HiAI Tensor
映射、输出索引恢复和图构建。NNRT 则由系统 Runtime 与设备驱动直接协商能力。MNN 多出来的框架层
本身不是主要拷贝瓶颈，但它决定了提交给 NPU 的图结构，可能无法获得 NNRT 路径相同的融合质量。

### 3.4 当前还存在模型输出一致性问题

此前姿态变化时曾出现输出方向不一致：NNRT 能识别的正脸，MNN 一度只有 `0.18~0.19`；重新调整
角度后 MNN 恢复为 `0.893~1.000`。本轮性能样本已经是有效完整链路，但该现象说明 MNN 的检测鲁棒性
仍需用固定图片逐元素对齐，不能只看单张自检图片通过。

本次有效样本中两边都执行了人脸和关键点，`55.50 ms` 对 `33.17 ms` 的链路差可以用于性能比较；
但精度和鲁棒性结论仍需固定输入验证。

### 3.5 在线编译不是稳定态差距

MNN 的 `BuildIRModel`、OM 生成和模型 Load 发生在 `createSession()`/初始化阶段。本报告采集前已预热，
表中的 `RunSession` 是稳定态数据。在线编译会影响首次打开时间，但不是本轮每帧慢 `20 ms` 的主因。

## 4. 当前结论

1. 当前 Demo 中，NNRT/NPU 完整链路平均 `33.17 ms`，MNN-NPU 为 `55.50 ms`，MNN 慢 `67.3%`。
2. 差距同时存在于人脸和关键点 NPU 图；MNN 分别约为 NNRT 的 `3.94 倍`和 `6.35 倍`。
3. MNN 的上传、下载不是主要问题；应优先比较编译后的图、精度和算子融合。
4. MNN 本轮功能链路有效，但姿态变化下曾出现置信度异常，仍需固定输入做精度对齐。

## 5. 下一步验证

1. 用 `front_camera_face_input.bin` 对两后端输出逐元素比较：最大绝对误差、Top-10 anchor、Softmax 前后值。
2. 记录或导出 MNN HiAI 与 NNRT 的编译图信息，比较算子数、Transpose 数、融合 Conv 数和输入精度。
3. 分别测试 MNN `Precision_High`、`Precision_Normal`、`Precision_Low`，同时校验检测精度。
4. 修复输出一致性后，用固定输入循环 200 次，分别统计人脸模型与关键点模型，不让误检门限影响链路长度。
5. 最后再进行相机实景 A/B，比较 CPU、DDR、NPU 频点和长尾。

## 6. 数据文件

```text
artifacts/backend_compare/
  npu.trace / npu_hilog.txt / npu_summary.json
  mnn_npu.trace / mnn_npu_hilog.txt / mnn_npu_summary.json
  nnrt_paired.trace / nnrt_paired_hilog.txt / nnrt_paired_summary.json
  nnrt_face.trace / nnrt_face_hilog.txt / nnrt_face_summary.json
  mnn_face.trace / mnn_face_hilog.txt / mnn_face_summary.json
  mnn_face_valid.trace / mnn_face_valid_hilog.txt / mnn_face_valid_summary.json
```
