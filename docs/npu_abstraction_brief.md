# HarmonyOS 多推理框架 NPU 统一接入方案汇报

> 汇报版 | 2026-08-31

## 1. 背景

当前 MNN、ncnn、LiteRT、ONNX Runtime 等第三方推理框架需要分别适配华为 NPU。每个框架独立实现算子映射、分图、Tensor Layout、模型编译、缓存和执行，导致接入成本高，正确性和性能表现不一致。

现有 Demo 已验证两类典型问题：

- **逐算子委托性能差**：MNN `USER_1` 只把部分卷积逐个交给 HiAI，固定图片下完整帧平均耗时从原生 NNRT 的 `51.57 ms` 增加到 `102.66 ms`，NPU 完成中断频率达到 `3.72 倍`。
- **整图适配正确性难保证**：MNN `USER_0` 虽有整图执行潜力，但当前人脸检测结果 `IoU=0.328`，未达到 `IoU>=0.95` 的正确性要求，问题涉及 Shape、Axis、Layout 和算子语义转换。

结论：当前问题不是 NPU 算力不足，而是不同框架各自实现 NPU 适配，优化和正确性责任过于分散。

## 2. 当前架构问题

```text
MNN       ncnn       LiteRT       ONNX Runtime
 │         │           │               │
独立后端  独立适配层   独立Delegate   独立Execution Provider
 │         │           │               │
各自实现算子映射、分图、Layout、Cache、执行
 └───────────────┬─────────────────────┘
           HiAI / NNRt / CANN
                  │
               Kirin NPU
```

主要问题：

1. 单算子或小子图频繁提交，CPU/NPU 往返过多。
2. 同一算子在不同框架中重复适配，结果和性能不一致。
3. Layout、Axis、Broadcast 和量化语义容易转换错误。
4. 编译缓存和内存管理无法跨框架复用。
5. 应用只能看到“NPU 后端已选择”，看不到实际下沉比例和回退原因。

## 3. 优化方案

借鉴 Google LiteRT Delegate 的可插拔思路，但建设面向多个框架的统一抽象层：

```text
MNN       ncnn       LiteRT       ONNX Runtime
 │         │           │               │
薄插件    薄插件      Delegate          EP
 └───────────────┬─────────────────────┘
                 │
    Huawei Framework Adapter API
       统一图、能力查询和内存协议
                 │
         NNRt Graph Runtime
   分图 / 编译 / Cache / 调度 / Profiling
                 │
              CANN Kit
                 │
             Kirin NPU
```

核心原则：

- 每个框架只保留一个轻量插件，不再自行实现完整 NPU 编译逻辑。
- 华为统一负责算子能力、图划分、成本评估、编译缓存、内存和性能分析。
- NPU 以连续大子图为执行单位，避免逐卷积提交。
- 不支持算子继续由框架 CPU/GPU 执行，但回退必须可见、可解释。
- 正确性校验先于性能发布。

## 4. 核心设计

### 4.1 统一框架接口

提供稳定的 C ABI，覆盖：

```text
设备和能力查询
完整图分析
分图方案生成
NPU 子图编译
编译结果加载和缓存
Tensor 与共享内存绑定
同步/异步执行
Profiling 和回退原因
```

### 4.2 基于收益的分图

不能采用“算子支持就下沉”的方式，应综合计算：

```text
下沉收益
= CPU/GPU执行时间
- NPU计算时间
- 数据拷贝
- Layout转换
- 提交与同步
- 摊销后的编译成本
```

只有总收益为正、子图计算量足够时才进入 NPU。

### 4.3 统一正确性契约

统一 IR 必须明确表达：

- 逻辑 Shape 和物理 Shape；
- Tensor Layout 和通道 Packing；
- Axis、Broadcast 和动态维度；
- 数据类型与量化参数；
- 算子语义版本。

每个设备和框架组合必须通过固定输入 Fixture，与 CPU Reference 对齐后才能发布。

### 4.4 内存与缓存

- 支持 Shared Memory、Native Buffer、DMA-BUF 和 NPU Buffer 导入。
- 通过 Layout 协商减少 Host 中转和格式转换。
- 编译缓存同时绑定模型、Shape、SoC、驱动、CANN 和算子库版本。
- 稳态执行不重复编译，不重复创建 Tensor 和工作区。

### 4.5 可观测性

统一输出：

- NPU 节点占比和子图数量；
- CPU/GPU 回退节点及原因；
- 编译时间和 Cache 命中；
- 边界拷贝字节数；
- Layout 转换次数和耗时；
- 子图排队、执行和 Fence 等待时间；
- P50/P95/P99、功耗和频点。

## 5. 预期收益

### 对应用开发者

- 继续使用 MNN、ncnn、LiteRT 等原有接口。
- 只需选择 Huawei NPU Backend，不直接理解 CANN 和驱动。
- 同一模型可在 NPU、CPU、GPU 之间可靠回退。

### 对框架生态

- 减少重复的算子和 NPU 后端开发。
- 华为底层优化可以同时覆盖多个框架。
- 统一兼容矩阵和测试标准。

### 对性能

- 减少 NPU 提交和 CPU/NPU 边界。
- 提高中间 Tensor 驻留和算子融合机会。
- 降低 CPU 占用、DDR 搬运和长尾。
- 目标是第三方框架稳定态性能与 MindSpore Lite/NNRT 差距控制在 `10%` 以内。

## 6. 落地计划

| 阶段 | 主要工作 | 目标 |
| --- | --- | --- |
| 阶段 0 | 统一 Trace、固定 Fixture、后端命中报告 | 能解释是否使用 NPU、为什么慢 |
| 阶段 1 | 定义最小 Adapter API，改造 MNN 插件 | 人脸/关键点正确，性能接近 NNRT |
| 阶段 2 | 统一分图、缓存、共享内存，支持 LiteRT/ncnn | 多框架复用同一套优化 |
| 阶段 3 | 异步执行、动态 Shape、多模型调度、ORT | 支撑相机、直播、语音和大模型场景 |

## 7. 建议优先决策

1. 统一抽象层是作为 NNRt 新接口，还是独立的 Framework Adapter Kit。
2. 统一图格式复用 MindIR/NNRt IR，还是定义稳定交换 IR。
3. 第一批官方支持 MNN 和 LiteRT，还是同时覆盖 ncnn。
4. 编译缓存是否允许跨应用系统级复用。
5. CANN 高级能力以什么范围向三方框架开放。

## 8. 最终建议

建议采用：

> **多框架薄插件 + 统一 Framework Adapter API + NNRt/CANN 高性能 Runtime**

这不是简单增加一层接口包装，而是把分散在各框架中的正确性和性能优化能力收敛到华为统一维护，使第三方框架能够稳定、高效地使用麒麟 NPU。

完整设计见：[HarmonyOS 多推理框架 NPU 统一抽象层设计](harmonyos_multi_framework_npu_abstraction_design.md)。
