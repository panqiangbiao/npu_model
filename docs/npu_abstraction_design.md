# HarmonyOS 多推理框架 NPU 统一抽象层设计

> 文档状态：方案设计稿  
> 更新时间：2026-08-31  
> 适用范围：HarmonyOS/Huawei 手机端，MNN、ncnn、LiteRT、ONNX Runtime、MindSpore Lite 等第三方推理框架接入麒麟 NPU

## 1. 文档目标

当前多个第三方推理框架分别接入 HiAI、NNRt 或 CANN，形成了多套算子映射、分图、模型编译、内存转换和执行逻辑。框架虽然可以“调用 NPU”，但性能、正确性和可维护性不稳定。

本文给出一套完整设计，回答以下问题：

1. 当前接入架构和实际问题是什么。
2. 优化后的目标架构是什么。
3. 是否需要改造成类似 LiteRT Delegate 的方式。
4. 统一抽象层提供哪些接口和运行机制。
5. 图划分、编译缓存、内存、调度、回退和可观测性如何设计。
6. 华为、框架社区和应用开发者分别承担什么责任。
7. 如何从现有 MNN/HiAI 路径逐步迁移。

## 2. 结论摘要

建议借鉴 LiteRT Delegate 的**可插拔框架后端**，但不只实现一个 LiteRT Delegate。目标形态应是：

```text
多框架薄插件
    +
统一的 Huawei Framework Adapter API
    +
增强后的 NNRt 图分析、编译、内存与执行服务
    +
CANN Kit 和麒麟 NPU 驱动
```

核心原则：

- 每个框架保留自己的模型解析、CPU Kernel 和 Session 生命周期。
- 框架插件只完成图和 Tensor 的标准化桥接，不重复实现 NPU 编译器逻辑。
- 算子能力、图分析、成本评估、NPU 子图编译、缓存和 Profiling 由华为统一实现。
- 不支持算子继续由原框架 CPU/GPU 执行，但回退必须可见、可解释、可测试。
- 以子图而不是单算子作为 NPU 调度单位。
- 把正确性校验作为后端发布门槛，性能优化不能建立在输出错误之上。

## 3. 背景与现状证据

### 3.1 当前 Demo 的两条 MNN 接入路径

当前 mobiInfer 分支通过两个自定义 `MNNForwardType` 接入华为 NPU：

| 类型 | 实现 | 实际行为 |
| --- | --- | --- |
| `MNN_FORWARD_USER_0` | `NPUBackend` | 将 MNN 算子转换成 HiAI/GE 图，尝试整图或大子图执行 |
| `MNN_FORWARD_USER_1` | `HiAIDelegateBackend` | CPU Backend 为主，只把部分卷积逐个交给 HiAI |

`USER_1` 当前结果正确，但固定静态图片测试中表现为：

| 指标 | 原生 NNRT/NPU | MNN USER_1 | 相对变化 |
| --- | ---: | ---: | ---: |
| 完整有效帧平均 | 51.57 ms | 102.66 ms | +99.1% |
| 人脸推理平均 | 6.44 ms | 40.59 ms | 6.31 倍 |
| 关键点推理平均 | 3.50 ms | 24.23 ms | 6.93 倍 |
| NPU 完成中断/秒 | 70.18 | 261.12 | 3.72 倍 |

上传和下载总计约 1.20 ms，不足以解释整体差距。主要问题是多个离散 NPU 调用、频繁同步、CPU/NPU 边界和无法整图融合。

`USER_0` 具备整图性能潜力，但当前输出未通过正确性校验：

```text
预期 bestIndex = 4271，IoU >= 0.95
实际 bestIndex = 4382，IoU = 0.328
```

已暴露的风险包括：

- 逻辑二维 Tensor 被补齐成四维物理 Shape 后，Softmax Axis 映射错误。
- MNN NC4HW4 和 NCHW 不能用普通 Reshape/Permute 等价转换。
- Boxes 和 Scores 同时偏离，问题不局限于最终 Softmax。
- 某些等价算子展开能够通过 C++ 编译，却被设备侧 `BuildIRModel` 拒绝。
- MNN 和 HiAI 对 Shape、Layout、Broadcast、Quantization 的语义缺少统一契约。

### 3.2 当前接入方式的共性问题

MNN、ncnn、LiteRT 和 ONNX Runtime 分别接入时，容易形成如下重复建设：

```text
MNN Backend       -> MNN Op 到 HiAI Op 映射
ncnn Backend      -> ncnn Layer 到 NPU Op 映射
LiteRT Delegate   -> TFLite Op 到 NPU Op 映射
ORT EP            -> ONNX Op 到 NPU Op 映射
```

每套实现分别处理：

- 算子支持表；
- Shape 推导；
- Axis 和 Broadcast 转换；
- Tensor Layout；
- 量化参数；
- 图划分；
- 在线编译；
- Cache Key；
- 输入输出内存；
- 同步和执行；
- 错误码；
- Profiling。

结果是同一款 NPU 在不同框架下表现不同，底层优化无法一次覆盖所有框架。

## 4. 当前架构

### 4.1 架构图

```text
┌─────────────────────────────────────────────────────────────┐
│                         Applications                        │
└───────────┬──────────────────┬──────────────────┬───────────┘
            │                  │                  │
       ┌────▼────┐        ┌────▼────┐        ┌────▼──────┐
       │   MNN   │        │  ncnn   │        │ LiteRT/ORT│
       └────┬────┘        └────┬────┘        └────┬──────┘
            │                  │                  │
       独立 Backend        独立 Adapter       独立 Delegate/EP
            │                  │                  │
            ├────── 各自算子映射、分图、Layout、Cache ──────┤
            │                  │                  │
       ┌────▼──────────────────▼──────────────────▼─────┐
       │          HiAI / NNRt / CANN 不同接入口          │
       └───────────────────────┬─────────────────────────┘
                               │
                         Kirin NPU Driver
                               │
                           Kirin NPU
```

### 4.2 当前执行模式

#### 模式 A：逐算子委托

```text
Framework CPU Op
  -> 上传 Tensor
  -> 构建/调用单个 NPU Op
  -> 同步等待
  -> 下载 Tensor
  -> Framework CPU Op
  -> 再次调用 NPU
```

优点是容易回退，缺点是提交次数、同步和边界转换过多。

#### 模式 B：框架自行构建整图

```text
Framework Graph
  -> 框架 Backend 自行转换每个 Op
  -> 构建 HiAI/GE Graph
  -> BuildIRModel
  -> Load
  -> Process
```

性能潜力较高，但每个框架必须独立正确处理全部图语义。当前 `USER_0` 的错误说明此模式的正确性维护成本很高。

#### 模式 C：应用提前转换离线模型

```text
Training Model
  -> Vendor Converter
  -> Offline OM/Hardware Model
  -> NNRt/CANN Load
  -> Execute
```

性能和首帧加载较好，但模型与硬件、系统和编译器版本耦合，通用框架的动态能力和 CPU 回退较弱。

## 5. 当前问题归因

### 5.1 P0：NPU 调度粒度错误

把单个卷积作为委托单位，会让 NPU 执行时间被提交、同步和转换开销淹没。NPU 调度单位应是有足够计算量、能够保持中间 Tensor 驻留的连续子图。

### 5.2 P0：图语义转换责任分散

各框架独立处理低维补齐、Axis、Layout、Broadcast 和 Quantization，容易出现“图能够编译和执行，但输出错误”。

### 5.3 P0：能力查询仅判断支持，不计算收益

传统 `IsOpSupported()` 只能回答某个算子能否运行，不能判断下沉后是否更快。即使一个卷积被支持，如果前后发生两次拷贝和 Layout 转换，也可能不值得下沉。

### 5.4 P1：内存互操作不统一

框架 Host Tensor、相机 Buffer、GPU Texture、NNRt Tensor 和 NPU Buffer 之间缺少统一导入、导出和生命周期协议，导致不必要的中间 Buffer 和格式转换。

### 5.5 P1：在线编译与 Cache 不统一

不同框架分别生成 Cache Key 和缓存文件，无法稳定复用编译结果，也难以处理驱动、固件、算子库和模型变化导致的失效。

### 5.6 P1：后端命中不可见

应用只能看到“选择了 NPU Backend”，无法确认：

- 实际下沉多少算子；
- 生成多少 NPU 子图；
- 哪些节点回退；
- 每个边界搬运多少数据；
- 是否发生在线编译；
- 性能损失发生在哪一层。

### 5.7 P1：版本和发布责任不清晰

推理框架、华为插件、NNRt、CANN、驱动和系统固件分别演进。没有明确兼容矩阵时，框架升级可能造成静默回退或数值变化。

## 6. 设计目标与非目标

### 6.1 设计目标

1. 为主流推理框架提供稳定的 NPU 接入协议。
2. 同一个 NPU 优化能够覆盖多个框架。
3. 优先形成大子图，避免逐算子提交。
4. 不支持算子能够安全回退原框架 CPU/GPU。
5. 减少跨设备拷贝和 Layout 转换。
6. 支持在线编译、离线模型和编译缓存。
7. 提供可观测、可解释、可回归的执行链路。
8. 保持 ABI 稳定，使 Runtime 可以随系统升级而插件不必频繁重编。

### 6.2 非目标

- 不要求一个通用 IR 表达所有未来算子。
- 不要求第一阶段支持 NPU 自定义算子开发。
- 不在统一层重写 MNN、ncnn 等框架的 CPU Kernel。
- 不保证所有模型都比 CPU/GPU 快。
- 不允许为了 NPU 命中率牺牲输出正确性。

## 7. 优化后的目标架构

### 7.1 总体架构

```text
┌──────────────────────────────────────────────────────────────┐
│                         Applications                         │
└─────────────┬─────────────┬──────────────┬──────────────────┘
              │             │              │
         ┌────▼────┐   ┌────▼────┐   ┌─────▼─────┐
         │   MNN   │   │  ncnn   │   │LiteRT/ORT │
         └────┬────┘   └────┬────┘   └─────┬─────┘
              │             │              │
        MNN NPU Plugin  ncnn NPU Plugin  Delegate / EP
              └─────────────┬──────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────┐
│ Huawei Framework Adapter API                                │
│ Graph IR | Capability | Partition Plan | Tensor/Memory | ABI │
└───────────────────────────┬──────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────┐
│ NNRt Graph Runtime vNext                                    │
│ Validation | Cost Model | Partition | Compile | Cache        │
│ Async Execute | Memory Interop | Scheduling | Profiling      │
└───────────────────────────┬──────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────┐
│ CANN Kit / Kirin AI Compiler / Operator Library              │
└───────────────────────────┬──────────────────────────────────┘
                            │
                     Kirin NPU Driver / NPU
```

### 7.2 与 Google Delegate 的关系

借鉴部分：

- 框架通过可插拔插件接入硬件。
- 插件可以声明支持节点并替换连续子图。
- 未委托节点保留在原框架执行。
- 应用使用统一框架 API，不直接调用驱动。

华为增强部分：

- 不是只定义某一个框架的 Delegate，而是定义跨框架 Adapter API。
- 分图、成本模型、缓存和内存策略下沉到统一 NNRt Runtime。
- 华为同时掌握系统、芯片、CANN 和驱动，可以提供统一机型验证和系统升级。
- 官方维护主流框架插件，避免核心性能适配完全依赖社区。

## 8. 组件设计

### 8.1 Framework Plugin

每个框架插件负责：

1. 从框架获取完整静态图或当前可编译子图。
2. 将框架 Tensor、Op 和属性转换成 Huawei Graph IR。
3. 调用 `AnalyzeGraph()` 获取分图方案。
4. 用一个 `HuaweiSubgraphExecution` 替换每个 NPU 子图。
5. 绑定框架 Tensor 与 Huawei Tensor/Buffer。
6. 把未下沉节点留在框架原生 Backend。
7. 将 Runtime Profiling 关联回框架节点名称。

插件不负责：

- 维护 NPU 算子实现；
- 手写每个算子的 GE 图展开；
- 独立实现编译缓存；
- 直接控制 NPU 驱动；
- 自行决定所有性能策略。

### 8.2 Huawei Framework Adapter API

这是面向框架插件的稳定 C ABI。建议避免暴露 C++ STL 类型，降低编译器和版本耦合。

#### 设备与能力

```cpp
HFA_Status HFA_GetDevices(HFA_DeviceList* devices);
HFA_Status HFA_GetDeviceProperties(
    HFA_DeviceId device,
    HFA_DeviceProperties* properties);

HFA_Status HFA_QueryGraphCapabilities(
    HFA_DeviceId device,
    const HFA_Graph* graph,
    const HFA_CompileOptions* options,
    HFA_CapabilityReport** report);
```

能力报告不能只返回布尔值，应包含：

```text
节点是否支持
不支持原因
支持的 dtype/layout/shape 范围
是否要求静态 Shape
是否需要插入转换节点
预计编译成本
预计执行收益等级
可融合关系
```

#### 图分析和分区

```cpp
HFA_Status HFA_AnalyzeGraph(
    HFA_DeviceId device,
    const HFA_Graph* graph,
    const HFA_PartitionOptions* options,
    HFA_PartitionPlan** plan);
```

`HFA_PartitionPlan` 返回：

- NPU 子图节点集合；
- 每个子图的输入输出；
- 边界 Layout；
- 必需转换；
- 预计收益；
- 放弃下沉的原因；
- 推荐 CPU/GPU 回退节点。

#### 编译与缓存

```cpp
HFA_Status HFA_CompileSubgraph(
    HFA_DeviceId device,
    const HFA_Graph* graph,
    const HFA_SubgraphId subgraph,
    const HFA_CompileOptions* options,
    HFA_CompiledModel** model);

HFA_Status HFA_LoadCompiledModel(
    HFA_DeviceId device,
    const void* data,
    size_t size,
    HFA_CompiledModel** model);
```

#### Tensor 与内存

```cpp
HFA_Status HFA_CreateTensor(
    const HFA_TensorDesc* desc,
    HFA_Tensor** tensor);

HFA_Status HFA_ImportMemory(
    const HFA_MemoryDesc* desc,
    HFA_Memory** memory);

HFA_Status HFA_BindTensorMemory(
    HFA_Tensor* tensor,
    HFA_Memory* memory,
    size_t offset);
```

#### 执行

```cpp
HFA_Status HFA_CreateExecution(
    HFA_CompiledModel* model,
    HFA_Execution** execution);

HFA_Status HFA_ExecuteAsync(
    HFA_Execution* execution,
    const HFA_TensorBinding* inputs,
    size_t inputCount,
    const HFA_TensorBinding* outputs,
    size_t outputCount,
    const HFA_Fence* waitFence,
    HFA_Fence** signalFence);
```

即使 NNRt 当前只支持同步推理，抽象层也应预留异步接口。早期实现可以在内部线程池包装同步执行，后续再替换成驱动原生异步能力。

### 8.3 Huawei Graph IR

统一 IR 是框架语义和 NPU 编译器之间的契约，必须显式表达：

- Tensor Rank、逻辑 Shape 和物理 Shape；
- 动态维度范围；
- 数据类型；
- 量化 Scale、Zero Point、Axis；
- 逻辑 Layout 和物理 Packing；
- Axis 的语义基准；
- Broadcast 规则；
- Constant 数据；
- OpSet/算子语义版本；
- 可变状态和模型输入输出。

禁止用普通 `Reshape` 隐式表达 NC4HW4 解包。Layout 转换必须是显式 IR 节点或 Tensor View 语义。

### 8.4 NNRt Graph Runtime vNext

NNRt vNext 负责：

- Graph IR 校验；
- Shape 推导与约束检查；
- 算子能力查询；
- 分图和成本估算；
- CANN 图转换；
- 编译与缓存；
- Tensor/Buffer 管理；
- 异步执行；
- 多应用调度提示；
- Profiling 和诊断。

NNRt 不需要实现所有 CPU Kernel。CPU/GPU 回退仍由上层框架执行。

## 9. 图划分设计

### 9.1 分图输入

分图器至少需要：

- 完整计算图；
- 节点拓扑；
- Tensor Shape、dtype、layout 和大小；
- NPU 算子能力；
- 框架 CPU/GPU 执行成本提示；
- 边界转换成本；
- 模型性能目标；
- 首帧优先或吞吐优先策略。

### 9.2 成本模型

不能采用“支持就下沉”的策略。子图收益建议按以下模型估算：

```text
Benefit(subgraph)
  = FrameworkCost(subgraph)
  - NpuComputeCost(subgraph)
  - UploadCost(inputs)
  - DownloadCost(outputs)
  - LayoutTransformCost(boundaries)
  - SynchronizationCost
  - SubmissionCost
  - AmortizedCompileCost
```

只有满足以下条件才下沉：

```text
Benefit > MinBenefitThreshold
ComputeIntensity > MinComputeThreshold
BoundaryCount <= MaxBoundaryCount
CorrectnessRisk == Accepted
```

### 9.3 分图约束

- 尽量让连续 Conv/MatMul/Activation/Normalization 留在同一子图。
- 避免只下沉单个小卷积。
- 对高带宽小计算算子，除非能与前后节点融合，否则优先留在 CPU/GPU。
- 避免 NPU -> CPU -> NPU 的短间隔往返。
- Layout 转换节点必须参与成本计算。
- 动态 Shape 超出已编译 Profile 时，可选择新编译、回退或使用 Padding Profile。

### 9.4 分图结果示例

```text
原图：
Conv -> ReLU -> Resize -> Conv -> Add -> Softmax

错误策略：
NPU Conv -> CPU ReLU/Resize -> NPU Conv -> CPU Add -> NPU Softmax

推荐策略：
NPU Subgraph(Conv + ReLU)
  -> CPU Resize
  -> NPU Subgraph(Conv + Add + Softmax)

如果第二个子图计算量太小：
NPU Subgraph(Conv + ReLU)
  -> CPU Resize + Conv + Add + Softmax
```

## 10. 正确性设计

### 10.1 三阶段校验

#### 编译前

- Graph IR Schema 校验；
- Shape/Axis/Layout 一致性；
- Quantization 参数完整性；
- Constant 大小与 Tensor 描述匹配；
- Broadcast 合法性。

#### 编译后

- 编译器返回真实输入输出描述；
- 比较 Runtime Layout 与 IR 声明；
- 检查插入的转换和融合节点；
- 保存图摘要和编译器版本。

#### 执行验证

- 固定 Fixture 与 CPU Reference 对比；
- 每个输出设置绝对误差、相对误差和业务指标；
- 支持导出中间 Tensor；
- 用二分方式定位第一个偏离节点。

### 10.2 发布门槛

每个框架插件、模型和设备组合必须先通过正确性，再发布性能数据。当前人脸模型可使用：

```text
bestIndex = 4271
IoU >= 0.95
landmark count = 136
landmark maxError <= 0.04
```

## 11. 内存与零拷贝设计

### 11.1 支持的内存类型

```text
Host Heap
Shared Memory
DMA-BUF / Native Buffer
Camera Buffer
GPU Buffer/Image
NPU Device Buffer
```

`HFA_MemoryDesc` 应表达：

- 内存类型和 Handle；
- 总大小和 Offset；
- Stride；
- Cache 属性；
- Producer/Consumer；
- 生命周期；
- Fence；
- CPU Map 权限。

### 11.2 Layout 协商

编译阶段返回每个子图边界支持的 Layout，框架插件选择代价最低的共同布局：

```text
Framework Tensor Layout
        ↓ negotiate
Boundary Layout
        ↓
NPU Internal Layout
```

优先顺序：

1. 直接共享同一 Buffer 和 Layout。
2. 共享 Buffer，仅由 NPU 编译器插入内部转换。
3. 使用专用转换 Kernel。
4. 最后才回到 Host 做转换。

### 11.3 Buffer 复用

- Execution 创建时完成中间 Buffer 规划。
- 稳定 Shape 下复用输入输出和工作区。
- 多帧场景使用双 Buffer 或环形 Buffer。
- 不在每次推理中创建和销毁 Tensor。

## 12. 编译和缓存设计

### 12.1 Cache Key

Cache Key 至少包含：

```text
Graph Fingerprint
Constants Fingerprint
Shape Profile
Precision/Quantization Mode
Device SoC ID
NPU Driver Version
CANN Compiler Version
Operator Library Version
NNRt ABI Version
Compile Options
Security Domain
```

### 12.2 缓存层级

```text
进程内 CompiledModel Cache
        ↓
应用私有磁盘 Cache
        ↓
可选系统共享 Cache（需签名和权限隔离）
```

### 12.3 缓存失效

驱动、CANN、算子库、模型权重、Shape Profile 或编译选项变化时必须失效。禁止仅根据模型文件名复用 OM。

### 12.4 冷启动策略

- 安装期或首次空闲时预编译常用 Shape。
- 首次运行可先走 CPU/GPU，同时后台完成 NPU 编译。
- 编译完成后在下一帧安全切换。
- 大模型按 Chunk 编译和加载，限制单图峰值内存。

## 13. 执行与调度设计

### 13.1 执行模式

- `LATENCY_HIGH`：单次延迟优先。
- `THROUGHPUT_HIGH`：流水和批处理优先。
- `POWER_SAVER`：功耗和温升优先。
- `BACKGROUND`：低优先级、可被前台任务抢占。

### 13.2 异步流水

```text
Frame N:   NPU Infer
Frame N+1: CPU Preprocess
Frame N-1: GPU Render/Postprocess
```

通过 Fence 避免 CPU 主动轮询和全局同步。

### 13.3 多模型与多应用

Runtime 应向系统资源管理器提交：

- 优先级；
- 期望完成时间；
- 估算计算量；
- 内存占用；
- 可抢占性；
- 前台/后台属性。

系统仍负责最终 NPU、DDR 和 CPU 频率投票。框架插件不得直接控制硬件频率。

## 14. 回退设计

### 14.1 回退类型

| 类型 | 触发条件 | 处理方式 |
| --- | --- | --- |
| 编译前回退 | 算子、Shape、dtype 不支持 | 分图时留在框架 CPU/GPU |
| 编译回退 | CANN 编译失败 | 缩小子图或整体回退，记录错误 |
| 运行时回退 | 设备忙、内存不足、驱动错误 | 按策略重试或切换已准备好的 CPU/GPU Session |
| 精度回退 | Fixture 或在线校验失败 | 禁用对应编译产物和设备组合 |

### 14.2 禁止静默回退

生产模式可以自动回退，但必须记录结构化原因：

```text
modelId
graphFingerprint
node/subgraph
reasonCode
requestedBackend
actualBackend
compile/runtime stage
device/driver version
```

开发模式应允许配置 `FAIL_IF_NPU_NOT_USED`，用于性能验证和兼容性认证。

## 15. 可观测性设计

### 15.1 统一 Trace 前缀

```text
HFA/AnalyzeGraph
HFA/Partition/N
HFA/Compile/N
HFA/CacheHit/N
HFA/BindInput/N
HFA/LayoutConvert/N
HFA/Execute/N
HFA/Fallback/N
HFA/WaitFence/N
```

### 15.2 必须输出的指标

#### 图级

- 总节点数；
- NPU 节点数和占比；
- NPU 子图数量；
- CPU/GPU 回退节点；
- Layout 转换数量；
- 编译时间和 Cache 命中。

#### 执行级

- 每个子图排队、执行和等待时间；
- 上传和下载字节数；
- Layout 转换时间；
- Fence 等待；
- NPU 完成中断数量；
- CPU 调度和线程占用；
- DDR/NPU 频点与执行区间关联。

#### 业务级

- P50/P95/P99；
- 首帧延迟；
- 稳态帧率；
- 功耗和温升；
- 错误和回退率。

### 15.3 开发工具输出

建议提供统一报告：

```text
Model Summary
Partition Visualization
Unsupported Ops
Fallback Reasons
Boundary Tensor Sizes
Layout Conversions
Compile Cache Status
Per-subgraph Timeline
Optimization Suggestions
```

## 16. 版本与兼容设计

### 16.1 稳定 C ABI

- 接口结构包含 `structSize` 和 `version`。
- 新字段只追加，不改变旧字段含义。
- Runtime 支持查询 Feature Bit。
- 插件和 Runtime 启动时完成版本协商。

### 16.2 语义版本

分别维护：

```text
HFA ABI Version
Graph IR Version
Operator Semantics Version
Cache Format Version
Profiling Schema Version
```

### 16.3 兼容矩阵

官方发布：

```text
Framework Version
Plugin Version
HarmonyOS Version
NNRt Version
CANN Version
Kirin SoC
Supported Precision
Certified Models
Known Fallbacks
```

## 17. 安全与模型保护

- 插件运行在应用进程，但不能直接访问驱动私有接口。
- NNRt 校验 Graph 和 Buffer 边界，防止非法 Shape 和越界访问。
- 编译 Cache 存在应用私有目录并绑定签名身份。
- 系统共享 Cache 需要模型签名、租户隔离和版本校验。
- 支持加密模型或受保护权重映射。
- Profiling 默认不导出权重和完整中间 Tensor。

## 18. 各框架插件设计

### 18.1 MNN Plugin

- 使用正式的 Huawei NPU ForwardType，停止用 `USER_0/USER_1` 表达产品语义。
- 在 Session Resize 阶段提交完整 Pipeline Graph。
- 根据 Partition Plan 创建 `HuaweiSubgraphExecution`。
- MNN CPU/OpenCL 作为回退 Backend。
- 正确处理 NC4HW4，并通过 Layout 协商避免隐式解包错误。

### 18.2 ncnn Plugin

- 将 ncnn Layer Graph 转为 Huawei Graph IR。
- 保留 Vulkan/CPU Pipeline 作为回退。
- 对动态输入尺寸建立有限 Shape Profile。

### 18.3 LiteRT Delegate

- 实现标准 LiteRT Delegate 接口。
- Delegate 内部不直接维护完整 CANN 映射，而调用 HFA。
- 由 LiteRT 完成节点替换，HFA 返回推荐子图和边界要求。

### 18.4 ONNX Runtime Execution Provider

- 实现 Huawei Execution Provider。
- 使用 ORT Graph Viewer 导出子图到 HFA。
- 使用 I/O Binding 对接共享 Buffer。

### 18.5 MindSpore Lite

- 作为系统内置参考实现，优先使用 NNRt 原生 Graph/Offline Model 能力。
- 用于验证 HFA/NNRt 的正确性和性能上限。

## 19. 端到端流程

### 19.1 首次加载

```text
App Load Model
  -> Framework Parse
  -> Plugin Export Graph IR
  -> HFA Validate
  -> HFA AnalyzeGraph
  -> Return Partition Plan
  -> Framework Replace Subgraphs
  -> HFA Check Cache
     -> Hit: Load Compiled Model
     -> Miss: CANN Compile -> Save Cache
  -> Create Execution and Reusable Buffers
```

### 19.2 稳态执行

```text
Framework Prepare Input
  -> Import/Bind Shared Buffer
  -> Wait Producer Fence
  -> HFA ExecuteAsync
  -> NNRt/CANN Submit NPU Subgraph
  -> Signal Fence
  -> Framework Continues CPU/GPU Nodes
  -> Produce Output
```

### 19.3 动态 Shape

```text
Input Shape
  -> Match Existing Profile
     -> Yes: Execute
     -> No:
        ├─ Use padded compatible profile
        ├─ Compile new profile asynchronously
        └─ Temporary CPU/GPU fallback
```

## 20. 运作与责任划分

### 20.1 华为负责

- HFA API、Graph IR 和稳定 ABI。
- NNRt/CANN Runtime 和 NPU 编译器。
- MNN、ncnn、LiteRT、ORT 官方参考插件。
- 算子库、成本模型、缓存、内存和 Profiling。
- 设备兼容矩阵和系统联合验证。
- 正确性 Fixture、模型集和性能基准。

### 20.2 框架社区负责

- 提供稳定的图、Tensor 和 Backend 插件接口。
- 跟踪框架内部 Graph/Op 变化。
- 保留 CPU/GPU 回退能力。
- 审核和合入华为官方插件。

### 20.3 应用开发者负责

- 选择模型、精度和性能模式。
- 提供代表业务输入的正确性和性能 Fixture。
- 根据兼容矩阵选择发布范围。
- 处理功能级降级，不直接处理 NPU 驱动细节。

### 20.4 认证机制

建议建立 `Huawei NPU Ready`：

- 正确性通过；
- NPU 子图命中率达标；
- 无静默回退；
- 冷启动和稳定态性能达标；
- 功耗和温升达标；
- 指定系统/SoC 组合通过回归。

## 21. 分阶段落地

### 阶段 0：统一观测，4~6 周

- 给现有 MNN USER_0/USER_1、NNRt 路径增加统一 Trace。
- 输出节点命中、子图数量、边界、拷贝和 Layout 转换。
- 建立固定人脸、关键点和小 Transformer Fixture。

交付：能够解释“为什么慢”和“是否真的运行在 NPU”。

### 阶段 1：HFA 最小接口，8~12 周

- 定义 Graph IR、Capability、Compile、Execute 和 Profiling C ABI。
- 先实现静态 Shape、FP16/FP32、同步执行。
- MNN Plugin 使用 HFA 替代逐卷积 Delegate。

交付：人脸/关键点模型正确，稳定态性能接近 MindSpore Lite/NNRt。

### 阶段 2：统一分图、缓存和内存，12~16 周

- 引入成本模型和 Partition Plan。
- 实现编译缓存、共享内存、Layout 协商。
- 增加 LiteRT Delegate 和 ncnn Plugin。

交付：多框架复用同一优化，减少 CPU/NPU 边界和首帧编译时间。

### 阶段 3：异步、动态 Shape 和系统调度，16 周以上

- 异步执行和 Fence。
- 多 Shape Profile 和后台编译。
- 多模型、多应用优先级和资源调度。
- ONNX Runtime EP 和大模型 Chunk 支持。

交付：面向相机、直播、语音和端侧大模型的生产能力。

## 22. 验收指标

### 22.1 正确性

- 固定 Fixture 全部通过。
- 中间 Tensor 可定位首个偏离节点。
- 不允许只检查“执行成功”。

### 22.2 性能

- 人脸/关键点推理稳定态与 MindSpore Lite/NNRt 差距不超过 10%。
- 相比当前 USER_1，NPU 完成中断次数下降 50% 以上。
- 单模型 NPU 子图数量优先控制在 1~3 个。
- 稳态执行中不重复编译。
- P95 不超过 P50 的 1.5 倍，具体场景可调整。

### 22.3 内存

- 输入输出 Buffer 稳态复用。
- 可观测每个边界的拷贝字节数。
- 支持至少一种 Camera/Native Buffer 零拷贝通路。

### 22.4 开发体验

- 应用不需要直接调用 CANN 驱动接口。
- 切换 NPU/CPU/GPU 不改变业务模型接口。
- 一份报告能够解释未下沉算子和性能损失。

## 23. 风险与对策

| 风险 | 影响 | 对策 |
| --- | --- | --- |
| 通用 IR 语义不足 | 输出错误 | 显式 Layout/Axis/Quantization；语义版本化；Fixture 门禁 |
| 分图成本模型不准 | NPU 比 CPU 更慢 | 收集实机数据校准；支持框架成本提示和规则覆盖 |
| ABI 演进困难 | 插件频繁重编 | C ABI、Feature Bit、结构追加、兼容测试 |
| Cache 跨版本失效 | 加载失败或结果异常 | 完整 Cache Key、签名校验、原子替换 |
| 动态 Shape 编译抖动 | 首帧和长尾恶化 | Profile、Padding、后台编译和临时回退 |
| 官方插件跟不上框架版本 | 生态覆盖不足 | LTS 版本矩阵、上游合入、自动化 CI |
| 多应用争抢 NPU | 帧率和功耗不稳定 | 系统调度、优先级、Deadline、可抢占设计 |

## 24. 需要优先决策的问题

1. HFA 是作为 NNRt 新接口直接发布，还是作为 NNRt 之上的独立 Framework Adapter Kit。
2. Graph IR 复用 MindIR、NNRt 内部 IR，还是定义独立稳定交换格式。
3. CPU/GPU 成本由框架上报还是由系统 Profile 数据估算。
4. 编译 Cache 是否允许系统跨应用共享。
5. 第一批官方支持的框架版本和模型范围。
6. CANN 高级能力通过 HFA 扩展配置开放到什么程度。

## 25. 推荐决策

建议采用以下产品边界：

```text
对应用开发者：
使用原有 MNN/ncnn/LiteRT/ORT API，只选择 Huawei NPU Backend。

对框架开发者：
使用 Huawei Framework Adapter API，不直接适配 CANN 私有接口。

对高级性能开发者：
通过受控的 HFA Extension/CANN Kit 使用自定义算子和精细调度。

对系统：
NNRt/CANN 统一负责图编译、缓存、内存、执行和硬件调度。
```

这不是简单增加一层包装，而是重新收敛性能优化和正确性责任：框架插件负责语义输入，华为 Runtime 负责 NPU 最优执行，原框架负责通用后端回退。

## 26. 参考材料

- 当前 Demo：`HarmonyBeautyDemo/docs/performance/mnn_hiai_user0_user1.md`
- 当前代码审查：`HarmonyBeautyDemo/docs/performance/mnn_npu_code_path_audit.md`
- 当前性能对比：`HarmonyBeautyDemo/docs/performance/nnrt_vs_mnn_npu_report.md`
- HarmonyOS Neural Network Runtime Kit：<https://developer.huawei.com/consumer/en/doc/harmonyos-guides/neural-network-runtime-kit-introduction>
- HarmonyOS NNRt 开发指导：<https://developer.huawei.com/consumer/cn/doc/harmonyos-guides-V13/neural-network-runtime-guidelines-V13>
- HarmonyOS MindSpore Lite 推理：<https://developer.huawei.com/consumer/cn/doc/harmonyos-guides-V13/mindspore-lite-guidelines-V13>
- Google LiteRT NPU Delegate：<https://ai.google.dev/edge/litert/android/npu>
- Android NNAPI 迁移指南：<https://developer.android.com/ndk/guides/neuralnetworks/migration-guide>

