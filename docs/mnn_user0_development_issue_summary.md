# MNN HiAI USER_0 开发问题总结

## 1. 文档目的

本文总结 HarmonyBeautyDemo 将 MNN 模型通过 mobiInfer `MNN_FORWARD_USER_0` 接入华为 HiAI/NPU 时遇到的主要问题、定位过程、修复方法和当前状态。

这里的 USER_0 不是 MNN 官方统一定义的通用 NPU Delegate，而是 mobiInfer 分支注册的 HiAI 整图后端：

```text
.mnn
  -> MNN Interpreter
  -> MNN_FORWARD_USER_0 / NPUBackend
  -> MNN Op 转换为 HiAI/GE Operator
  -> GE Graph
  -> CreateModelBuff
  -> BuildIRModel
  -> LoadModelSync
  -> AiModelMngerClient::Process
  -> Kirin NPU
```

当前结论是：USER_0 已经能够正确执行人脸检测和 68 点模型，固定输入数值通过校验，真实后摄端到端性能与原生 NPU 基本持平。但开发过程中暴露出的算子语义、后端识别、测试方法和可观测性问题，仍需要沉淀为长期工程能力。

## 2. 问题总览

| 编号 | 问题 | 典型现象 | 状态 |
|---|---|---|---|
| U0-01 | 最初选成 USER_1 | 标成“MNN-NPU”，实际是 CPU + 逐卷积 HiAI | 已解决 |
| U0-02 | USER_0/USER_1 含义不透明 | 无法从界面和日志判断实际路径 | 已解决 |
| U0-03 | 后端注册和运行库依赖 | Session 创建失败或后端不匹配 | 已解决 |
| U0-04 | BuildIRModel 对图语义敏感 | 图转换完成但在线编译失败 | 当前模型已解决 |
| U0-05 | 图能编译但输出错误 | 人脸框、置信度和最佳候选不正确 | 已解决 |
| U0-06 | 三维输出物理布局错误 | `[1,N,C]` 被按 `[1,C,1,N]` 读取 | 已解决 |
| U0-07 | ConvertTensor 只看 shape | 相同 shape 下仍选错 layout 转换 | 已解决 |
| U0-08 | Flatten 语义不一致 | HiAI 默认 Flatten 与 MNN axis/endAxis 不一致 | 已解决 |
| U0-09 | Softmax axis 处理不完整 | 负轴、补维后的物理轴存在歧义 | 已解决 |
| U0-10 | 自检过弱 | 错误输出仍可能显示“模型可运行” | 已解决 |
| U0-11 | 大输出跨 NAPI 搬运 | 人脸模型执行快，但完整调用没有优势 | 待优化 |
| U0-12 | 每帧临时缓冲和 Tensor | 增加 memcpy、分配和长尾 | 待优化 |
| U0-13 | 公共 ROI 预处理过重 | 约 31 ms，掩盖后端差异 | 待优化 |
| U0-14 | 单帧截图作为性能结论 | 曾出现 NPU 22 ms、U0 43 ms 的错误口径 | 已纠正 |
| U0-15 | 无效帧混入统计 | 没有新相机帧的快速返回拉低均值 | 已纠正 |
| U0-16 | 固定采集顺序造成偏差 | 交换顺序后 NPU/U0 快慢反转 | 已纠正 |
| U0-17 | 缺少 NPU 内部观测 | 无法拆分 MNN、HiAI、驱动和 NPU kernel | 待补齐 |
| U0-18 | 修复源码复现链不完整 | Demo 带修复后的 `libMNN.so`，后端补丁仍需独立固化 | 待治理 |

## 3. 第一阶段：接入路径选错

### 3.1 把 USER_1 当成整图 NPU

Demo 最初使用 `MNN_FORWARD_USER_1`，界面统一显示为“MNN-NPU”。代码审查后确认：

| 后端 | 实际实现 | 行为 |
|---|---|---|
| USER_0 | `NPUBackend` | 尽量将完整 MNN 图转换成 HiAI/GE 图 |
| USER_1 | `HiAIDelegateBackend` | CPU 主图，仅将部分卷积逐个委托给 HiAI |

USER_1 不是 USER_0 的另一个性能档位。它存在大量 CPU/NPU 边界、多次小图提交和同步，而且后端代码中还有全局 70 个 HiAI 卷积上限。超过上限或不支持的算子由 CPU 执行，性能会随模型加载顺序和进程历史变化。

早期 USER_1 固定图结果中，人脸和关键点推理分别约为原生 NPU 的 6.3 倍和 6.9 倍。这些数据只能证明逐卷积委托路径较慢，不能用来证明 MNN 整图后端比 NNRT 慢。

### 3.2 修复方法

- Demo 同时定义 `NpuUser0`、`NpuUser1` 和 GPU 三种 MNN Session；
- USER_0 对应 `MNN_FORWARD_USER_0`，USER_1 对应 `MNN_FORWARD_USER_1`；
- 界面分别显示 `MNN-U0` 和 `MNN-U1`；
- `getSessionInfo(..., BACKENDS, ...)` 校验主后端编号，不匹配就拒绝运行；
- Trace 名称中明确包含 `MNNNPU/User0` 或 `MNNNPU/User1`。

需要注意：主后端编号只能证明 Session 选择了 USER_0 或 USER_1，不能替代逐算子覆盖率统计。

## 4. 第二阶段：后端能启动但 BuildIRModel 失败

USER_0 会将整图转换为 GE Graph，再调用 `BuildIRModel` 在线编译。这个阶段曾出现两类失败：

1. MNN 侧已经创建了 HiAI Operator，但属性、shape、format 或广播关系不被设备 DDK 接受；
2. 为修复 Softmax 尝试展开为 `ReduceMax -> Sub -> Exp -> ReduceSum -> RealDiv`，代码可以编译，但设备端 DDK 在 BuildIRModel 阶段拒绝该组合图。

这说明：

```text
MNN算子转换成功 != GE图合法
GE图合法 != 当前手机DDK能够编译
BuildIRModel成功 != 最终数值正确
```

定位过程中为 `CreateModelBuff`、`BuildIRModel`、模型缓存、`LoadModelSync` 和 `Process` 增加了分阶段日志。当前人脸检测和68点两个模型可以完成在线编译和加载，但对新模型仍应将 BuildIRModel 失败视为常规兼容性问题处理。

## 5. 第三阶段：BuildIRModel 成功但输出错误

这是 USER_0 最关键的问题。早期固定输入曾得到：

```text
预期 bestIndex = 4271
实际 bestIndex = 4382
IoU = 0.328
confidence = 0.097
```

同时 boxes 输出也有明显误差，因此问题不可能只靠应用层调整 Softmax、类别索引或 anchor 顺序解决。根因位于更早的 layout、Flatten、ConvertTensor 或图输出读取语义。

### 5.1 三维输出物理布局

MNN 的逻辑输出是：

```text
[1, candidates, channels]
```

HiAI 对低维 Tensor 进行了物理补维，终端缓冲表现为：

```text
[1, channels, 1, candidates]
```

原实现直接按 MNN 逻辑顺序读取，导致候选维和通道维交错。修复后，USER_0 输出回读针对 rank=3 的 float Tensor 执行：

```text
destination[candidate * channels + channel]
  = source[channel * candidates + candidate]
```

这一步恢复了人脸 scores 和 boxes 的候选优先布局。

### 5.2 ConvertTensor 只依据 shape 推断 Permute

原 `NPUConvertTensor` 通过输入、输出 shape 是否能由某个维度排列匹配来选择 `Permute`。问题是：

- NCHW、NC4HW4 和 NHWC 可能具有相同逻辑 shape；
- 相同 shape 不代表物理 layout 相同；
- NC4HW4 还涉及通道按 4 打包，普通 Reshape 或恒等 Permute 不能完成解包。

修复后优先读取 MNN Tensor 的 `dimensionFormat`，显式区分：

```text
NCHW/NC4HW4 -> NHWC
NHWC -> NCHW/NC4HW4
3D NCH <-> NHC
相同 format 或 rank <= 2
```

只有无法通过格式语义判断时，才回退到 shape 匹配。

### 5.3 Flatten 语义不一致

MNN Flatten 包含明确的 `axis/endAxis` 语义，而 HiAI `Flatten` 使用自身默认规则。直接一对一映射会在后续分类头中改变 Tensor 逻辑顺序。

修复方式是不用 HiAI 默认 Flatten，而是读取 MNN 已计算出的输出 shape，构造 shape Const，并使用显式 `Reshape`。这样 Flatten 的结果完全服从 MNN 图的输出定义。

### 5.4 Softmax axis

原实现直接把 MNN axis 传给 HiAI。修复后：

- 负 axis 先按照 MNN 输入 rank 归一化；
- 越界 axis 直接返回 `NOT_SUPPORT`；
- Softmax 使用归一化后的逻辑 axis；
- 不再用设备不支持的复杂算子展开替代简单 Softmax。

当前人脸模型最终输出正确，说明这组 layout、Flatten、Softmax 和输出回读修复对现有模型有效。但它不是通用算子兼容性的终点，新模型仍需要逐模型验证。

## 6. 第四阶段：正确性验证方法不足

早期自检只检查“有输出、置信度较高、框尺寸为正”。这种检查无法发现：

- 最佳候选 anchor 发生变化；
- 人脸框整体偏移；
- scores 和 boxes 局部乱序；
- 68 点坐标整体看似合理但数值已经漂移。

后续建立固定 Tensor 自检，检查：

| 模型 | 验证项 |
|---|---|
| 人脸检测 | 输出长度、分数范围、最佳候选、置信度、box IoU |
| 68 点 | 136 个 float、最大绝对误差、连续运行漂移 |

当前固定输入结果：

| 指标 | 原生 NPU | MNN-U0 |
|---|---:|---:|
| 最佳候选 | 4251 | 4251 |
| 置信度 | 0.9980 | 0.9980 |
| 人脸框 IoU | 0.954 | 0.954 |
| 68 点最大绝对误差 | 0.00168 | 0.00168 |
| 连续运行漂移 | 0 | 0 |

固定输入通过后，才允许进入实景性能测试。这个顺序避免了“错误图跑得快”被误判成性能优化。

## 7. 第五阶段：性能测试口径多次误导

### 7.1 单帧截图不是性能结论

早期界面曾显示：

```text
原生 NPU 22 ms
MNN-U0   43 ms
```

它们来自不同时间点的一次界面刷新，混入了相机取帧、ROI变化、调度和温度波动，不能代表均值。该结论已经撤销。

### 7.2 Frame/Total 包含空帧

相机生产者与推理消费者节奏不同。没有取得新帧时也会产生很短的 `Frame/Total`，如果直接统计全部事件，会人为降低平均耗时。

正式解析只保留同一个 trace ID 下同时存在以下阶段的完整帧：

```text
Frame/Total
Face/Infer
Landmarks/Infer
```

### 7.3 后端按钮和相机状态曾造成无效样本

后端选择栏可横向滑动，自动点击坐标曾选到 USER_1 或 CPU，却按 USER_0/NPU 命名保存 trace。相机切换按钮也曾把“当前镜头”和“点击后的目标镜头”混在一起。

此后每次采集都需要同时验证：

- 左上角当前相机状态；
- 界面高亮的后端名称；
- trace 中 `Frame/Total/<backend>` 的真实标记；
- 人脸数、68点和固定画面完整性。

### 7.4 固定顺序造成窗口偏差

首轮后摄数据中，NPU 为 50.25 ms，U0 为 53.35 ms；交换顺序后，U0 为 45.47 ms，NPU 为 48.79 ms，快慢关系反转。

两轮合并后：

| 指标 | 原生 NPU | MNN-U0 |
|---|---:|---:|
| 完整链路加权均值 | 49.64 ms | 49.90 ms |
| 两次模型推理合计 | 9.58 ms | 8.95 ms |
| 公共 ROI 预处理 | 31.16 ms | 31.86 ms |

端到端只差 0.5%，低于公共 CPU 预处理的时间窗波动。当前不能再表述为“原生 NPU 稳定比 USER_0 快 6%”。

## 8. 当前仍未解决的性能问题

### 8.1 人脸大输出跨 NAPI

人脸模型每帧返回：

```text
scores: 4420 x 2
boxes:  4420 x 4
合计约 106 KB
```

当前链路为：

```text
NPU输出
  -> MNN copyToHostTensor
  -> C++ vector
  -> NAPI ArrayBuffer
  -> ArkTS Float32Array
  -> ArkTS SSD候选框解码
```

固定输入中，U0 人脸 `RunSession` 并不慢，但 NAPI/结果封装残差约 3.6 ms。后续应在 C++ 中完成最佳候选和 box 解码，只返回 `score + box + index`。

### 8.2 Tensor 和缓冲区重复创建

当前每帧仍会创建 Host Tensor、vector、ArrayBuffer 和 `buffer.slice(0)`。应让 Session 长期持有输入、输出和双缓冲，稳态阶段避免大块内存分配。

### 8.3 公共 ROI 预处理占主导

关键点 ROI 裁剪、resize、颜色转换和归一化约 31 ms，而两次模型推理合计约 9 ms。继续只优化 USER_0 `runSession()` 的收益有限。

优先方案：

1. 将 ROI 预处理移到 Native C++；
2. 使用 MNN ImageProcess、NEON、GPU shader 或 NPU图内前处理；
3. 直接写入复用的 MNN 输入 Tensor；
4. 降低人脸检测频率，中间帧只更新关键点或跟踪结果。

### 8.4 缺少同层级 NPU 执行观测

当前可以看到 MNN `TensorUpload/RunSession/TensorDownload`，但无法继续拆分：

```text
MNN调度
HiAI Process同步等待
驱动排队
NPU kernel执行
```

原生 NNRT `predict()` 内部也缺少同层级标记。因此目前只能比较框架完整调用，不能断言某个 NPU kernel 本身更快。

### 8.5 修复源码需要固化

GitHub Demo 已包含当前可运行的 `libMNN.so`，但 USER_0 的后端修复涉及 mobiInfer 中的：

```text
NPUBackend.cpp
NPUConvertTensor.cpp
NPUFlatten.cpp/.hpp
NPUSoftmax.cpp
```

这些补丁应形成独立提交或可应用 patch，并记录对应 mobiInfer commit、DDK版本和编译参数。否则只有二进制可以复现运行，无法稳定复现构建。

## 9. 当前状态

USER_0 当前已达到：

- Session 明确选择 `MNN_FORWARD_USER_0`；
- 人脸和68点模型均能 BuildIRModel、Load和Process；
- 固定输入最佳候选、IoU和68点误差通过；
- 实景持续输出1张人脸和68点；
- 两轮后摄端到端均值与原生NPU基本持平；
- 纯模型执行合计略快于当前原生NPU路径。

尚不能宣称：

- USER_0 已兼容任意 MNN 模型；
- 所有算子都具备完整 layout/axis/量化语义；
- 没有任何隐藏 CPU fallback；
- USER_0 在功耗上一定优于原生 NNRT；
- 当前测到的差异等于纯 NPU kernel 差异。

## 10. 后续优先级

1. 固化并提交 mobiInfer USER_0 后端补丁及构建说明；
2. 在 C++ 完成人脸候选框解码，移除106 KB每帧NAPI返回；
3. 复用 MNN Host Tensor、输出容器和 ArrayBuffer；
4. 将关键点 ROI 预处理下沉 Native/GPU/NPU；
5. 输出每个模型的 USER_0 算子覆盖率、子图数和 CPU fallback 列表；
6. 增加 `AiModelMngerClient::Process()` 同层 trace；
7. 使用 ABBA 交替顺序、多轮固定输入、温度和功耗统计形成最终基准。

## 11. 关联材料

- `mnn_hiai_user0_user1.md`：USER_0与USER_1架构区别；
- `mnn_npu_code_path_audit.md`：早期错误后端路径审计；
- `npu_vs_mnn_user0_fixed_input_analysis.md`：固定Tensor正确性和性能；
- `rear_camera_five_backend_quick_comparison.md`：正式后摄五后端及反向复测；
- `trace_markers.md`：当前应用和MNN内部Trace点。
