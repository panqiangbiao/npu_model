# 手机端侧 AI 自有模型开放能力竞品调研

> 调研对象：华为、Apple、Google/Pixel、vivo、OPPO、小米
> 调研时间：2026-08-31  
> 调研目的：判断第三方应用能否部署自己的模型、能否使用 NPU，以及能否调用系统内置大模型。

## 1. 结论摘要

“手机支持 AI”不能直接等价为“第三方应用可以调用 NPU”。需要拆成三层能力：

1. **自有模型运行**：应用能否携带或下载自己的模型权重并完成推理。
2. **专用硬件加速**：运行时能否把模型稳定地下沉到 NPU/Neural Engine/DSP，而不只是 CPU/GPU。
3. **系统模型调用**：应用能否直接使用系统维护的基础模型，而不需要携带权重。

综合公开资料：

- **Apple 的完整度最高**：Core ML/Core AI 支持自有模型，系统负责 CPU、GPU、Neural Engine 调度，同时通过 Foundation Models 开放系统模型。
- **Google 的框架生态成熟但 NPU 路径碎片化**：LiteRT 支持自有模型，Gemini Nano/AICore 支持系统模型；NPU 加速仍明显依赖 SoC 厂商 Delegate 和具体设备。
- **vivo VCAP 是国产厂商中公开资料最完整的自有模型异构推理平台**：支持模型转换、CPU/GPU/DSP/NPU 推理和端侧训练，但存在私有格式、账号申请和机型适配成本。
- **OPPO 具备端侧 AI 和模型加速能力，但当前公开技术资料不足以证明其已形成面向普通开发者、跨机型稳定可用的统一 NPU SDK**。
- **小米 MACE 支持自有模型和 CPU/GPU/Hexagon DSP，但不能等同于 HyperOS 统一 NPU 服务**；系统智能体开放也不能替代应用自有模型运行时。

## 2. 评价口径

| 评价项 | 关键问题 |
| --- | --- |
| 自有模型 | 是否支持应用自带或动态下载模型 |
| 输入格式 | 是否接收 PyTorch、TensorFlow、ONNX、TFLite 等常见来源 |
| 专用 AI 硬件 | 是否明确支持 NPU/Neural Engine/DSP |
| 调度控制 | 开发者能否指定或影响后端，是否有自动分图和回退 |
| 系统基础模型 | 是否开放系统维护的大模型给第三方应用 |
| 工具链 | 是否提供转换、量化、编译、缓存、分析和调试工具 |
| 开放方式 | 普通开发者可直接使用，还是需要申请、合作或白名单 |
| 跨机型一致性 | 模型和接口能否跨 SoC、跨机型稳定运行 |

## 3. 能力与 SDK 汇总

| 厂商 | 支持应用自有模型 | 自有模型 SDK/框架名称 | 明确支持的加速后端 | 系统模型 SDK/能力 | 主要限制 |
| --- | --- | --- | --- | --- | --- |
| 华为/HarmonyOS | 是 | 应用层：**MindSpore Lite Kit**、**HiAI Foundation Kit**；硬件桥接：**NNRt**；深度优化：**CANN Kit** | MindSpore Lite 可做 CPU + NNRt 异构；NNRt 连接 NPU；CANN Kit 为麒麟平台后端 | HiAI Engine 及 HarmonyOS AI 开放能力，具体系统大模型接口需按版本确认 | SDK 层级较多；NNRt 算子和同步执行能力有限；CANN Kit 与麒麟设备强相关 |
| Apple | 是 | **Core ML**；生成式/新架构可用 **Core AI** | CPU、GPU、Neural Engine，由系统调度 | **Foundation Models** | 不能保证任意算子强制进入 Neural Engine |
| Google/Pixel | 是 | **LiteRT**（原 TensorFlow Lite）；上层可配合 MediaPipe | CPU、GPU；NPU 依赖 Qualcomm/Pixel/MediaTek 等 Delegate | **ML Kit GenAI API + AICore/Gemini Nano** | NPU Delegate、算子和机型覆盖碎片化；NNAPI 已废弃 |
| vivo | 是 | **VCAP SDK**（Java/C++），模型转换为 `vaimlite` | CPU、GPU、DSP、NPU | 蓝心及 vivo 端 AI 能力，公开端侧调用边界需确认 | 转换平台账号、私有模型格式和具体机型需要申请/验证 |
| OPPO | 有历史公开支持，当前需确认 | **AI Boost**（历史公开的自有模型加速框架）；**AIUnit** 主要是成品算法 API | 历史资料表述为端侧硬件加速，当前 NPU/算子/机型矩阵未公开 | 安第斯大模型、端侧 AI 能力开放服务 | 当前未找到面向普通开发者的完整 SDK 下载、模型格式及兼容矩阵 |
| 小米 | 是 | **MACE**；也可使用 Android 通用 LiteRT/MNN 等 | MACE 主要是 CPU、OpenCL GPU、Hexagon DSP | HyperOS 智能体、Skills、MCP 生态 | MACE 不等同于 HyperOS 统一 NPU SDK，进入 NPU 仍依赖芯片运行时 |

### 3.1 只看自有模型 SDK

| 厂商 | 建议记录的 SDK 名称 | 判断 |
| --- | --- | --- |
| 华为/HarmonyOS | `MindSpore Lite Kit` / `HiAI Foundation Kit`；底层 `NNRt`；高级 `CANN Kit` | 官方能力完整，但应用层入口尚未像 Core ML 一样收敛为单一框架 |
| Apple | `Core ML` / `Core AI` | 官方、公开、可直接集成 |
| Google/Pixel | `LiteRT` | 官方、公开；NPU 需要额外 Delegate |
| vivo | `VCAP SDK` | 官方平台；通常需要模型转换和账号/合作流程 |
| OPPO | `AI Boost` | 历史公开名称；当前可获得性和支持范围必须向 OPPO 复核 |
| 小米 | `MACE` | 小米开源框架；主要后端不是统一 HyperOS NPU |

> 注：表中的“支持”只代表存在公开框架或接口证据，不代表任意模型、任意算子、任意机型都能进入 NPU。

## 4. 华为 / HarmonyOS

华为公开的端侧自有模型能力分为四层：

| 层级 | SDK/Kit | 作用 |
| --- | --- | --- |
| 应用推理框架 | **MindSpore Lite Kit** | 加载和执行模型，支持 CPU 与 NNRt 异构推理；NNRt 不支持的算子可以调度到 CPU |
| 应用模型服务 | **HiAI Foundation Kit** | 提供模型优化、转换和端侧部署，并强调跨硬件异构兼容 |
| 硬件运行时 | **Neural Network Runtime Kit（NNRt）** | 用 `OH_NN*` Native API 完成模型构造、编译、Tensor 管理和执行，连接上层框架与 AI 加速芯片 |
| NPU 深度优化 | **CANN Kit** | 麒麟平台 NPU 后端，支持 Ascend C 自定义算子和更深的性能、功耗优化 |

NNRt 支持在线构图、模型缓存和硬件专用离线模型，也支持共享内存零拷贝；但官方资料指出其仅提供 AI 加速硬件推理，不提供 CPU 后端，目前公共算子数量有限，并且只支持同步推理。因此应用通常更适合通过 MindSpore Lite 使用 CPU + NNRt 异构路径，而不是自行用 `OH_NN*` 重建整个复杂模型。

华为当前的主要差异不是缺少能力，而是开发者需要理解多套接口和层级。对外产品形态还需要进一步收敛成类似 Core ML 的统一应用入口。

## 5. Apple

### 5.1 自有模型

Core ML 支持把其他训练框架产生的模型转换为 Core ML 模型并集成到应用。Core AI 面向新一代深度学习和生成式模型，使用 `.aimodel`，支持模型专门化、缓存、状态化执行、量化和调试。

### 5.2 Neural Engine

Core ML/Core AI 可以在 CPU、GPU 和 Neural Engine 之间调度。开发者可以通过 `MLComputeUnits` 限定允许使用的计算单元，例如 `.cpuAndNeuralEngine`，但这不是“全部算子强制进入 Neural Engine”；不支持的算子仍可能由 CPU 执行。

### 5.3 系统模型

Foundation Models 为第三方应用提供 Apple Intelligence 端侧模型能力，形成“应用自有模型”和“系统基础模型”两条并行路径。

### 5.4 评价

- 优势：模型格式、编译、运行时、硬件和分析工具统一；跨 Apple 芯片的一致性强。
- 限制：后端由系统主导，开发者不能把任意算子强制放入 Neural Engine；模型仍受算子和设备能力约束。

## 6. Google / Pixel

### 6.1 自有模型

Google 推荐使用 LiteRT 运行自定义端侧模型，并通过 MediaPipe、ML Kit 等上层能力完成常见视觉、音频和文本任务。

### 6.2 NPU

LiteRT 的 NPU 能力依赖芯片厂商 Delegate。Google 官方资料明确列出 Qualcomm AI Engine Direct Delegate；不同 SoC 和手机的 Delegate、算子覆盖和性能表现可能不同。

Android 过去尝试用 NNAPI 提供统一硬件接口，但 NNAPI 已在 Android 15 废弃。Google 当前推荐 LiteRT、GPU Delegate，以及厂商提供的硬件 Delegate。

### 6.3 系统模型

AICore 负责在系统侧管理 Gemini Nano，第三方应用通过 ML Kit GenAI API 使用摘要、改写、校对、图像描述、语音识别和通用 Prompt 等能力。

### 6.4 评价

- 优势：通用模型生态、Play 服务更新能力和系统大模型开放较完整。
- 限制：Android 硬件碎片化导致 NPU 路径不如 Apple 统一，必须建立设备能力检测和 CPU/GPU 回退。

## 7. vivo

VCAP 是 vivo 自研的端侧 AI 计算加速平台，官方资料说明其支持 CPU/GPU/DSP/NPU 异构计算、端侧推理和离线训练，并覆盖高通、三星、联发科平台。

典型接入流程：

1. 准备 TensorFlow `.pb`、TFLite 或 ONNX 模型。
2. 使用 vivo 转换平台生成 `vaimlite` 私有模型。
3. 集成 VCAP Java/C++ SDK。
4. 在目标机型验证后端选择、精度和性能。

限制包括：模型转换平台账号可能需要 vivo 侧注册；模型进入私有格式；NPU 支持与具体 SoC 和算子有关。因此它更接近“厂商合作型 SDK”，而不是完全无门槛的 Android 标准接口。

## 8. OPPO

OPPO 开放平台当前列出“端侧 AI”“AI 能力开放服务”和“安第斯大模型”等入口。历史公开资料中：

- AIUnit 提供视觉、视频、音频和感知等成品算法 API。
- AI Boost 用于把开发者算法模型部署到 OPPO 终端并利用端侧硬件加速。

但当前公开资料没有像 Apple、Google、vivo 那样完整呈现模型格式、SDK 下载、NPU 算子列表、机型矩阵和回退策略。因此现阶段只能确认 OPPO 具备相关平台和技术，不能据此认定普通第三方应用可在所有 OPPO 机型上直接使用统一 NPU 运行时。

## 9. 小米

MACE 是小米开源的移动端异构推理框架，支持 TensorFlow、Caffe、ONNX 等模型来源，提供 ARM CPU、OpenCL GPU、Qualcomm Hexagon DSP 等后端，以及内存复用、模型保护和功耗相关优化。

需要注意：

- MACE 是跨平台推理框架，不等同于 HyperOS 系统级 NPU 服务。
- 在小米手机上使用 LiteRT、MNN、ONNX Runtime、ncnn 或 MACE，都可能运行自有模型，但是否进入 NPU 取决于具体芯片运行时和 Delegate。
- 小米智能体、Skills、MCP 开放属于系统服务生态，不是 APK 加载自有权重的底层推理接口。

## 10. 对华为端侧 AI 开放能力的启示

华为已经具备 NNRT、HiAI、CANN Kit、MNN+CANN 等技术路径，真正影响第三方采用的不是“有没有 NPU”，而是能否降低接入和验证成本。

建议形成两条稳定产品通路：

### 10.1 自有模型运行时

- 一个稳定、公开、长期兼容的应用级 API。
- 支持 ONNX/TFLite/MNN 等常见模型来源。
- 自动分图、NPU 优先、CPU/GPU 可解释回退。
- 公开算子、量化、动态 Shape、内存限制和机型兼容矩阵。
- 提供离线编译、在线编译缓存和模型加密能力。
- 提供可观测的后端命中率、子图边界、数据拷贝、编译和执行耗时。

### 10.2 系统基础模型服务

- 类似 Apple Foundation Models 或 Google AICore 的系统模型 API。
- 系统负责模型下载、升级、内存复用、安全和硬件调度。
- 应用通过 Prompt、结构化输出或 Tool Calling 使用，不接触模型权重。

## 11. 建议验证项目

公开资料只能判断“产品能力是否存在”，不能替代实机验证。建议选同一组模型，在各品牌设备上完成：

1. MobileNet/UltraFace：验证卷积模型的 NPU 命中率和小模型延迟。
2. Whisper Tiny：验证音频模型和长时间持续负载。
3. 小型 Transformer：验证 Attention、动态 Shape 和内存压力。
4. 记录冷启动编译、热启动缓存、单次延迟、功耗和 CPU/GPU/NPU 负载。
5. 制造一个不支持算子，观察分图和回退行为是否可见、是否稳定。

## 12. 参考资料

- Apple Core ML: <https://developer.apple.com/documentation/CoreML>
- Apple Core AI: <https://developer.apple.com/documentation/CoreAI>
- Apple Core AI overview: <https://developer.apple.com/core-ai/>
- Apple `MLComputeUnits.cpuAndNeuralEngine`: <https://developer.apple.com/documentation/coreml/mlcomputeunits/cpuandneuralengine>
- Google Android AI overview: <https://developer.android.com/ai/overview>
- Google Gemini Nano/AICore: <https://developer.android.com/ai/gemini-nano>
- Google LiteRT NPU Delegate: <https://ai.google.dev/edge/litert/android/npu>
- Google NNAPI migration guide: <https://developer.android.com/ndk/guides/neuralnetworks/migration-guide>
- vivo VCAP: <https://developers.vivo.com/product/ai/vcap>
- vivo VCAP 快速入门手册: <https://swsdl.vivo.com.cn/appstore/developer/uploadFile/20250926/3pbuu5/VCAP%E5%BF%AB%E9%80%9F%E5%85%A5%E9%97%A8%E6%89%8B%E5%86%8C.pdf>
- vivo VCAP 模型转换说明: <https://swsdl.vivo.com.cn/appstore/developer/uploadFile/20250926/ipe3qg/VCAP%E6%A8%A1%E5%9E%8B%E5%B7%A5%E5%85%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C.pdf>
- OPPO 开放平台: <https://open.oppomobile.com/new/wiki>
- Xiaomi MACE: <https://github.com/XiaoMi/mace>
- HarmonyOS Neural Network Runtime Kit: <https://developer.huawei.com/consumer/en/doc/harmonyos-guides/neural-network-runtime-kit-introduction>
- HarmonyOS NNRt 开发指导: <https://developer.huawei.com/consumer/cn/doc/harmonyos-guides-V13/neural-network-runtime-guidelines-V13>
- HarmonyOS MindSpore Lite 推理: <https://developer.huawei.com/consumer/cn/doc/harmonyos-guides-V13/mindspore-lite-guidelines-V13>
- HiAI Foundation Kit: <https://developer.huawei.com/consumer/cn/doc/harmonyos-guides-V5/hiai-foundation-kit-guide-V5>
- 华为 HiAI 开放平台: <https://developer.huawei.com/consumer/cn/hiai/>
