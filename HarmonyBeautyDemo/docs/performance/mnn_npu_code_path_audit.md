# MNN-NPU 与 NNRT 代码路径审查

## 1. 结论

当前 Demo 的 `MNN-NPU` 存在后端选择错误。代码使用 `MNN_FORWARD_USER_1`，但 mobiInfer 将
`USER_1` 注册为逐算子 `HiAIDelegateBackend`：只把部分卷积交给 HiAI，其余算子直接在 CPUBackend
执行。真正的整图 HiAI 后端注册在 `MNN_FORWARD_USER_0`。

因此当前性能数据不是“整图 MNN-NPU 对 NNRT”，而是“CPU 与多个离散 HiAI Conv 混合执行对
NNRT”。在修正为 `USER_0` 并重新采集前，不应继续分析两种框架本身的性能差异。

## 2. 代码问题

### P0：选择了错误的 MNN 后端

Demo 在 `gpu_face_inference.cpp` 中把 NPU 常量和 `ScheduleConfig.type` 都设置为
`MNN_FORWARD_USER_1`。

mobiInfer 的注册关系是：

| MNN 类型 | 实际实现 | 行为 |
|---|---|---|
| `MNN_FORWARD_USER_0` | `NPUBackend` | 将整张 MNN 图映射为 GE Graph，执行一次 `BuildIRModel` 和一次模型 Load |
| `MNN_FORWARD_USER_1` | `HiAIDelegateBackend` | 继承 `CPUBackend`，只对 Convolution 创建独立 `HiAIConvExecution` |

`HiAIDelegateBackend::onCreate()` 对非卷积算子明确调用 `CPUBackend::onCreate()`。当前路径必然是
CPU/NPU 混合执行，不是整图 NPU。

### P0：逐卷积委托存在全局 70 个上限

`HiAIDelegateBackend` 使用静态全局 `sHiAIConvCount`，只允许前 70 个卷积创建 HiAI Execution；超过
上限后卷积也回到 CPU。该计数器：

- 同时跨越人脸模型和关键点模型；
- Session 释放时不递减；
- 后端重新初始化时不清零；
- 达到上限后仍继续 `fetch_add()`。

所以实际下沉比例依赖模型加载顺序和进程内历史。重复初始化后，新的 Session 可能有更多卷积落在
CPU，性能不可稳定复现。

### P1：现有后端校验会产生误导

Demo 用 `getSessionInfo(..., BACKENDS, ...)` 检查第一项是否为 `USER_1`。MNN 的 `BACKENDS` 返回
Pipeline 的 main forward type，不会列出 `HiAIDelegateBackend` 内部每个算子的实际设备。

即使校验得到 `USER_1`，也不能证明所有算子在 NPU。`backupType = schedule.type` 同样无效，因为 CPU
执行不是 MNN 外部 fallback，而是 `USER_1` 后端自身主动调用 `CPUBackend::onCreate()`。

### P1：两条路径的精度配置不对等

MNN Session 固定使用 `BackendConfig::Precision_High`。NNRT 使用 `PERFORMANCE_HIGH` 和
`PRIORITY_HIGH`，这是设备性能和调度模式，不等价于 MNN 的高精度模式。

修正为 `USER_0` 后，应分别测试 `Precision_Normal`、`Precision_High`、`Precision_Low`，并同时做输出
精度校验，不能直接沿用当前配置得出框架性能结论。

### P1：MNN 人脸自检弱于 NNRT

NNRT 自检会比较置信度和检测框 IoU，要求 `score >= 0.99` 且 `IoU >= 0.95`。MNN 自检只检查
置信度和框尺寸为正，没有与 NNRT/CPU 的期望 anchor、box 或完整输出做对齐。

这解释了为什么固定自检通过后，实景姿态变化仍曾出现明显置信度差异。当前自检不足以证明两个转换
模型功能等价。

### P2：应用层 Tensor 拷贝不是主要性能问题

MNN 路径会创建 Host Tensor，再执行 `copyFromHostTensor()`；输出也通过 Host Tensor 回读。该实现有
优化空间，但有效 Trace 中两次推理的上传和下载总计约 `0.66 ms`，无法解释约 `22 ms` 的链路差异。

当前主要开销与 `USER_1` 的混合执行、逐卷积模型调度和 CPU/NPU 边界更吻合。

### P2：每帧没有重复构图

`BuildIRModel` 和模型 Load 位于 Session resize/初始化阶段。每帧 `runSession()` 不会重新生成 OM。
因此在线编译会影响首次初始化，不是稳定态逐帧差异的原因。

## 3. 两条实际调用栈

### 当前 MNN-NPU（实际）

```text
ArkTS runMnnNpuFace()
  -> MNN::Interpreter::runSession()
  -> MNN_FORWARD_USER_1 / HiAIDelegateBackend
     -> Convolution: 独立 HiAIConvExecution / AiModelMngerClient::Process
     -> 其他算子: CPUBackend
     -> 超过全局 70 个的 Convolution: CPUBackend
  -> Host Tensor 输出回读
```

### NNRT

```text
ArkTS model.predict()
  -> MindSpore Lite .ms graph
  -> HarmonyOS NNRT accelerator device
  -> 厂商整图/子图编译与 NPU 执行
  -> MSTensor 输出
```

## 4. 正确的下一步

1. 将 Demo 的 MNN NPU 类型改为 `MNN_FORWARD_USER_0`，后端校验期望值同步改为 `USER_0`。
2. 保持 `backupType = USER_0`，整图出现不支持算子时直接失败，不允许静默 CPU fallback。
3. 加强固定输入自检：人脸 scores/boxes 逐元素误差、Top anchor、检测框 IoU；关键点保留 136 元素误差。
4. 对三种 MNN Precision 做固定输入正确性和 200 次稳定态基准。
5. 只有 `USER_0` 路径跑通且输出对齐后，再重新采集 NNRT 与 MNN-NPU Trace。

## 5. 现有数据应如何解释

当前 `mnn_face_valid.trace` 和 `55.50 ms` 结果仍然是真实测量，但应改名理解为
`MNN-HiAI-Delegate`。它证明逐卷积委托方案不适合作为本 Demo 的默认 NPU 后端，不能证明整图 MNN
后端比 NNRT 慢 `67.3%`。

