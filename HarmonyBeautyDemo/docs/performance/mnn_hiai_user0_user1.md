# MNN HiAI USER_0 与 USER_1 的区别及当前问题

## 1. 背景

`USER_0` 和 `USER_1` 是 mobiInfer 在 MNN 中注册的两个自定义 ForwardType。它们不是 MNN
官方定义的通用 NPU 模式，而是该分支为华为 HiAI 后端分配的入口编号。

两条路径都从 MNN API 进入，但图划分、算子执行位置和性能特征完全不同。

## 2. 核心区别

| 对比项 | USER_0 | USER_1 |
|---|---|---|
| 后端定位 | HiAI 整图后端 | 逐卷积 HiAI 委托后端 |
| 图划分 | 尽量将完整 MNN 图转换为一张或少量 HiAI 图 | 每个支持的卷积分别交给 HiAI |
| 非卷积算子 | 需要存在正确的 HiAI 算子映射 | 主要由 MNN CPU 后端执行 |
| CPU/NPU 边界 | 理论上较少 | 较多 |
| NPU 图提交次数 | 少 | 多，可能每个卷积一次 |
| 整图算子融合 | 有机会 | 基本无法跨委托边界融合 |
| 兼容性 | 依赖整套算子映射，当前较低 | 较高 |
| 性能潜力 | 正确适配后较高 | 容易受小图提交和同步开销影响 |
| 当前正确性 | 未通过 | 已通过 |
| 当前用途 | 目标整图实现 | 兼容性与性能基线 |

## 3. USER_0 执行路径

```text
.mnn 模型
  -> MNN Interpreter
  -> MNN Op 转换为 HiAI/GE Operator
  -> 构建完整 HiAI IR 图
  -> BuildIRModel 在线编译
  -> 加载 NPU 模型
  -> 整图执行
  -> scores / boxes / landmarks
```

USER_0 需要正确处理模型中的全部关键算子，包括卷积、激活、二元运算、Reshape、Flatten、
Permute、ConvertTensor 和 Softmax 等。任一算子的 shape、axis、format 或数据布局转换错误，都可能
导致整图最终输出错误。

理论优势是中间 Tensor 可以留在 NPU 图内，减少 CPU/NPU 往返，并允许编译器进行常量折叠、
格式优化和算子融合。

## 4. USER_1 执行路径

```text
.mnn 模型
  -> MNN Interpreter
  -> MNN CPU 后端执行通用算子
  -> 遇到支持的 Convolution
  -> 单独构建并调用 HiAI Conv
  -> 结果返回 MNN
  -> CPU 继续执行后续算子
  -> 下一个 Conv 再次调用 HiAI
```

USER_1 主要复用 MNN CPU 算子的正确实现，只将部分卷积下沉到 NPU。因此它更容易获得正确结果，
但会增加小图构建或调度、数据边界、同步等待和 NPU 提交次数。

固定静态图测试中，USER_1 相比原生 NPU：

| 指标 | 原生 NPU | USER_1 |
|---|---:|---:|
| 完整有效帧平均 | 51.57 ms | 102.66 ms |
| 人脸推理平均 | 6.44 ms | 40.59 ms |
| 关键点推理平均 | 3.50 ms | 24.23 ms |
| NPU 完成中断/秒 | 70.18 | 261.12 |

该结果只能说明 USER_1 逐卷积委托路径的性能，不能作为 USER_0 整图 MNN-NPU 与原生 NPU 的
最终对比。

## 5. USER_0 当前状态

USER_0 已完成以下基础能力：

- 后端能够注册并被 MNN Session 选择；
- HiAI IR 图能够在线构建；
- 模型能够加载并进入 NPU 执行；
- 应用能够获得最终输出 Tensor。

当前失败发生在结果正确性校验。离线人脸 fixture 的预期结果为：

```text
bestIndex = 4271
IoU >= 0.95
confidence 接近 1.0
```

当前 USER_0 的一轮结果为：

```text
bestIndex = 4382
IoU = 0.328
confidence = 0.097
```

因此当前问题不是“没有使用 NPU”，而是“整图在 NPU 上执行后数值不正确”。在正确性通过前，
不能采集 USER_0 与原生 NPU 的性能对比数据。

## 6. 已发现的风险点

### 6.1 低维 Tensor 的物理 shape

MNN 中的分类 Tensor 逻辑 shape 为：

```text
[4420, 2]
```

HiAI 后端可能将低维 NCHW Tensor 补齐为：

```text
[1, 1, 4420, 2]
```

如果仍直接使用 MNN 的 Softmax axis，逻辑轴和 HiAI 物理轴可能不一致。

### 6.2 NC4HW4 与 NCHW 转换

当前 `NPUConvertTensor` 主要根据 shape 选择 `Permute`。但 NC4HW4 与 NCHW 不一定只是维度顺序
不同，还可能存在通道按 4 打包的内存布局差异。恒等 Permute 或普通 Reshape 无法完成实际解包。

这是当前重点怀疑对象，但尚未通过中间 Tensor 对比证明它是第一个错误节点。

### 6.3 问题不局限于 Softmax

USER_0 与正确的 MNN OpenCL 输出曾得到以下最终 Tensor 差异：

```text
score MAE = 0.4020
box MAE   = 1.3486
```

`boxes` 不经过最终分类 Softmax，但同样明显错误。因此不能只在应用侧调整概率、类别轴或 anchor
索引。更早的 Conv 输出布局、Bias、Flatten、Reshape、Permute 或 ConvertTensor 也可能存在问题。

### 6.4 部分替代图无法在线编译

曾尝试将 Softmax 展开为：

```text
ReduceMax -> Sub -> Exp -> ReduceSum -> RealDiv
```

代码可以编译，但当前手机 HiAI DDK 在 BuildIRModel 阶段拒绝该图。直接对补齐后的物理轴使用
Softmax 也出现过在线构图失败。这说明修复方案还必须满足设备侧 DDK 的实际算子和广播约束。

## 7. 为什么暂时不能用 USER_1 代替 USER_0

USER_1 可以正确运行，但它改变了目标问题：

```text
USER_0：验证 MNN 整图接入华为 NPU的性能潜力
USER_1：验证 CPU 主导、卷积逐个下沉 NPU 的兼容性
```

如果用 USER_1 的数据代表 USER_0，会把逐卷积提交、同步和 CPU 执行开销错误归因于 MNN 整图
后端。因此后续报告必须分别标记两条路径。

## 8. 后续修复方法

USER_0 是整图提交，当前只能直接看到最终输出。下一步需要为模型增加中间输出，使用相同 fixture
分别运行 MNN CPU/OpenCL 与 USER_0：

```text
选择若干主干中间节点
  -> 比较 CPU/OpenCL 与 USER_0 输出
  -> 二分缩小错误区间
  -> 找到第一个数值偏离节点
  -> 判断是算子语义、axis、shape 还是 format 问题
  -> 修复对应 HiAI Execution
  -> 重新执行最终 fixture
```

USER_0 只有满足以下条件后才能重新开展性能测试：

```text
bestIndex = 4271
IoU >= 0.95
landmark count = 136
landmark maxError <= 0.04
```

## 9. 当前结论

1. USER_0 是目标整图 HiAI 后端，理论性能潜力更高，但当前输出不正确。
2. USER_1 是 CPU + 逐卷积 HiAI 委托，当前结果正确，但性能不能代表 USER_0。
3. USER_0 的问题同时影响 scores 和 boxes，不能通过最终 Softmax 后处理解决。
4. 当前应暂停 USER_0 性能采集，优先通过中间 Tensor 对比定位首个错误算子。
