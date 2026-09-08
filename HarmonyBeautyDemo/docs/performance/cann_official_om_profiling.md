# CANN Kit官方OM Profiling验证

## 结论

基于华为官方`CANNKit-SampleCode-Clientdemo-cpp`中的SqueezeNet离线模型和官方API顺序，当前手机已成功完成模型加载、20次NPU推理并生成算子级Profiling产物。

这次验证说明：

- 不需要构造一种特殊命名的“Profiling OM”；Profiling由模型加载阶段的`HMS_HiAIOptions_SetOmOptions`开启。
- 不能用任意系统OM验证。模型首先要通过当前设备的兼容性检查，并且必须选择名为`HIAI_F`的设备。
- CANN Kit运行时会直接生成模型级和算子级CSV，无需先用SmartPerf解析`.prof`才能查看基础耗时。

## 测试条件

| 项目 | 内容 |
|---|---|
| 手机 | MLN-AL00，HarmonyOS 7.0.0.37 |
| HiAI版本 | `110.636.120.010` |
| HDC | `3.2.0b` |
| 模型 | 官方Sample自带SqueezeNet `hiai.om` |
| 模型大小 | 2,496,459 bytes |
| SHA-256 | `533B84458D7694174A220D3AA8B984B1F0021FB7D3997A1CC2732AA3B51C7AD3` |
| 推理设备 | `HIAI_F` |
| 推理次数 | 20 |
| 输入 | 全零Tensor，仅用于执行链路与Profiling验证 |

正式采集前，应用停止相机采集和人脸检测，避免美颜模型与分类模型并发执行。

## 官方调用顺序

```text
读取官方OM资源
  -> OH_NNCompilation_ConstructWithOfflineModelBuffer
  -> HMS_HiAICompatibility_CheckFromBuffer
  -> HMS_HiAIOptions_SetOmOptions(PROFILING, outputDir)
  -> OH_NNCompilation_SetDevice(HIAI_F)
  -> HMS_HiAIOptions_SetBandMode
  -> HMS_HiAIOptions_SetModelDeviceOrder(NPU)
  -> OH_NNCompilation_Build
  -> OH_NNExecutor_Construct
  -> 创建输入/输出Tensor
  -> OH_NNExecutor_RunSync x 20
  -> 销毁Tensor、Executor和Compilation
```

关键返回值均为成功：

```text
compatibility=0
SetOmOptions code=0
SetDevice code=0
SetBandMode code=0
SetModelDeviceOrder code=0
CompilationBuild code=0
```

设备枚举发现两个accelerator。之前按设备类型选择第一个设备会选到`NPU_ohos.boot.hardware.Kirin9030S_v2_0`并导致Build失败；改为按官方要求选择`HIAI_F`后成功。

## Profiling结果

模型级结果来自`*_model.csv`：

| 指标 | 结果 |
|---|---:|
| 模型加载 | 12,453 us |
| 推理次数 | 20 |
| 平均推理 | 1,660.8 us |
| 最小推理 | 1,069 us |
| P50 | 1,649 us |
| P95 | 1,953 us |
| 最大推理 | 2,337 us |

算子级结果来自`*_op.csv`：

| 指标 | 结果 |
|---|---:|
| 总记录数 | 660 |
| 每次推理算子数 | 33 |
| 设备类型 | 660条均为`npu_aicore` |

按20次累计耗时排序的主要算子：

| 算子 | 平均耗时(us) | 最大耗时(us) |
|---|---:|---:|
| `transdata_for_nd_18(TransData)` | 90.8 | 93 |
| `pool1(PoolingD)` | 82.6 | 102 |
| `conv10(Convolution)` | 81.0 | 85 |
| `conv1(Convolution)` | 69.2 | 82 |
| `pool3(PoolingD)` | 45.2 | 55 |
| `fire9/expand3x3(Convolution)&&fire9/concat(ConcatD)` | 38.0 | 42 |

## 产物

正式采集目录：

`artifacts/cann_profile/20260905_105408`

包含：

- `default_ndk_20260905_105408.prof`：设备侧原始Profiling数据。
- `default_ndk_20260905_105408_client.prof`：客户端原始Profiling数据。
- `default_ndk_20260905_105408_model.csv`：模型加载和每次推理的端到端耗时。
- `default_ndk_20260905_105408_op.csv`：算子名称、开始时间、耗时和执行设备。

## 测量边界

本次输入为全零Tensor，目标是验证离线OM、NNRT/CANN Kit执行和算子Profiling通路，不用于评价SqueezeNet分类准确率。当前CSV没有直接提供NPU频率、DDR带宽、算子输入输出Shape、Cache命中或DMA拷贝字节，这些信息仍需结合trace或更深层驱动维测。

## 参考

- [华为CANN Kit集成模型](https://developer.huawei.com/consumer/cn/doc/doccenter-capabilities/cannkit-integration-model)
- [官方CANN Kit C++ Sample](https://gitee.com/harmonyos_samples/cannkit-samplecode-clientdemo-cpp)
