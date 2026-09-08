# HarmonyOS NPU 统一接入方案

GitHub Pages 静态文档站源码。

发布方式：在仓库 `Settings -> Pages` 中选择 `Deploy from a branch`，分支选择 `main`，目录选择 `/docs`。

发布后的默认地址为：

```text
https://panqiangbiao.github.io/npu_model/
```

## HarmonyBeautyDemo

`HarmonyBeautyDemo/` 是完整的 HarmonyOS 美颜与人脸推理实验工程，支持在界面切换并比较：

- 原生 MindSpore Lite + NNRT NPU；
- MNN USER_0 + HiAI NPU；
- MNN USER_1 委托路径；
- MindSpore Lite CPU；
- MNN OpenCL GPU。

工程包含 ArkTS 应用代码、Native C++ 推理与渲染代码、MNN/HiAI 运行库、模型资源、离线测试、Trace 分析脚本和性能报告。使用 DevEco Studio 打开 `HarmonyBeautyDemo` 目录即可同步和构建，签名配置需要在本机 DevEco Studio 中生成。

详细说明见 [`HarmonyBeautyDemo/README.md`](HarmonyBeautyDemo/README.md)。

TinyGPT2 的 CPU、MNN USER_0/NPU 与 CANN 算子 Profiling 实测见
[`HarmonyBeautyDemo/docs/performance/tiny_gpt2_cpu_npu_profiling_report.md`](HarmonyBeautyDemo/docs/performance/tiny_gpt2_cpu_npu_profiling_report.md)，
原始 `.prof`、逐次耗时和算子 CSV 位于
[`HarmonyBeautyDemo/docs/data/tiny_gpt2_cpu_npu_20260908_120500/`](HarmonyBeautyDemo/docs/data/tiny_gpt2_cpu_npu_20260908_120500/)。
