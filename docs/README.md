# HarmonyOS NPU 统一接入方案

面向 MNN、ncnn、LiteRT、ONNX Runtime 等第三方推理框架接入麒麟 NPU 的架构设计。

本网站包含两份材料：

- [汇报短版](npu_abstraction_brief.md)：适合方案评审和管理汇报。
- [完整设计](npu_abstraction_design.md)：包含现状、目标架构、接口、分图、内存、缓存、调度、可观测性和落地计划。

## 核心结论

建议建设：

```text
多框架薄插件
  + Huawei Framework Adapter API
  + NNRt 图分析、编译、缓存和执行服务
  + CANN Kit / 麒麟 NPU
```

这样可以把分散在各推理框架中的 NPU 算子适配、分图、缓存和性能分析能力统一收敛到华为 Runtime，避免逐算子提交与重复适配。

