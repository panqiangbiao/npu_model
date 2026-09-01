# HarmonyOS NPU 统一接入方案

面向 MNN、ncnn、LiteRT、ONNX Runtime 等第三方推理框架接入麒麟 NPU 的架构设计。

本网站包含以下材料：

- [汇报版](npu_abstraction_brief.md)：适合方案评审和管理汇报。
- [完整设计](npu_abstraction_design.md)：包含现状、目标架构、接口、分图、内存、缓存、调度、可观测性和落地计划。
- [后摄五后端性能对比](rear_camera_five_backend_comparison.md)：基于同一固定人脸输入，对比原生 NPU、MNN-U0、MNN-U1、CPU 和 GPU，并进一步分析 NPU 与 MNN-U0 的反向顺序复测、阶段耗时和频点差异。
- [MNN USER_0开发问题总结](mnn_user0_development_issue_summary.md)：总结从错误选择USER_1、BuildIRModel失败、图语义错误，到USER_0正确运行和性能复测的完整问题链路。

## 核心结论

建议建设：

```text
多框架薄插件
  + Huawei Framework Adapter API
  + NNRt 图分析、编译、缓存和执行服务
  + CANN Kit / 麒麟 NPU
```

这样可以把分散在各推理框架中的 NPU 算子适配、分图、缓存和性能分析能力统一收敛到华为 Runtime，避免逐算子提交与重复适配。
