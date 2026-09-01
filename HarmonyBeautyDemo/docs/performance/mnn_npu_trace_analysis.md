# MNN-NPU 人脸推理 Trace 分析

> 与原生 NNRT 的差异定位见 [nnrt_vs_mnn_npu_report.md](nnrt_vs_mnn_npu_report.md)。
>
> 注意：本文记录的是第一轮 MNN 数据。后续已重新采集有效正脸完整链路，最新数据和 NNRT 对比
> 以 [nnrt_vs_mnn_npu_report.md](nnrt_vs_mnn_npu_report.md) 为准。

## 1. 采集说明

- 后端：MNN 3.6 mobiInfer HiAI，`MNN_FORWARD_USER_1`
- 场景：前摄、亮白、实时美颜开启，与既有 NPU/CPU/GPU 测试口径一致
- 预热：约 10 秒；采集：20 秒
- 应用 PID：`40487`
- Trace 环形缓冲区有效窗口：`9.616 s`
- 有效推理：35 次；日志完整样本：7 次

原始数据位于本地忽略提交的 `artifacts/backend_compare/mnn_npu.*`。

## 2. 结论

MNN-NPU 已真实进入 HiAI 后端，没有回退 CPU。稳定态完整链路平均 `62.9 ms`，P95 `69.8 ms`；
性能好于 CPU 和 GPU，但暂时慢于直接 NNRT/NPU 通路。MNN-NPU 的应用 CPU 占用最低，但 DDR 与
系统总忙度高于直接 NPU，仍需在相同画面和温度下复测后才能归因为框架差异。

当前瓶颈不是 MNN Tensor 拷贝，而是两次串行 NPU 执行和两次推理之间的 CPU 关键点 ROI 预处理。

## 3. 稳定态耗时

| 阶段 | 平均 | P95 | 最大 |
|---|---:|---:|---:|
| 完整链路（hilog） | 62.86 ms | 69.80 ms | 71.00 ms |
| 人脸推理 | 18.09 ms | 23.44 ms | 26.23 ms |
| 人脸 Tensor 上传 | 0.11 ms | 0.15 ms | 0.25 ms |
| 人脸 `RunSession` | 17.39 ms | 22.85 ms | 23.76 ms |
| 人脸输出回读 | 0.57 ms | 1.18 ms | 3.31 ms |
| 关键点 ROI 预处理 | 20.13 ms | 26.08 ms | 27.50 ms |
| 关键点推理 | 18.16 ms | 22.57 ms | 24.04 ms |
| 关键点 Tensor 上传 | 0.05 ms | 0.14 ms | 0.18 ms |
| 关键点 `RunSession` | 17.99 ms | 22.46 ms | 23.91 ms |
| 关键点输出回读 | 0.06 ms | 0.07 ms | 0.09 ms |

`Frame/Total/MNN-NPU` 的 53 次事件中包含 18 次没有新帧时的快速返回，因此其 `39.99 ms` 原始均值
不能作为成功推理耗时；完整链路应使用 35 次真实推理区间或 hilog 样本。

## 4. 四后端位置

| 后端 | 完整链路平均 | P95 | 应用 CPU | 系统 capacity busy | DDR cluster1 | GPU load |
|---|---:|---:|---:|---:|---:|---:|
| NNRT/NPU | **42.38 ms** | **50.00 ms** | 68.93% | **23.16%** | **0.632 GHz** | **27.0%** |
| MNN-NPU | 62.86 ms | 69.80 ms | **55.96%** | 35.74% | 0.936 GHz | 32.2% |
| CPU | 65.20 ms | 83.60 ms | 91.50% | 32.60% | 0.875 GHz | 28.5% |
| GPU | 79.83 ms | 107.45 ms | 77.76% | 36.97% | 0.915 GHz | 38.7% |

MNN-NPU 相对直接 NPU 平均慢 `48.3%`，但比 CPU 快 `3.6%`、比 GPU 快 `21.3%`。它将应用 CPU
占用降到四组最低，但本轮系统 busy 和 DDR cluster1 较高。由于四组是连续实拍而非固定输入，系统级
指标只能用于发现方向，不能直接视为纯框架开销。

## 5. 优化方向

1. 将人脸检测和关键点改为单模型或共享骨干，减少一次 NPU graph 调度。
2. 优化 `Landmarks/Preprocess`：直接从原始帧生成关键点输入，避免当前 CPU ROI Float32 中间处理。
3. 把两次推理改为异步流水，使第 N 帧关键点与第 N+1 帧人脸检测重叠；需要先验证 HiAI 多 Session 并发策略。
4. 固定输入、锁定画面与温度，重新采集 NNRT/NPU 和 MNN-NPU，单独比较框架调度差异。
5. 下次增加 IRQ/NPU 驱动类别；本次 `app sched freq` 未包含 NPU completion IRQ，不能由该计数判断 NPU 利用率。

## 6. 复现

```powershell
python tools/analyze_backend_trace.py `
  --trace artifacts/backend_compare/mnn_npu.trace `
  --hilog artifacts/backend_compare/mnn_npu_hilog.txt `
  --pid 40487 `
  --output artifacts/backend_compare/mnn_npu_summary.json
```
