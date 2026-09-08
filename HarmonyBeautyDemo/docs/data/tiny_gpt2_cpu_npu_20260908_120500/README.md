# TinyGPT2 CPU/NPU Profiling 原始数据

采集时间：2026-09-08 12:05:05（手机时间）。

| 文件 | 内容 |
|---|---|
| `benchmark_summary.csv` | CPU、MNN USER_0、NNRT Profiling 汇总 |
| `mnn_benchmark.csv` | CPU 与 MNN USER_0 各 50 次逐次耗时及 setup 耗时 |
| `cann_model.csv` | NNRT/CANN 模型加载、50 次推理和卸载时间 |
| `cann_op.csv` | 50 次推理的算子级时间、算子名和执行设备 |
| `cann_client.prof` | CANN Profiling 客户端原始记录 |
| `cann_device.prof` | CANN Profiling 设备侧原始记录 |
| `run_hilog.txt` | 本次运行的 `PQB:TINY_GPT2` 与 `PQB:CANN_PROFILE` 日志 |

测试输入为同一份 `float32[1,16,2]` embeddings。CPU 和 MNN USER_0 均预热 5 次，再测量 50 次；NNRT 使用 USER_0 导出的 OM 和相同输入运行 50 次。
