# Beauty Demo Trace Markers

应用关键路径统一使用 `HBAI_TRACE/` 前缀。抓取 hitrace 后，可以只搜索该前缀，过滤系统中其他应用和服务的标记。

## 节点分类

| 分类 | Trace 名称 | 含义 |
| --- | --- | --- |
| 总链路 | `Frame/Total/NPU`、`Frame/Total/MNN-NPU`、`Frame/Total/CPU`、`Frame/Total/GPU` | 一次人脸检测和关键点处理的总耗时 |
| 输入 | `Frame/ConsumeInput` | ArkTS 获取 Native 侧准备好的分析帧 |
| 输入 | `Native/Input/CaptureFaceFrame` | 从相机渲染链路发起分析帧采集 |
| 输入 | `Native/Input/AnalysisFboDraw` | 将相机纹理绘制到分析尺寸 FBO |
| 输入 | `Native/Input/GlReadPixels` | 从 GPU 读取 RGBA 像素 |
| 输入 | `Native/Input/RgbaToNchw` | RGBA 转模型 NCHW 输入 |
| 输入 | `Native/Input/PublishFaceFrame` | 发布一帧给 ArkTS 推理线程 |
| 人脸检测 | `Face/Infer/<backend>` | 指定后端的人脸检测推理 |
| 人脸检测 | `Face/DecodeSsd` | SSD 输出解码、阈值和坐标处理 |
| 关键点 | `Landmarks/Preprocess` | 人脸区域裁剪和关键点输入预处理 |
| 关键点 | `Landmarks/Infer/<backend>` | 指定后端的关键点推理 |
| 关键点 | `Landmarks/Decode` | 关键点输出解码 |
| FaceMesh | `FaceMesh/Preprocess`、`FaceMesh/Infer/NPU`、`FaceMesh/Decode` | FaceMesh 预处理、推理和解码 |
| 输出 | `Output/SetFaceRegion`、`Native/Output/SetFaceRegion` | 人脸区域从 ArkTS 传到 Native |
| 输出 | `Output/SetLandmarks`、`Native/Output/SetLandmarks` | 关键点从 ArkTS 传到 Native |
| GPU | `GPU/CreateOpenClSession`、`GPU/Initialize` | MNN OpenCL 会话创建和后端初始化 |
| GPU | `GPU/TensorUpload`、`GPU/RunSession`、`GPU/TensorDownload` | GPU 输入上传、执行和结果下载 |
| GPU | `GPU/Face/Total`、`GPU/Landmarks/Total` | GPU 检测和关键点调用总耗时 |
| MNN-NPU | `MNNNPU/CreateSession`、`MNNNPU/Initialize` | MNN HiAI 会话创建和后端初始化 |
| MNN-NPU | `MNNNPU/TensorUpload`、`MNNNPU/RunSession`、`MNNNPU/TensorDownload` | MNN NPU 输入、执行和输出 |
| MNN-NPU | `MNNNPU/Face/Total`、`MNNNPU/Landmarks/Total` | MNN NPU 检测和关键点调用总耗时 |
| 渲染 | `Native/Render/Frame` | 一帧相机预览渲染总耗时 |
| 渲染 | `Native/Render/BeautyEffects` | 美颜和面具效果绘制 |
| 渲染 | `Native/Render/EglSwapBuffers` | EGL 提交显示帧 |

其中 `<backend>` 为 `NPU`、`MNN-NPU`、`CPU` 或 `GPU`。

## 状态计数

| Counter | 值 |
| --- | --- |
| `State/Backend` | `1=NPU`，`2=CPU`，`3=GPU`，`4=MNN-NPU` |
| `State/InferenceBusy` | `1` 表示推理处理中，`0` 表示结束 |
| `State/FaceInputReady` | Native 侧是否已有待消费的分析帧 |
| `State/FaceConfidencePermille` | 人脸置信度乘以 1000 |
| `State/LandmarkCount` | 当前输出的关键点数量 |

ArkTS 异步区间在原始 trace 中表现为 `S/F`，Native 同步区间表现为 `B/E`，状态计数表现为 `C`。

## 抓取命令

建议在手机侧开始和结束抓取，避免持续输出被终端截断：

```powershell
hdc -t <device-id> shell "hitrace --trace_begin -b 32768 app sched freq"
# 在手机上执行待分析操作
hdc -t <device-id> shell "hitrace --trace_finish -o /data/local/tmp/hbai_trace.txt"
hdc -t <device-id> file recv /data/local/tmp/hbai_trace.txt .\hbai_trace.txt
```

`app` 用于采集应用 trace；`sched` 和 `freq` 用于关联线程调度及 CPU 频点。若只验证标记，可只抓取 `app`。

## 搜索示例

```powershell
rg "HBAI_TRACE/" .\hbai_trace.txt
rg "HBAI_TRACE/(Frame|Face|Landmarks|GPU)" .\hbai_trace.txt
rg "HBAI_TRACE/State/" .\hbai_trace.txt
```

分析单次调用时，先定位一个 `Frame/Total/<backend>` 区间，再按时间戳观察该区间内的输入、推理、解码和输出节点。GPU 路径还可以比较 `TensorUpload`、`RunSession`、`TensorDownload`，判断瓶颈来自数据搬运还是计算。
