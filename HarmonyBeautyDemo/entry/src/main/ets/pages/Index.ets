import { abilityAccessCtrl, common, Permissions } from '@kit.AbilityKit';
import { camera } from '@kit.CameraKit';
import { mindSporeLite } from '@kit.MindSporeLiteKit';
import { hilog, hiTraceMeter } from '@kit.PerformanceAnalysisKit';
import { BeautyCameraController, CameraStartResult } from '../services/BeautyCameraController';
import beautyPipeline, { FaceFrame, GpuFaceResult, GpuLandmarkResult } from 'libbeauty_pipeline.so';

const DOMAIN = 0xB001;
const TRACE_PREFIX = 'HBAI_TRACE/';

interface BeautyPreset {
  name: string;
  smooth: number;
  whiten: number;
  rosy: number;
}

interface FaceDetectionResult {
  confidence: number;
  bestIndex: number;
  rawLocation: Array<number>;
  left: number;
  top: number;
  width: number;
  height: number;
}

class LandmarkInput {
  data: Float32Array;
  cropX: number;
  cropY: number;
  padX: number;
  padY: number;
  scale: number;

  constructor(data: Float32Array, cropX: number, cropY: number, padX: number, padY: number, scale: number) {
    this.data = data;
    this.cropX = cropX;
    this.cropY = cropY;
    this.padX = padX;
    this.padY = padY;
    this.scale = scale;
  }
}

enum InferenceBackend {
  NPU = 'NPU',
  MNN_USER_0 = 'MNN-U0',
  MNN_USER_1 = 'MNN-U1',
  CPU = 'CPU',
  GPU = 'GPU'
}

@Entry
@Component
struct Index {
  private xComponentController: XComponentController = new XComponentController();
  private cameraController: BeautyCameraController = new BeautyCameraController();
  private npuFaceModel: mindSporeLite.Model | null = null;
  private npuLandmarkModel: mindSporeLite.Model | null = null;
  private npuFaceMeshModel: mindSporeLite.Model | null = null;
  private cpuFaceModel: mindSporeLite.Model | null = null;
  private cpuLandmarkModel: mindSporeLite.Model | null = null;
  private gpuReady: boolean = false;
  private mnnNpuReadyBackend: InferenceBackend | null = null;
  private traceSequence: number = 0;
  private faceDetectionTimer: number = -1;
  private staticImageTestTimer: number = -1;
  private staticImageTestFixture: ArrayBuffer | null = null;
  private staticLandmarkTestFixture: ArrayBuffer | null = null;
  private staticLandmarkExpected: Float32Array | null = null;
  private staticImageTestUnstableCount: number = 0;
  private staticImageBaselineIndex: number = -1;
  private staticImageBaselineScore: number = 0;
  private staticUser1Scores: Float32Array | null = null;
  private staticUser1Boxes: Float32Array | null = null;
  private staticImageInferenceBusy: boolean = false;
  private staticElapsedTotalMs: number = 0;
  private staticElapsedMinMs: number = Number.POSITIVE_INFINITY;
  private staticElapsedMaxMs: number = 0;
  private staticLandmarkElapsedTotalMs: number = 0;
  private staticTotalWallMs: number = 0;
  private inferenceBusy: boolean = false;
  private inferenceCount: number = 0;
  private faceTrackingActive: boolean = false;
  private consecutiveFaceMisses: number = 0;
  private consecutiveFaceMeshMisses: number = 0;
  private faceMeshScore: number = 0;
  private facePriors: Array<number> = this.createFacePriors();
  private surfaceId: string = '';
  private cameraSurfaceId: string = '';
  private presets: Array<BeautyPreset> = [
    { name: '自然', smooth: 22, whiten: 25, rosy: 8 },
    { name: '通透', smooth: 28, whiten: 45, rosy: 10 },
    { name: '亮白', smooth: 34, whiten: 65, rosy: 12 }
  ];

  @State private smooth: number = 34;
  @State private whiten: number = 65;
  @State private rosy: number = 12;
  @State private slimFace: number = 30;
  @State private bigEye: number = 20;
  @State private selectedPreset: number = 2;
  @State private spiderMaskEnabled: boolean = false;
  @State private isFront: boolean = true;
  @State private isStarting: boolean = false;
  @State private isReady: boolean = false;
  @State private statusText: string = '正在准备相机';
  @State private systemBeautyAvailable: boolean = false;
  @State private systemBeautyEnabled: boolean = false;
  @State private realtimeEffectsEnabled: boolean = true;
  @State private debugOverlayEnabled: boolean = false;
  @State private detectedFaceCount: number = 0;
  @State private detectedLandmarkCount: number = 0;
  @State private inferenceBackend: InferenceBackend = InferenceBackend.NPU;
  @State private inferenceStatus: string = '正在检查 NPU';
  @State private staticImageTestRunning: boolean = false;
  @State private staticImageTestCount: number = 0;
  aboutToDisappear(): void {
    this.stopFaceDetection();
    this.stopStaticImageTest();
    this.cameraController.release();
    beautyPipeline.release();
  }

  private createFacePriors(): Array<number> {
    const featureWidths: Array<number> = [40, 20, 10, 5];
    const featureHeights: Array<number> = [30, 15, 8, 4];
    const minBoxes: Array<Array<number>> = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]];
    const priors: Array<number> = [];
    for (let level = 0; level < featureWidths.length; level++) {
      for (let y = 0; y < featureHeights[level]; y++) {
        for (let x = 0; x < featureWidths[level]; x++) {
          minBoxes[level].forEach((boxSize: number) => {
            priors.push((x + 0.5) / featureWidths[level]);
            priors.push((y + 0.5) / featureHeights[level]);
            priors.push(boxSize / 320);
            priors.push(boxSize / 240);
          });
        }
      }
    }
    return priors;
  }

  private startFaceDetection(): void {
    this.stopFaceDetection();
    this.faceTrackingActive = false;
    this.consecutiveFaceMisses = 0;
    this.consecutiveFaceMeshMisses = 0;
    beautyPipeline.setFaceRegion(false, 0, 0, 1, 1);
    this.faceDetectionTimer = setInterval(() => this.runFaceDetection(), 180);
  }

  private stopFaceDetection(): void {
    if (this.faceDetectionTimer >= 0) {
      clearInterval(this.faceDetectionTimer);
      this.faceDetectionTimer = -1;
    }
    this.inferenceBusy = false;
  }

  private scorePairSumRange(scores: Float32Array): Array<number> {
    let minValue = Number.POSITIVE_INFINITY;
    let maxValue = Number.NEGATIVE_INFINITY;
    for (let index = 0; index < 4420; index++) {
      const pairSum = scores[index * 2] + scores[index * 2 + 1];
      minValue = Math.min(minValue, pairSum);
      maxValue = Math.max(maxValue, pairSum);
    }
    return [minValue, maxValue];
  }

  private scoresFromNchw(scores: Float32Array): Float32Array {
    const result = new Float32Array(scores.length);
    for (let index = 0; index < 4420; index++) {
      result[index * 2] = scores[index];
      result[index * 2 + 1] = scores[4420 + index];
    }
    return result;
  }

  private maxAbsDifference(left: Float32Array, right: Float32Array): number {
    if (left.length !== right.length) {
      return Number.POSITIVE_INFINITY;
    }
    let maxDifference = 0;
    for (let index = 0; index < left.length; index++) {
      maxDifference = Math.max(maxDifference, Math.abs(left[index] - right[index]));
    }
    return maxDifference;
  }

  private async toggleStaticImageTest(enabled: boolean): Promise<void> {
    if (!enabled) {
      this.stopStaticImageTest();
      this.inferenceStatus = '静态图测试已停止';
      return;
    }
    try {
      const rawFixture = await this.getAbilityContext().resourceManager
        .getRawFileContent('front_camera_face_input.bin');
      this.staticImageTestFixture = rawFixture.buffer.slice(rawFixture.byteOffset,
        rawFixture.byteOffset + rawFixture.byteLength);
      const rawLandmarkFixture = await this.getAbilityContext().resourceManager
        .getRawFileContent('front_camera_landmark_input.bin');
      this.staticLandmarkTestFixture = rawLandmarkFixture.buffer.slice(rawLandmarkFixture.byteOffset,
        rawLandmarkFixture.byteOffset + rawLandmarkFixture.byteLength);
      const rawLandmarkExpected = await this.getAbilityContext().resourceManager
        .getRawFileContent('front_camera_landmark_expected.bin');
      this.staticLandmarkExpected = new Float32Array(rawLandmarkExpected.buffer.slice(
        rawLandmarkExpected.byteOffset, rawLandmarkExpected.byteOffset + rawLandmarkExpected.byteLength));
      this.stopFaceDetection();
      this.stopStaticImageTest();
      this.staticImageTestCount = 0;
      this.staticImageTestUnstableCount = 0;
      this.staticImageBaselineIndex = -1;
      this.staticImageBaselineScore = 0;
      this.staticElapsedTotalMs = 0;
      this.staticElapsedMinMs = Number.POSITIVE_INFINITY;
      this.staticElapsedMaxMs = 0;
      this.staticLandmarkElapsedTotalMs = 0;
      this.staticTotalWallMs = 0;
      this.staticImageTestRunning = true;
      this.inferenceStatus = `${this.inferenceBackend} 静态图测试启动`;
      this.staticImageTestTimer = setInterval(() => this.runStaticImageTest(), 500);
      this.runStaticImageTest();
    } catch (error) {
      this.stopStaticImageTest();
      this.inferenceStatus = `静态图测试启动失败 · ${String(error)}`;
    }
  }

  private stopStaticImageTest(): void {
    if (this.staticImageTestTimer >= 0) {
      clearInterval(this.staticImageTestTimer);
      this.staticImageTestTimer = -1;
    }
    this.staticImageTestRunning = false;
  }

  private async runStaticImageTest(): Promise<void> {
    if (!this.staticImageTestRunning || !this.staticImageTestFixture || !this.staticLandmarkTestFixture ||
      !this.staticLandmarkExpected || this.staticImageInferenceBusy) {
      return;
    }
    const backend = this.inferenceBackend;
    const traceId = this.nextTraceId();
    const totalTrace = this.startTrace(`Static/Total/${backend}`, traceId);
    this.staticImageInferenceBusy = true;
    try {
      const startedAt = Date.now();
      const nativeMnn = backend === InferenceBackend.GPU || this.isMnnNpuBackend(backend);
      let scores: Float32Array;
      let boxes: Float32Array;
      let elapsedMs: number;
      const faceTrace = this.startTrace(`Static/FaceInfer/${backend}`, traceId);
      try {
        if (nativeMnn) {
          const result: GpuFaceResult = backend === InferenceBackend.GPU ?
            beautyPipeline.runGpuFace(this.staticImageTestFixture.slice(0)) :
            beautyPipeline.runMnnNpuFace(this.staticImageTestFixture.slice(0));
          scores = new Float32Array(result.scores);
          boxes = new Float32Array(result.boxes);
          elapsedMs = result.elapsedMs;
        } else {
          const faceModel = backend === InferenceBackend.CPU ? this.cpuFaceModel : this.npuFaceModel;
          if (!faceModel) throw new Error(`${backend} static model is unavailable`);
          const inputs = faceModel.getInputs();
          const uploadTrace = this.startTrace(`Static/Native/Face/TensorUpload/${backend}`, traceId);
          try {
            inputs[0].setData(this.staticImageTestFixture.slice(0));
          } finally {
            this.finishTrace(uploadTrace, traceId);
          }
          const predictTrace = this.startTrace(`Static/Native/Face/Predict/${backend}`, traceId);
          let outputs: Array<mindSporeLite.MSTensor>;
          try {
            outputs = await faceModel.predict(inputs);
          } finally {
            this.finishTrace(predictTrace, traceId);
          }
          const downloadTrace = this.startTrace(`Static/Native/Face/TensorRead/${backend}`, traceId);
          try {
            const scoreTensor = outputs.find((tensor: mindSporeLite.MSTensor) => tensor.shape[2] === 2);
            const boxTensor = outputs.find((tensor: mindSporeLite.MSTensor) => tensor.shape[2] === 4);
            if (!scoreTensor || !boxTensor) throw new Error(`${backend} static output layout mismatch`);
            scores = this.tensorDataAsFloat32(scoreTensor);
            boxes = this.tensorDataAsFloat32(boxTensor);
          } finally {
            this.finishTrace(downloadTrace, traceId);
          }
          elapsedMs = Date.now() - startedAt;
        }
      } finally {
        this.finishTrace(faceTrace, traceId);
      }

      let landmarkElapsedMs = 0;
      let landmarkValues: Float32Array;
      const landmarkStartedAt = Date.now();
      const landmarkTrace = this.startTrace(`Static/LandmarkInfer/${backend}`, traceId);
      try {
        if (nativeMnn) {
          const result: GpuLandmarkResult = backend === InferenceBackend.GPU ?
            beautyPipeline.runGpuLandmarks(this.staticLandmarkTestFixture.slice(0)) :
            beautyPipeline.runMnnNpuLandmarks(this.staticLandmarkTestFixture.slice(0));
          landmarkValues = new Float32Array(result.landmarks);
          landmarkElapsedMs = result.elapsedMs;
        } else {
          const landmarkModel = backend === InferenceBackend.CPU ? this.cpuLandmarkModel : this.npuLandmarkModel;
          if (!landmarkModel) throw new Error(`${backend} static landmark model is unavailable`);
          const inputs = landmarkModel.getInputs();
          const uploadTrace = this.startTrace(`Static/Native/Landmark/TensorUpload/${backend}`, traceId);
          try {
            inputs[0].setData(this.staticLandmarkTestFixture.slice(0));
          } finally {
            this.finishTrace(uploadTrace, traceId);
          }
          const predictTrace = this.startTrace(`Static/Native/Landmark/Predict/${backend}`, traceId);
          let outputs: Array<mindSporeLite.MSTensor>;
          try {
            outputs = await landmarkModel.predict(inputs);
          } finally {
            this.finishTrace(predictTrace, traceId);
          }
          const readTrace = this.startTrace(`Static/Native/Landmark/TensorRead/${backend}`, traceId);
          try {
            landmarkValues = this.tensorDataAsFloat32(outputs[0]);
          } finally {
            this.finishTrace(readTrace, traceId);
          }
          landmarkElapsedMs = Date.now() - landmarkStartedAt;
        }
      } finally {
        this.finishTrace(landmarkTrace, traceId);
      }
      const landmarkMaxError = this.maxAbsDifference(landmarkValues, this.staticLandmarkExpected);
      const detection = this.decodeFaceArrays(scores, boxes, 0.5625, 1);
      const expectedDetection: FaceDetectionResult = {
        confidence: 1, bestIndex: 4271, rawLocation: [], left: 0.263269,
        top: 0.169422, width: 0.567942, height: 0.577192
      };
      const iou = this.faceBoxIou(detection, expectedDetection);
      const pairRange = this.scorePairSumRange(scores);
      const nchwScores = this.scoresFromNchw(scores);
      const nchwDetection = this.decodeFaceArrays(nchwScores, boxes, 0.5625, 1);
      const nchwPairRange = this.scorePairSumRange(nchwScores);
      if (backend === InferenceBackend.MNN_USER_1 && this.staticImageTestCount === 0) {
        this.staticUser1Scores = scores.slice();
        this.staticUser1Boxes = boxes.slice();
      }
      const compareSummary = backend === InferenceBackend.MNN_USER_0 && this.staticUser1Scores &&
        this.staticUser1Boxes ? ` · U1差 score ${this.maxAbsDifference(scores, this.staticUser1Scores).toFixed(3)}` +
        ` box ${this.maxAbsDifference(boxes, this.staticUser1Boxes).toFixed(3)}` : '';
      this.staticImageTestCount++;
      this.staticElapsedTotalMs += elapsedMs;
      this.staticElapsedMinMs = Math.min(this.staticElapsedMinMs, elapsedMs);
      this.staticElapsedMaxMs = Math.max(this.staticElapsedMaxMs, elapsedMs);
      this.staticLandmarkElapsedTotalMs += landmarkElapsedMs;
      this.staticTotalWallMs += Date.now() - startedAt;
      if (this.staticImageBaselineIndex < 0) {
        this.staticImageBaselineIndex = detection.bestIndex;
        this.staticImageBaselineScore = detection.confidence;
      } else if (detection.bestIndex !== this.staticImageBaselineIndex ||
        Math.abs(detection.confidence - this.staticImageBaselineScore) > 0.001) {
        this.staticImageTestUnstableCount++;
      }
      if (this.staticImageTestCount === 1 || this.staticImageTestCount % 10 === 0) {
        this.inferenceStatus = `${backend} 静态图 ${this.staticImageTestCount} 次 · ` +
          `人脸均 ${(this.staticElapsedTotalMs / this.staticImageTestCount).toFixed(2)} ms ` +
          `68点均 ${(this.staticLandmarkElapsedTotalMs / this.staticImageTestCount).toFixed(2)} ms ` +
          `总均 ${(this.staticTotalWallMs / this.staticImageTestCount).toFixed(2)} ms ` +
          `[${this.staticElapsedMinMs.toFixed(2)}..${this.staticElapsedMaxMs.toFixed(2)}] · ` +
          `idx ${detection.bestIndex} · IoU ${iou.toFixed(3)} · pair ${pairRange[0].toFixed(3)}..` +
          `${pairRange[1].toFixed(3)} · NCHW idx ${nchwDetection.bestIndex} score ` +
          `${nchwDetection.confidence.toFixed(3)} pair ${nchwPairRange[0].toFixed(3)}..` +
          `${nchwPairRange[1].toFixed(3)} · 68点误差 ${landmarkMaxError.toFixed(4)} · ` +
          `漂移 ${this.staticImageTestUnstableCount} · ` +
          `总计 ${Date.now() - startedAt} ms${compareSummary}`;
        hilog.warn(DOMAIN, 'BeautyDemo',
          'STATIC_IMAGE %{public}s run=%{public}d face=%{public}s landmark=%{public}s total=%{public}d idx=%{public}d score=%{public}s pair=%{public}s..%{public}s landmarkMaxError=%{public}s unstable=%{public}d',
          backend, this.staticImageTestCount, elapsedMs.toFixed(2), landmarkElapsedMs.toFixed(2),
          Date.now() - startedAt, detection.bestIndex,
          detection.confidence.toFixed(4), pairRange[0].toFixed(4), pairRange[1].toFixed(4),
          landmarkMaxError.toFixed(5), this.staticImageTestUnstableCount);
      }
    } catch (error) {
      this.stopStaticImageTest();
      this.inferenceStatus = `${backend} 静态图测试失败 · ${String(error)}`;
    } finally {
      this.staticImageInferenceBusy = false;
      this.finishTrace(totalTrace, traceId);
    }
  }

  private nextTraceId(): number {
    this.traceSequence = this.traceSequence >= 2147483646 ? 1 : this.traceSequence + 1;
    return this.traceSequence;
  }

  private startTrace(stage: string, traceId: number): string {
    const name = `${TRACE_PREFIX}${stage}`;
    hiTraceMeter.startTrace(name, traceId);
    return name;
  }

  private finishTrace(name: string, traceId: number): void {
    hiTraceMeter.finishTrace(name, traceId);
  }

  private traceBackend(backend: InferenceBackend): void {
    const value = backend === InferenceBackend.NPU ? 1 : backend === InferenceBackend.CPU ? 2 :
      backend === InferenceBackend.GPU ? 3 : backend === InferenceBackend.MNN_USER_0 ? 4 : 5;
    hiTraceMeter.traceByValue(`${TRACE_PREFIX}State/Backend`, value);
  }

  private isMnnNpuBackend(backend: InferenceBackend): boolean {
    return backend === InferenceBackend.MNN_USER_0 || backend === InferenceBackend.MNN_USER_1;
  }

  private mnnNpuMode(backend: InferenceBackend): number {
    return backend === InferenceBackend.MNN_USER_0 ? 0 : 1;
  }

  private mnnNpuDescription(backend: InferenceBackend): string {
    return backend === InferenceBackend.MNN_USER_0 ? 'USER_0 整图' : 'USER_1 逐卷积';
  }

  private tensorDataAsFloat32(tensor: mindSporeLite.MSTensor): Float32Array {
    const data = tensor.getData();
    if (tensor.dtype === mindSporeLite.DataType.NUMBER_TYPE_FLOAT32) {
      return new Float32Array(data);
    }
    if (tensor.dtype !== mindSporeLite.DataType.NUMBER_TYPE_FLOAT16) {
      throw new Error(`Unsupported output dtype: ${tensor.dtype}`);
    }
    const source = new Uint16Array(data);
    const result = new Float32Array(source.length);
    for (let index = 0; index < source.length; index++) {
      const bits = source[index];
      const sign = (bits & 0x8000) !== 0 ? -1 : 1;
      const exponent = (bits >> 10) & 0x1F;
      const fraction = bits & 0x03FF;
      if (exponent === 0) {
        result[index] = sign * Math.pow(2, -14) * (fraction / 1024);
      } else if (exponent === 0x1F) {
        result[index] = fraction === 0 ? sign * Number.POSITIVE_INFINITY : Number.NaN;
      } else {
        result[index] = sign * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
      }
    }
    return result;
  }

  private decodeNpuFace(outputs: Array<mindSporeLite.MSTensor>, xScale: number,
    yScale: number): FaceDetectionResult {
    const scoreTensor = outputs.find((tensor: mindSporeLite.MSTensor) => tensor.shape[2] === 2);
    const boxTensor = outputs.find((tensor: mindSporeLite.MSTensor) => tensor.shape[2] === 4);
    if (!scoreTensor || !boxTensor) {
      throw new Error('Face detector output layout mismatch');
    }
    return this.decodeFaceArrays(this.tensorDataAsFloat32(scoreTensor), this.tensorDataAsFloat32(boxTensor),
      xScale, yScale);
  }

  private decodeFaceArrays(scores: Float32Array, boxes: Float32Array, xScale: number,
    yScale: number): FaceDetectionResult {
    if (scores.length !== 4420 * 2 || boxes.length !== 4420 * 4) {
      throw new Error(`Face detector output size mismatch: ${scores.length}/${boxes.length}`);
    }
    let bestIndex = -1;
    let bestScore = 0;
    for (let index = 0; index < 4420; index++) {
      const score = scores[index * 2 + 1];
      if (score > bestScore) {
        bestScore = score;
        bestIndex = index;
      }
    }
    if (bestIndex < 0) {
      return { confidence: 0, bestIndex: -1, rawLocation: [], left: 0, top: 0, width: 0, height: 0 };
    }

    const offset = bestIndex * 4;
    const priorX = this.facePriors[offset];
    const priorY = this.facePriors[offset + 1];
    const priorWidth = this.facePriors[offset + 2];
    const priorHeight = this.facePriors[offset + 3];
    const centerX = boxes[offset] * 0.1 * priorWidth + priorX;
    const centerY = boxes[offset + 1] * 0.1 * priorHeight + priorY;
    const width = Math.exp(boxes[offset + 2] * 0.2) * priorWidth;
    const height = Math.exp(boxes[offset + 3] * 0.2) * priorHeight;
    const contentLeft = (1 - xScale) / 2;
    const contentTop = (1 - yScale) / 2;
    const left = Math.max(0, (centerX - width / 2 - contentLeft) / xScale);
    const top = Math.max(0, (centerY - height / 2 - contentTop) / yScale);
    return {
      confidence: bestScore,
      bestIndex,
      rawLocation: [boxes[offset], boxes[offset + 1], boxes[offset + 2], boxes[offset + 3]],
      left,
      top,
      width: Math.min(1 - left, width / xScale),
      height: Math.min(1 - top, height / yScale)
    };
  }

  private summarizeFaceScores(scores: Float32Array): string {
    let pairSumMin = Number.POSITIVE_INFINITY;
    let pairSumMax = Number.NEGATIVE_INFINITY;
    let topOneIndex = -1;
    let topTwoIndex = -1;
    let topThreeIndex = -1;
    let topOneScore = Number.NEGATIVE_INFINITY;
    let topTwoScore = Number.NEGATIVE_INFINITY;
    let topThreeScore = Number.NEGATIVE_INFINITY;
    for (let index = 0; index < 4420; index++) {
      const first = scores[index * 2];
      const second = scores[index * 2 + 1];
      const pairSum = first + second;
      pairSumMin = Math.min(pairSumMin, pairSum);
      pairSumMax = Math.max(pairSumMax, pairSum);
      if (second > topOneScore) {
        topThreeIndex = topTwoIndex;
        topThreeScore = topTwoScore;
        topTwoIndex = topOneIndex;
        topTwoScore = topOneScore;
        topOneIndex = index;
        topOneScore = second;
      } else if (second > topTwoScore) {
        topThreeIndex = topTwoIndex;
        topThreeScore = topTwoScore;
        topTwoIndex = index;
        topTwoScore = second;
      } else if (second > topThreeScore) {
        topThreeIndex = index;
        topThreeScore = second;
      }
    }
    return `pairSum=${pairSumMin.toFixed(3)}..${pairSumMax.toFixed(3)} ` +
      `top=${topOneIndex}:${topOneScore.toFixed(3)},${topTwoIndex}:${topTwoScore.toFixed(3)},` +
      `${topThreeIndex}:${topThreeScore.toFixed(3)}`;
  }

  private faceBoxIou(first: FaceDetectionResult, second: FaceDetectionResult): number {
    const left = Math.max(first.left, second.left);
    const top = Math.max(first.top, second.top);
    const right = Math.min(first.left + first.width, second.left + second.width);
    const bottom = Math.min(first.top + first.height, second.top + second.height);
    const intersection = Math.max(0, right - left) * Math.max(0, bottom - top);
    const union = first.width * first.height + second.width * second.height - intersection;
    return union > 0 ? intersection / union : 0;
  }

  private prepareLandmarkInput(frame: FaceFrame, detection: FaceDetectionResult): LandmarkInput {
    const inputWidth = 320;
    const inputHeight = 240;
    const modelSize = 112;
    const contentLeft = (1 - frame.xScale) / 2;
    const contentTop = (1 - frame.yScale) / 2;
    const boxX = (contentLeft + detection.left * frame.xScale) * inputWidth;
    const boxY = (contentTop + detection.top * frame.yScale) * inputHeight;
    const boxWidth = detection.width * frame.xScale * inputWidth;
    const boxHeight = detection.height * frame.yScale * inputHeight;
    const padding = Math.floor(Math.max(boxWidth, boxHeight) * 0.1);
    const cropX = Math.max(0, Math.floor(boxX) - padding);
    const cropY = Math.max(0, Math.floor(boxY) - padding);
    const cropWidth = Math.min(inputWidth - cropX, Math.floor(boxWidth) + 2 * padding);
    const cropHeight = Math.min(inputHeight - cropY, Math.floor(boxHeight) + 2 * padding);
    const scale = Math.min(modelSize / cropWidth, modelSize / cropHeight);
    const scaledWidth = Math.max(1, Math.round(cropWidth * scale));
    const scaledHeight = Math.max(1, Math.round(cropHeight * scale));
    const padX = Math.floor((modelSize - scaledWidth) / 2);
    const padY = Math.floor((modelSize - scaledHeight) / 2);
    const source = new Float32Array(frame.data);
    const target = new Float32Array(3 * modelSize * modelSize);
    target.fill(128 / 255);
    const sourcePlane = inputWidth * inputHeight;
    const targetPlane = modelSize * modelSize;
    for (let y = 0; y < scaledHeight; y++) {
      const sourceY = Math.max(0, Math.min(inputHeight - 1, cropY + (y + 0.5) / scale - 0.5));
      const y0 = Math.floor(sourceY);
      const y1 = Math.min(inputHeight - 1, y0 + 1);
      const yWeight = sourceY - y0;
      for (let x = 0; x < scaledWidth; x++) {
        const sourceX = Math.max(0, Math.min(inputWidth - 1, cropX + (x + 0.5) / scale - 0.5));
        const x0 = Math.floor(sourceX);
        const x1 = Math.min(inputWidth - 1, x0 + 1);
        const xWeight = sourceX - x0;
        const targetOffset = (y + padY) * modelSize + x + padX;
        for (let channel = 0; channel < 3; channel++) {
          const plane = channel * sourcePlane;
          const top = source[plane + y0 * inputWidth + x0] * (1 - xWeight) +
            source[plane + y0 * inputWidth + x1] * xWeight;
          const bottom = source[plane + y1 * inputWidth + x0] * (1 - xWeight) +
            source[plane + y1 * inputWidth + x1] * xWeight;
          const normalizedFaceValue = top * (1 - yWeight) + bottom * yWeight;
          target[channel * targetPlane + targetOffset] = Math.max(0,
            Math.min(1, (normalizedFaceValue * 128 + 127) / 255));
        }
      }
    }
    return new LandmarkInput(target, cropX, cropY, padX, padY, scale);
  }

  private decodeLandmarks(outputTensor: mindSporeLite.MSTensor, transform: LandmarkInput,
    frame: FaceFrame): Float32Array {
    return this.decodeLandmarkValues(this.tensorDataAsFloat32(outputTensor), transform, frame);
  }

  private decodeLandmarkValues(output: Float32Array, transform: LandmarkInput,
    frame: FaceFrame): Float32Array {
    if (output.length !== 136) {
      throw new Error(`Landmark output mismatch: ${output.length}`);
    }
    const result = new Float32Array(136);
    const contentLeft = (1 - frame.xScale) / 2;
    const contentTop = (1 - frame.yScale) / 2;
    for (let index = 0; index < 68; index++) {
      const inputX = transform.cropX + (output[index * 2] * 112 - transform.padX) / transform.scale;
      const inputY = transform.cropY + (output[index * 2 + 1] * 112 - transform.padY) / transform.scale;
      result[index * 2] = Math.max(0, Math.min(1, (inputX / 320 - contentLeft) / frame.xScale));
      result[index * 2 + 1] = Math.max(0, Math.min(1, (inputY / 240 - contentTop) / frame.yScale));
    }
    return result;
  }

  private detectionFromLandmarks(landmarks: Float32Array): FaceDetectionResult {
    let left = 1;
    let top = 1;
    let right = 0;
    let bottom = 0;
    for (let index = 0; index < 68; index++) {
      left = Math.min(left, landmarks[index * 2]);
      top = Math.min(top, landmarks[index * 2 + 1]);
      right = Math.max(right, landmarks[index * 2]);
      bottom = Math.max(bottom, landmarks[index * 2 + 1]);
    }
    return {
      confidence: 1,
      bestIndex: -1,
      rawLocation: [],
      left: left,
      top: top,
      width: Math.max(0, right - left),
      height: Math.max(0, bottom - top)
    };
  }

  private prepareFaceMeshInput(frame: FaceFrame, detection: FaceDetectionResult): LandmarkInput {
    const inputWidth = 320;
    const inputHeight = 240;
    const modelSize = 192;
    const contentLeft = (1 - frame.xScale) / 2;
    const contentTop = (1 - frame.yScale) / 2;
    const boxX = (contentLeft + detection.left * frame.xScale) * inputWidth;
    const boxY = (contentTop + detection.top * frame.yScale) * inputHeight;
    const boxWidth = detection.width * frame.xScale * inputWidth;
    const boxHeight = detection.height * frame.yScale * inputHeight;
    const side = Math.max(boxWidth, boxHeight) * 1.5;
    const cropX = boxX + boxWidth * 0.5 - side * 0.5;
    const cropY = boxY + boxHeight * 0.5 - side * 0.5;
    const scale = modelSize / side;
    const source = new Float32Array(frame.data);
    const target = new Float32Array(3 * modelSize * modelSize);
    const sourcePlane = inputWidth * inputHeight;
    const targetPlane = modelSize * modelSize;
    for (let y = 0; y < modelSize; y++) {
      const sourceY = cropY + (y + 0.5) / scale - 0.5;
      if (sourceY < 0 || sourceY > inputHeight - 1) continue;
      const y0 = Math.floor(sourceY);
      const y1 = Math.min(inputHeight - 1, y0 + 1);
      const yWeight = sourceY - y0;
      for (let x = 0; x < modelSize; x++) {
        const sourceX = cropX + (x + 0.5) / scale - 0.5;
        if (sourceX < 0 || sourceX > inputWidth - 1) continue;
        const x0 = Math.floor(sourceX);
        const x1 = Math.min(inputWidth - 1, x0 + 1);
        const xWeight = sourceX - x0;
        const targetOffset = y * modelSize + x;
        for (let channel = 0; channel < 3; channel++) {
          const plane = channel * sourcePlane;
          const top = source[plane + y0 * inputWidth + x0] * (1 - xWeight) +
            source[plane + y0 * inputWidth + x1] * xWeight;
          const bottom = source[plane + y1 * inputWidth + x0] * (1 - xWeight) +
            source[plane + y1 * inputWidth + x1] * xWeight;
          target[channel * targetPlane + targetOffset] = Math.max(0,
            Math.min(1, ((top * (1 - yWeight) + bottom * yWeight) * 128 + 127) / 255));
        }
      }
    }
    return new LandmarkInput(target, cropX, cropY, 0, 0, scale);
  }

  private decodeFaceMesh(outputs: Array<mindSporeLite.MSTensor>, transform: LandmarkInput,
    frame: FaceFrame): Float32Array {
    const meshTensor = outputs.find((tensor: mindSporeLite.MSTensor) => tensor.dataSize === 468 * 3 * 4);
    const scoreTensor = outputs.find((tensor: mindSporeLite.MSTensor) => tensor.dataSize === 4);
    if (!meshTensor || !scoreTensor) {
      throw new Error(`Face Mesh output mismatch: ${outputs.map((tensor: mindSporeLite.MSTensor) => tensor.dataSize).join(',')}`);
    }
    const scoreLogit = this.tensorDataAsFloat32(scoreTensor)[0];
    const score = 1 / (1 + Math.exp(-scoreLogit));
    this.faceMeshScore = score;
    if (score < 0.5) return new Float32Array(0);
    const output = this.tensorDataAsFloat32(meshTensor);
    const result = new Float32Array(468 * 3);
    const contentLeft = (1 - frame.xScale) / 2;
    const contentTop = (1 - frame.yScale) / 2;
    for (let index = 0; index < 468; index++) {
      const inputX = transform.cropX + output[index * 3] / transform.scale;
      const inputY = transform.cropY + output[index * 3 + 1] / transform.scale;
      result[index * 3] = Math.max(0, Math.min(1, (inputX / 320 - contentLeft) / frame.xScale));
      result[index * 3 + 1] = Math.max(0, Math.min(1, (inputY / 240 - contentTop) / frame.yScale));
      result[index * 3 + 2] = output[index * 3 + 2] / transform.scale / 320;
    }
    return result;
  }

  private async loadNpuFaceMeshModel(context: mindSporeLite.Context): Promise<mindSporeLite.Model> {
    this.inferenceStatus = 'NPU 正在编译 468 点模型';
    const rawModel = await this.getAbilityContext().resourceManager.getRawFileContent('face_mesh_192.ms');
    const modelBuffer = rawModel.buffer.slice(rawModel.byteOffset, rawModel.byteOffset + rawModel.byteLength);
    const model = await mindSporeLite.loadModelFromBuffer(modelBuffer.slice(0), context);
    const inputs = model.getInputs();
    if (inputs.length !== 1 || inputs[0].dataSize !== 3 * 192 * 192 * 4) {
      throw new Error(`Face Mesh input mismatch: ${inputs.length}/${inputs[0]?.dataSize}`);
    }
    inputs[0].setData(new Float32Array(3 * 192 * 192).buffer);
    const outputs = await model.predict(inputs);
    const sizes = outputs.map((tensor: mindSporeLite.MSTensor) => tensor.dataSize).sort((a: number, b: number) => a - b);
    if (sizes.length !== 2 || sizes[0] !== 4 || sizes[1] !== 468 * 3 * 4) {
      throw new Error(`Face Mesh self-test shape mismatch: ${sizes.join(',')}`);
    }
    hilog.warn(DOMAIN, 'BeautyDemo', 'Face Mesh NPU model ready: outputs=%{public}s', sizes.join(','));
    return model;
  }

  private async loadLandmarkModel(context: mindSporeLite.Context,
    backend: InferenceBackend): Promise<mindSporeLite.Model> {
    if (this.inferenceBackend === backend) {
      this.inferenceStatus = `${backend} 正在编译 68 点模型`;
    }
    const rawModel = await this.getAbilityContext().resourceManager
      .getRawFileContent('landmarks_68_pfld_nofusion.ms');
    const modelBuffer = rawModel.buffer.slice(rawModel.byteOffset, rawModel.byteOffset + rawModel.byteLength);
    const model = await mindSporeLite.loadModelFromBuffer(modelBuffer.slice(0), context);
    const rawInput = await this.getAbilityContext().resourceManager
      .getRawFileContent('front_camera_landmark_input.bin');
    const inputBuffer = rawInput.buffer.slice(rawInput.byteOffset, rawInput.byteOffset + rawInput.byteLength);
    const rawExpected = await this.getAbilityContext().resourceManager
      .getRawFileContent('front_camera_landmark_expected.bin');
    const expectedBuffer = rawExpected.buffer.slice(rawExpected.byteOffset,
      rawExpected.byteOffset + rawExpected.byteLength);
    const inputs = model.getInputs();
    if (inputs.length !== 1 || inputs[0].dataSize !== rawInput.byteLength) {
      throw new Error(`Landmark input mismatch: ${inputs.length}/${inputs[0]?.dataSize}/${rawInput.byteLength}`);
    }
    inputs[0].setData(inputBuffer.slice(0));
    const outputs = await model.predict(inputs);
    if (outputs.length !== 1) {
      throw new Error(`Landmark output count mismatch: ${outputs.length}`);
    }
    const actual = this.tensorDataAsFloat32(outputs[0]);
    const expected = new Float32Array(expectedBuffer);
    if (actual.length !== expected.length || actual.length !== 136) {
      throw new Error(`Landmark self-test shape mismatch: ${actual.length}/${expected.length}`);
    }
    let maxError = 0;
    for (let index = 0; index < actual.length; index++) {
      maxError = Math.max(maxError, Math.abs(actual[index] - expected[index]));
    }
    if (maxError > 0.04) {
      throw new Error(`Landmark ${backend} self-test failed: maxError=${maxError.toFixed(5)}`);
    }
    hilog.warn(DOMAIN, 'BeautyDemo',
      'Landmark %{public}s self-test passed: points=68 maxError=%{public}s dtype=%{public}d',
      backend, maxError.toFixed(5), outputs[0].dtype);
    return model;
  }

  private async runFaceDetection(): Promise<void> {
    const backend = this.inferenceBackend;
    const nativeMnn = backend === InferenceBackend.GPU || this.isMnnNpuBackend(backend);
    const faceModel = backend === InferenceBackend.CPU ? this.cpuFaceModel : this.npuFaceModel;
    const landmarkModel = backend === InferenceBackend.CPU ? this.cpuLandmarkModel : this.npuLandmarkModel;
    const faceMeshModel = backend === InferenceBackend.NPU ? this.npuFaceMeshModel : null;
    const nativeReady = backend === InferenceBackend.GPU ? this.gpuReady : this.mnnNpuReadyBackend === backend;
    if ((nativeMnn ? !nativeReady : !faceModel) || this.inferenceBusy) {
      return;
    }
    const traceId = this.nextTraceId();
    const totalTrace = this.startTrace(`Frame/Total/${backend}`, traceId);
    const consumeTrace = this.startTrace('Frame/ConsumeInput', traceId);
    let frame: FaceFrame | null = null;
    try {
      frame = beautyPipeline.consumeFaceFrame();
    } finally {
      this.finishTrace(consumeTrace, traceId);
    }
    if (!frame) {
      this.finishTrace(totalTrace, traceId);
      return;
    }
    this.inferenceBusy = true;
    hiTraceMeter.traceByValue(`${TRACE_PREFIX}State/InferenceBusy`, 1);
    const startedAt = Date.now();
    try {
      let detection: FaceDetectionResult;
      let scores: Float32Array;
      let boxes: Float32Array;
      const faceInferTrace = this.startTrace(`Face/Infer/${backend}`, traceId);
      try {
        if (nativeMnn) {
          const mnnResult: GpuFaceResult = backend === InferenceBackend.GPU ?
            beautyPipeline.runGpuFace(frame.data) : beautyPipeline.runMnnNpuFace(frame.data);
          scores = new Float32Array(mnnResult.scores);
          boxes = new Float32Array(mnnResult.boxes);
        } else {
          if (!faceModel) throw new Error(`${backend} face model is unavailable`);
          const inputs = faceModel.getInputs();
          inputs[0].setData(frame.data);
          const outputs = await faceModel.predict(inputs);
          const scoreTensor = outputs.find((tensor: mindSporeLite.MSTensor) => tensor.shape[2] === 2);
          const boxTensor = outputs.find((tensor: mindSporeLite.MSTensor) => tensor.shape[2] === 4);
          if (!scoreTensor || !boxTensor) throw new Error('Face detector output layout mismatch');
          scores = this.tensorDataAsFloat32(scoreTensor);
          boxes = this.tensorDataAsFloat32(boxTensor);
        }
      } finally {
        this.finishTrace(faceInferTrace, traceId);
      }
      const faceDecodeTrace = this.startTrace('Face/DecodeSsd', traceId);
      try {
        detection = this.decodeFaceArrays(scores, boxes, frame.xScale, frame.yScale);
      } finally {
        this.finishTrace(faceDecodeTrace, traceId);
      }
      if (backend !== this.inferenceBackend) {
        return;
      }
      const bestScore = detection.confidence;
      hiTraceMeter.traceByValue(`${TRACE_PREFIX}State/FaceConfidencePermille`, Math.round(bestScore * 1000));
      this.inferenceCount++;
      if (this.inferenceCount % 10 === 1) {
        this.inferenceStatus = `${backend} 人脸检测 ${Date.now() - startedAt} ms · 最高 ${(bestScore * 100).toFixed(0)}%`;
        hilog.info(DOMAIN, 'BeautyDemo', '%{public}s live frame %{public}d: %{public}dms maxScore=%{public}s',
          backend, this.inferenceCount, Date.now() - startedAt, bestScore.toFixed(3));
      }
      const trackingThreshold = this.faceTrackingActive ? 0.45 : 0.65;
      if (bestScore < trackingThreshold) {
        this.consecutiveFaceMisses++;
        if (this.faceTrackingActive && this.consecutiveFaceMisses <= 8) {
          return;
        }
        this.faceTrackingActive = false;
        this.consecutiveFaceMeshMisses = 0;
        this.detectedFaceCount = 0;
        this.detectedLandmarkCount = 0;
        beautyPipeline.setFaceRegion(false, 0, 0, 1, 1);
        beautyPipeline.setLandmarks(new ArrayBuffer(0));
        beautyPipeline.setFaceMesh(new ArrayBuffer(0));
        return;
      }
      this.faceTrackingActive = true;
      this.consecutiveFaceMisses = 0;

      if (detection.width <= 0 || detection.height <= 0) {
        this.faceTrackingActive = false;
        this.consecutiveFaceMeshMisses = 0;
        this.detectedFaceCount = 0;
        this.detectedLandmarkCount = 0;
        beautyPipeline.setFaceRegion(false, 0, 0, 1, 1);
        beautyPipeline.setLandmarks(new ArrayBuffer(0));
        beautyPipeline.setFaceMesh(new ArrayBuffer(0));
        return;
      }
      this.detectedFaceCount = 1;
      const faceCommitTrace = this.startTrace('Output/SetFaceRegion', traceId);
      try {
        beautyPipeline.setFaceRegion(true, detection.left, detection.top, detection.width, detection.height);
      } finally {
        this.finishTrace(faceCommitTrace, traceId);
      }
      if (this.spiderMaskEnabled && faceMeshModel) {
        beautyPipeline.setLandmarks(new ArrayBuffer(0));
        const meshStartedAt = Date.now();
        let meshDetection = detection;
        if (landmarkModel) {
          const landmarkPreTrace = this.startTrace('Landmarks/Preprocess', traceId);
          let landmarkInput: LandmarkInput;
          try {
            landmarkInput = this.prepareLandmarkInput(frame, detection);
          } finally {
            this.finishTrace(landmarkPreTrace, traceId);
          }
          const landmarkInputs = landmarkModel.getInputs();
          landmarkInputs[0].setData(landmarkInput.data.buffer.slice(0));
          const landmarkInferTrace = this.startTrace(`Landmarks/Infer/${backend}`, traceId);
          let landmarkOutputs: Array<mindSporeLite.MSTensor>;
          try {
            landmarkOutputs = await landmarkModel.predict(landmarkInputs);
          } finally {
            this.finishTrace(landmarkInferTrace, traceId);
          }
          const landmarkDecodeTrace = this.startTrace('Landmarks/Decode', traceId);
          let landmarks: Float32Array;
          try {
            landmarks = this.decodeLandmarks(landmarkOutputs[0], landmarkInput, frame);
          } finally {
            this.finishTrace(landmarkDecodeTrace, traceId);
          }
          meshDetection = this.detectionFromLandmarks(landmarks);
        }
        const meshPreTrace = this.startTrace('FaceMesh/Preprocess', traceId);
        let meshInput: LandmarkInput;
        try {
          meshInput = this.prepareFaceMeshInput(frame, meshDetection);
        } finally {
          this.finishTrace(meshPreTrace, traceId);
        }
        const meshInputs = faceMeshModel.getInputs();
        meshInputs[0].setData(meshInput.data.buffer.slice(0));
        const meshInferTrace = this.startTrace('FaceMesh/Infer/NPU', traceId);
        let meshOutputs: Array<mindSporeLite.MSTensor>;
        try {
          meshOutputs = await faceMeshModel.predict(meshInputs);
        } finally {
          this.finishTrace(meshInferTrace, traceId);
        }
        const meshDecodeTrace = this.startTrace('FaceMesh/Decode', traceId);
        let mesh: Float32Array;
        try {
          mesh = this.decodeFaceMesh(meshOutputs, meshInput, frame);
        } finally {
          this.finishTrace(meshDecodeTrace, traceId);
        }
        if (mesh.length === 468 * 3) {
          this.consecutiveFaceMeshMisses = 0;
          beautyPipeline.setFaceMesh(mesh.buffer.slice(0));
          this.detectedLandmarkCount = 468;
        } else {
          this.consecutiveFaceMeshMisses++;
          if (this.consecutiveFaceMeshMisses > 8) {
            beautyPipeline.setFaceMesh(new ArrayBuffer(0));
            this.detectedLandmarkCount = 0;
          }
        }
        if (this.inferenceCount % 10 === 1) {
          this.inferenceStatus = `${backend} 检测+Face Mesh ${Date.now() - startedAt} ms · ${this.detectedLandmarkCount} 点`;
          hilog.info(DOMAIN, 'BeautyDemo',
            'NPU Face Mesh frame %{public}d: %{public}dms points=%{public}d score=%{public}s roi=%{public}s',
            this.inferenceCount, Date.now() - meshStartedAt, this.detectedLandmarkCount,
            this.faceMeshScore.toExponential(3), JSON.stringify(meshDetection));
        }
      } else if (landmarkModel || nativeMnn) {
        this.consecutiveFaceMeshMisses = 0;
        beautyPipeline.setFaceMesh(new ArrayBuffer(0));
        const landmarkStartedAt = Date.now();
        const landmarkPreTrace = this.startTrace('Landmarks/Preprocess', traceId);
        let landmarkInput: LandmarkInput;
        try {
          landmarkInput = this.prepareLandmarkInput(frame, detection);
        } finally {
          this.finishTrace(landmarkPreTrace, traceId);
        }
        let landmarks: Float32Array;
        const landmarkInferTrace = this.startTrace(`Landmarks/Infer/${backend}`, traceId);
        if (nativeMnn) {
          let mnnLandmarks: GpuLandmarkResult;
          try {
            mnnLandmarks = backend === InferenceBackend.GPU ?
              beautyPipeline.runGpuLandmarks(landmarkInput.data.buffer.slice(0)) :
              beautyPipeline.runMnnNpuLandmarks(landmarkInput.data.buffer.slice(0));
          } finally {
            this.finishTrace(landmarkInferTrace, traceId);
          }
          const landmarkDecodeTrace = this.startTrace('Landmarks/Decode', traceId);
          try {
            landmarks = this.decodeLandmarkValues(new Float32Array(mnnLandmarks.landmarks), landmarkInput, frame);
          } finally {
            this.finishTrace(landmarkDecodeTrace, traceId);
          }
        } else {
          let landmarkOutputs: Array<mindSporeLite.MSTensor>;
          try {
            if (!landmarkModel) throw new Error(`${backend} landmark model is unavailable`);
            const landmarkInputs = landmarkModel.getInputs();
            landmarkInputs[0].setData(landmarkInput.data.buffer.slice(0));
            landmarkOutputs = await landmarkModel.predict(landmarkInputs);
          } finally {
            this.finishTrace(landmarkInferTrace, traceId);
          }
          const landmarkDecodeTrace = this.startTrace('Landmarks/Decode', traceId);
          try {
            landmarks = this.decodeLandmarks(landmarkOutputs[0], landmarkInput, frame);
          } finally {
            this.finishTrace(landmarkDecodeTrace, traceId);
          }
        }
        const landmarkCommitTrace = this.startTrace('Output/SetLandmarks', traceId);
        try {
          beautyPipeline.setLandmarks(landmarks.buffer.slice(0));
        } finally {
          this.finishTrace(landmarkCommitTrace, traceId);
        }
        this.detectedLandmarkCount = 68;
        hiTraceMeter.traceByValue(`${TRACE_PREFIX}State/LandmarkCount`, 68);
        if (this.inferenceCount % 10 === 1) {
          this.inferenceStatus = `${backend} 检测+68点 ${Date.now() - startedAt} ms · 最高 ${(bestScore * 100).toFixed(0)}%`;
          hilog.info(DOMAIN, 'BeautyDemo', '%{public}s landmarks frame %{public}d: %{public}dms points=68',
            backend, this.inferenceCount, Date.now() - landmarkStartedAt);
          hilog.info(DOMAIN, 'BeautyDemo', '%{public}s total frame %{public}d: %{public}dms points=68',
            backend, this.inferenceCount, Date.now() - startedAt);
        }
      }
    } catch (error) {
      this.inferenceStatus = `${backend} 实时检测失败 · ${String(error)}`;
      this.consecutiveFaceMeshMisses = 0;
      this.detectedLandmarkCount = 0;
      beautyPipeline.setLandmarks(new ArrayBuffer(0));
      beautyPipeline.setFaceMesh(new ArrayBuffer(0));
      hilog.error(DOMAIN, 'BeautyDemo', '%{public}s live face detection failed: %{public}s', backend, String(error));
      this.stopFaceDetection();
    } finally {
      this.inferenceBusy = false;
      hiTraceMeter.traceByValue(`${TRACE_PREFIX}State/InferenceBusy`, 0);
      this.finishTrace(totalTrace, traceId);
    }
  }

  private getAbilityContext(): common.UIAbilityContext {
    const context = this.getUIContext().getHostContext();
    if (!context) {
      throw new Error('UIAbility context is unavailable');
    }
    return context as common.UIAbilityContext;
  }

  private async loadBeautyLuts(): Promise<void> {
    try {
      const resources = this.getAbilityContext().resourceManager;
      const gray = await resources.getRawFileContent('gpupixel_lookup_gray.rgba');
      const origin = await resources.getRawFileContent('gpupixel_lookup_origin.rgb');
      const skin = await resources.getRawFileContent('gpupixel_lookup_skin.rgb');
      const light = await resources.getRawFileContent('gpupixel_lookup_light.rgb');
      beautyPipeline.setBeautyLuts(
        gray.buffer.slice(gray.byteOffset, gray.byteOffset + gray.byteLength),
        origin.buffer.slice(origin.byteOffset, origin.byteOffset + origin.byteLength),
        skin.buffer.slice(skin.byteOffset, skin.byteOffset + skin.byteLength),
        light.buffer.slice(light.byteOffset, light.byteOffset + light.byteLength));
      hilog.info(DOMAIN, 'BeautyDemo', 'GPUPixel beauty LUTs loaded');
    } catch (error) {
      hilog.error(DOMAIN, 'BeautyDemo', 'Beauty LUT loading failed: %{public}s', String(error));
    }
  }

  private async loadFaceMeshTexture(): Promise<void> {
    try {
      const texture = await this.getAbilityContext().resourceManager.getRawFileContent('spider_face_mesh.rgba');
      beautyPipeline.setFaceMeshTexture(
        texture.buffer.slice(texture.byteOffset, texture.byteOffset + texture.byteLength));
      hilog.info(DOMAIN, 'BeautyDemo', 'Face Mesh texture loaded');
    } catch (error) {
      hilog.error(DOMAIN, 'BeautyDemo', 'Face Mesh texture loading failed: %{public}s', String(error));
    }
  }

  private async requestCameraPermission(): Promise<boolean> {
    const context = this.getAbilityContext();
    const manager = abilityAccessCtrl.createAtManager();
    const permissions: Array<Permissions> = ['ohos.permission.CAMERA'];
    const result = await manager.requestPermissionsFromUser(context, permissions);
    return result.authResults.length > 0 && result.authResults.every((value: number) => {
      return value === abilityAccessCtrl.GrantStatus.PERMISSION_GRANTED;
    });
  }

  private async probeNpuFaceModel(): Promise<void> {
    const startedAt = Date.now();
    try {
      const devices = mindSporeLite.getAllNNRTDeviceDescriptions();
      const accelerator = devices.find((device: mindSporeLite.NNRTDeviceDescription) => {
        return device.deviceType() === mindSporeLite.NNRTDeviceType.NNRTDEVICE_ACCELERATOR;
      });
      if (!accelerator) {
        this.inferenceStatus = 'NNRT 未发现 NPU';
        return;
      }

      this.inferenceStatus = 'NPU 正在编译人脸模型';
      const rawModel = await this.getAbilityContext().resourceManager
        .getRawFileContent('version-slim-320-nofusion.ms');
      const modelBuffer = rawModel.buffer.slice(rawModel.byteOffset, rawModel.byteOffset + rawModel.byteLength);
      const rawFixture = await this.getAbilityContext().resourceManager
        .getRawFileContent('front_camera_face_input.bin');
      const fixtureBuffer = rawFixture.buffer.slice(rawFixture.byteOffset,
        rawFixture.byteOffset + rawFixture.byteLength);
      const cpuExpected: FaceDetectionResult = {
        confidence: 1,
        bestIndex: 4271,
        rawLocation: [-0.781089, 0.512958, 0.313372, 1.833555],
        left: 0.263269,
        top: 0.169422,
        width: 0.567942,
        height: 0.577192
      };

      const context: mindSporeLite.Context = {
        target: ['nnrt'],
        nnrt: {
          deviceID: accelerator.deviceID(),
          performanceMode: mindSporeLite.PerformanceMode.PERFORMANCE_HIGH,
          priority: mindSporeLite.Priority.PRIORITY_HIGH
        }
      };
      const model = await mindSporeLite.loadModelFromBuffer(modelBuffer.slice(0), context);
      const inputs = model.getInputs();
      if (inputs.length !== 1) {
        throw new Error(`Unexpected input count: ${inputs.length}`);
      }
      if (rawFixture.byteLength !== inputs[0].dataSize) {
        throw new Error(`NPU fixture size mismatch: ${rawFixture.byteLength}/${inputs[0].dataSize}`);
      }
      inputs[0].setData(fixtureBuffer.slice(0));
      const outputs = await model.predict(inputs);
      const fixtureResult = this.decodeNpuFace(outputs, 0.5625, 1);
      const fixtureIou = this.faceBoxIou(fixtureResult, cpuExpected);
      if (fixtureResult.confidence < 0.99 || fixtureIou < 0.95) {
        const dtypes = outputs.map((tensor: mindSporeLite.MSTensor) => tensor.dtype.toString()).join(',');
        throw new Error(`NPU self-test failed: score=${fixtureResult.confidence.toFixed(4)} ` +
          `iou=${fixtureIou.toFixed(4)} dtypes=${dtypes} result=${JSON.stringify(fixtureResult)}`);
      }
      const landmarkModel = await this.loadLandmarkModel(context, InferenceBackend.NPU);
      let faceMeshModel: mindSporeLite.Model | null = null;
      try {
        faceMeshModel = await this.loadNpuFaceMeshModel(context);
      } catch (meshError) {
        hilog.warn(DOMAIN, 'BeautyDemo', 'Face Mesh NPU unavailable, keeping 68-point fallback: %{public}s',
          String(meshError));
      }
      this.npuFaceModel = model;
      this.npuLandmarkModel = landmarkModel;
      this.npuFaceMeshModel = faceMeshModel;
      const shapes = outputs.map((tensor: mindSporeLite.MSTensor) => tensor.shape.join('x')).join(', ');
      const dtypes = outputs.map((tensor: mindSporeLite.MSTensor) => tensor.dtype.toString()).join(', ');
      const elapsed = Date.now() - startedAt;
      if (this.inferenceBackend === InferenceBackend.NPU) {
        this.inferenceStatus = faceMeshModel ? `NPU 人脸 + 468 点就绪 ${elapsed} ms` :
          `NPU 人脸 + 68 点自检通过 ${elapsed} ms · IoU ${fixtureIou.toFixed(3)}`;
      }
      hilog.warn(DOMAIN, 'BeautyDemo',
        'NPU self-test passed: device=%{public}s elapsed=%{public}dms outputs=%{public}s dtypes=%{public}s ' +
        'score=%{public}s box=%{public}s iou=%{public}s', accelerator.deviceName(), elapsed, shapes, dtypes,
        fixtureResult.confidence.toFixed(4), JSON.stringify(fixtureResult), fixtureIou.toFixed(4));
      if (this.inferenceBackend === InferenceBackend.NPU) {
        this.startFaceDetection();
      }
    } catch (error) {
      const elapsed = Date.now() - startedAt;
      const detail = String(error);
      if (this.inferenceBackend === InferenceBackend.NPU) {
        this.inferenceStatus = detail.includes('NPU self-test failed') ?
          'NPU 自检未通过' : `NPU 模型失败 ${elapsed} ms · ${detail}`;
      }
      hilog.error(DOMAIN, 'BeautyDemo', 'NPU face model failed after %{public}dms: %{public}s', elapsed, detail);
    }
  }

  private async probeCpuFaceModel(): Promise<void> {
    const startedAt = Date.now();
    try {
      this.inferenceStatus = 'CPU 正在编译人脸模型';
      const resources = this.getAbilityContext().resourceManager;
      const rawModel = await resources.getRawFileContent('version-slim-320-nofusion.ms');
      const modelBuffer = rawModel.buffer.slice(rawModel.byteOffset, rawModel.byteOffset + rawModel.byteLength);
      const rawFixture = await resources.getRawFileContent('front_camera_face_input.bin');
      const fixtureBuffer = rawFixture.buffer.slice(rawFixture.byteOffset,
        rawFixture.byteOffset + rawFixture.byteLength);
      const context: mindSporeLite.Context = {
        target: ['cpu'],
        cpu: { threadNum: 4 }
      };
      const model = await mindSporeLite.loadModelFromBuffer(modelBuffer.slice(0), context);
      const inputs = model.getInputs();
      if (inputs.length !== 1 || inputs[0].dataSize !== rawFixture.byteLength) {
        throw new Error(`CPU input mismatch: ${inputs.length}/${inputs[0]?.dataSize}/${rawFixture.byteLength}`);
      }
      inputs[0].setData(fixtureBuffer.slice(0));
      const outputs = await model.predict(inputs);
      const fixtureResult = this.decodeNpuFace(outputs, 0.5625, 1);
      if (fixtureResult.confidence < 0.99 || fixtureResult.width <= 0 || fixtureResult.height <= 0) {
        throw new Error(`CPU self-test failed: ${JSON.stringify(fixtureResult)}`);
      }
      const landmarkModel = await this.loadLandmarkModel(context, InferenceBackend.CPU);
      this.cpuFaceModel = model;
      this.cpuLandmarkModel = landmarkModel;
      const elapsed = Date.now() - startedAt;
      if (this.inferenceBackend === InferenceBackend.CPU) {
        this.inferenceStatus = `CPU 人脸 + 68 点就绪 ${elapsed} ms · 4 线程`;
        this.startFaceDetection();
      }
      hilog.warn(DOMAIN, 'BeautyDemo',
        'CPU self-test passed: elapsed=%{public}dms score=%{public}s box=%{public}s threads=4',
        elapsed, fixtureResult.confidence.toFixed(4), JSON.stringify(fixtureResult));
    } catch (error) {
      const elapsed = Date.now() - startedAt;
      if (this.inferenceBackend === InferenceBackend.CPU) {
        this.inferenceStatus = `CPU 模型失败 ${elapsed} ms · ${String(error)}`;
      }
      hilog.error(DOMAIN, 'BeautyDemo', 'CPU face model failed after %{public}dms: %{public}s',
        elapsed, String(error));
    }
  }

  private async probeGpuFaceModel(): Promise<void> {
    const startedAt = Date.now();
    try {
      this.inferenceStatus = 'GPU 正在创建 MNN OpenCL 会话';
      const resources = this.getAbilityContext().resourceManager;
      const rawFaceModel = await resources.getRawFileContent('gpu_face_detector.mnn');
      const rawLandmarkModel = await resources.getRawFileContent('gpu_landmarks_68.mnn');
      const backend: string = beautyPipeline.initializeGpuInference(
        rawFaceModel.buffer.slice(rawFaceModel.byteOffset, rawFaceModel.byteOffset + rawFaceModel.byteLength),
        rawLandmarkModel.buffer.slice(rawLandmarkModel.byteOffset,
          rawLandmarkModel.byteOffset + rawLandmarkModel.byteLength));

      const rawFaceInput = await resources.getRawFileContent('front_camera_face_input.bin');
      const faceResult: GpuFaceResult = beautyPipeline.runGpuFace(
        rawFaceInput.buffer.slice(rawFaceInput.byteOffset, rawFaceInput.byteOffset + rawFaceInput.byteLength));
      const detection = this.decodeFaceArrays(new Float32Array(faceResult.scores),
        new Float32Array(faceResult.boxes), 0.5625, 1);
      if (detection.confidence < 0.99 || detection.width <= 0 || detection.height <= 0) {
        throw new Error(`GPU face self-test failed: ${JSON.stringify(detection)}`);
      }

      const rawLandmarkInput = await resources.getRawFileContent('front_camera_landmark_input.bin');
      const rawLandmarkExpected = await resources.getRawFileContent('front_camera_landmark_expected.bin');
      const landmarkResult: GpuLandmarkResult = beautyPipeline.runGpuLandmarks(rawLandmarkInput.buffer.slice(
        rawLandmarkInput.byteOffset, rawLandmarkInput.byteOffset + rawLandmarkInput.byteLength));
      const actual = new Float32Array(landmarkResult.landmarks);
      const expected = new Float32Array(rawLandmarkExpected.buffer.slice(rawLandmarkExpected.byteOffset,
        rawLandmarkExpected.byteOffset + rawLandmarkExpected.byteLength));
      let maxError = 0;
      for (let index = 0; index < actual.length; index++) {
        maxError = Math.max(maxError, Math.abs(actual[index] - expected[index]));
      }
      if (actual.length !== 136 || maxError > 0.04) {
        throw new Error(`GPU landmark self-test failed: count=${actual.length} maxError=${maxError.toFixed(5)}`);
      }

      this.gpuReady = true;
      const elapsed = Date.now() - startedAt;
      this.inferenceStatus = `GPU 人脸 + 68 点就绪 ${elapsed} ms · MNN OpenCL`;
      hilog.warn(DOMAIN, 'BeautyDemo',
        'GPU self-test passed: %{public}s init=%{public}dms face=%{public}sms landmark=%{public}sms score=%{public}s',
        backend, elapsed, faceResult.elapsedMs.toFixed(2), landmarkResult.elapsedMs.toFixed(2),
        detection.confidence.toFixed(4));
      if (this.inferenceBackend === InferenceBackend.GPU) this.startFaceDetection();
    } catch (error) {
      this.gpuReady = false;
      this.inferenceStatus = `GPU OpenCL 失败 ${Date.now() - startedAt} ms · ${String(error)}`;
      hilog.error(DOMAIN, 'BeautyDemo', 'GPU OpenCL initialization failed: %{public}s', String(error));
    }
  }

  private async probeMnnNpuFaceModel(selectedBackend: InferenceBackend): Promise<void> {
    const startedAt = Date.now();
    const description = this.mnnNpuDescription(selectedBackend);
    try {
      this.inferenceStatus = `${selectedBackend} ${description}正在在线编译模型`;
      const resources = this.getAbilityContext().resourceManager;
      const rawFaceModel = await resources.getRawFileContent('gpu_face_detector.mnn');
      const rawLandmarkModel = await resources.getRawFileContent('gpu_landmarks_68.mnn');
      const backend: string = beautyPipeline.initializeMnnNpuInference(
        rawFaceModel.buffer.slice(rawFaceModel.byteOffset, rawFaceModel.byteOffset + rawFaceModel.byteLength),
        rawLandmarkModel.buffer.slice(rawLandmarkModel.byteOffset,
          rawLandmarkModel.byteOffset + rawLandmarkModel.byteLength), this.mnnNpuMode(selectedBackend));
      const rawFaceInput = await resources.getRawFileContent('front_camera_face_input.bin');
      const faceResult: GpuFaceResult = beautyPipeline.runMnnNpuFace(
        rawFaceInput.buffer.slice(rawFaceInput.byteOffset, rawFaceInput.byteOffset + rawFaceInput.byteLength));
      const detection = this.decodeFaceArrays(new Float32Array(faceResult.scores),
        new Float32Array(faceResult.boxes), 0.5625, 1);
      const expectedDetection: FaceDetectionResult = {
        confidence: 1,
        bestIndex: 4271,
        rawLocation: [-0.781089, 0.512958, 0.313372, 1.833555],
        left: 0.263269,
        top: 0.169422,
        width: 0.567942,
        height: 0.577192
      };
      const detectionIou = this.faceBoxIou(detection, expectedDetection);
      if (detection.confidence < 0.99 || detectionIou < 0.95) {
        const rawScores = new Float32Array(faceResult.scores);
        const scoreSummary = this.summarizeFaceScores(rawScores);
        if (selectedBackend === InferenceBackend.MNN_USER_0) {
          throw new Error(`${selectedBackend} score diagnostic: ${scoreSummary}`);
        }
        throw new Error(`${selectedBackend} face self-test failed: iou=${detectionIou.toFixed(3)} ` +
          `idx=${detection.bestIndex} score=${detection.confidence.toFixed(3)} ${scoreSummary} ` +
          `p0=${rawScores[0].toFixed(3)},${rawScores[1].toFixed(3)} ` +
          `p4271=${rawScores[8542].toFixed(3)},${rawScores[8543].toFixed(3)} ` +
          `box=${detection.left.toFixed(3)},${detection.top.toFixed(3)},` +
          `${detection.width.toFixed(3)},${detection.height.toFixed(3)}`);
      }

      const rawLandmarkInput = await resources.getRawFileContent('front_camera_landmark_input.bin');
      const rawLandmarkExpected = await resources.getRawFileContent('front_camera_landmark_expected.bin');
      const landmarkResult: GpuLandmarkResult = beautyPipeline.runMnnNpuLandmarks(rawLandmarkInput.buffer.slice(
        rawLandmarkInput.byteOffset, rawLandmarkInput.byteOffset + rawLandmarkInput.byteLength));
      const actual = new Float32Array(landmarkResult.landmarks);
      const expected = new Float32Array(rawLandmarkExpected.buffer.slice(rawLandmarkExpected.byteOffset,
        rawLandmarkExpected.byteOffset + rawLandmarkExpected.byteLength));
      let maxError = 0;
      for (let index = 0; index < actual.length; index++) {
        maxError = Math.max(maxError, Math.abs(actual[index] - expected[index]));
      }
      if (actual.length !== 136 || maxError > 0.04) {
        throw new Error(`${selectedBackend} landmark self-test failed: count=${actual.length} maxError=${maxError.toFixed(5)}`);
      }

      this.mnnNpuReadyBackend = selectedBackend;
      const elapsed = Date.now() - startedAt;
      this.inferenceStatus = `${selectedBackend} 人脸 + 68 点就绪 ${elapsed} ms · ${description}`;
      hilog.warn(DOMAIN, 'BeautyDemo',
        '%{public}s self-test passed: %{public}s init=%{public}dms face=%{public}sms landmark=%{public}sms score=%{public}s',
        selectedBackend, backend, elapsed, faceResult.elapsedMs.toFixed(2), landmarkResult.elapsedMs.toFixed(2),
        detection.confidence.toFixed(4));
      if (this.inferenceBackend === selectedBackend) this.startFaceDetection();
    } catch (error) {
      this.mnnNpuReadyBackend = null;
      this.inferenceStatus = `${selectedBackend} ${description}失败 ${Date.now() - startedAt} ms · ${String(error)}`;
      hilog.error(DOMAIN, 'BeautyDemo', '%{public}s initialization failed: %{public}s',
        selectedBackend, String(error));
    }
  }

  private async selectInferenceBackend(backend: InferenceBackend): Promise<void> {
    if (this.inferenceBackend === backend) {
      return;
    }
    this.stopFaceDetection();
    this.stopStaticImageTest();
    this.inferenceBackend = backend;
    this.traceBackend(backend);
    this.inferenceCount = 0;
    this.faceTrackingActive = false;
    this.detectedFaceCount = 0;
    this.detectedLandmarkCount = 0;
    beautyPipeline.setFaceRegion(false, 0, 0, 1, 1);
    beautyPipeline.setLandmarks(new ArrayBuffer(0));
    beautyPipeline.setFaceMesh(new ArrayBuffer(0));
    hilog.warn(DOMAIN, 'BeautyDemo', 'Inference backend selected: %{public}s', backend);

    if (backend === InferenceBackend.GPU) {
      if (this.gpuReady) {
        this.inferenceStatus = 'GPU 人脸 + 68 点就绪 · MNN OpenCL';
        this.startFaceDetection();
      } else {
        await this.probeGpuFaceModel();
      }
      return;
    }
    if (this.isMnnNpuBackend(backend)) {
      if (this.mnnNpuReadyBackend === backend) {
        this.inferenceStatus = `${backend} 人脸 + 68 点就绪 · ${this.mnnNpuDescription(backend)}`;
        this.startFaceDetection();
      } else {
        await this.probeMnnNpuFaceModel(backend);
      }
      return;
    }
    if (backend === InferenceBackend.CPU) {
      if (this.cpuFaceModel && this.cpuLandmarkModel) {
        this.inferenceStatus = 'CPU 人脸 + 68 点就绪 · 4 线程';
        this.startFaceDetection();
      } else {
        await this.probeCpuFaceModel();
      }
      return;
    }
    if (this.npuFaceModel && this.npuLandmarkModel) {
      this.inferenceStatus = this.npuFaceMeshModel ? 'NPU 人脸 + 468 点就绪' : 'NPU 人脸 + 68 点就绪';
      this.startFaceDetection();
    } else {
      await this.probeNpuFaceModel();
    }
  }

  private async startCamera(targetFront: boolean = this.isFront): Promise<boolean> {
    if (!this.cameraSurfaceId || this.isStarting) {
      return false;
    }
    this.isStarting = true;
    this.isReady = false;
    this.statusText = '正在启动相机';
    try {
      const granted = await this.requestCameraPermission();
      if (!granted) {
        this.statusText = '需要相机权限';
        return false;
      }
      const position = targetFront ? camera.CameraPosition.CAMERA_POSITION_FRONT :
        camera.CameraPosition.CAMERA_POSITION_BACK;
      const context = this.getAbilityContext();
      const result: CameraStartResult = await this.cameraController.start(context, this.cameraSurfaceId, position,
        (faces: Array<camera.Rect>) => {
          if (this.npuFaceModel) {
            return;
          }
          this.detectedFaceCount = faces.length;
          if (faces.length > 0) {
            const face = faces[0];
            beautyPipeline.setFaceRegion(true, face.topLeftX, face.topLeftY, face.width, face.height);
          } else {
            beautyPipeline.setFaceRegion(false, 0, 0, 1, 1);
          }
        });
      this.systemBeautyAvailable = result.controlCenterSupported && result.beautyEffectSupported;
      this.systemBeautyEnabled = this.cameraController.isControlCenterEnabled();
      this.isFront = targetFront;
      this.isReady = true;
      this.statusText = `${targetFront ? '前摄' : '后摄'} · ${result.previewWidth} x ${result.previewHeight}`;
      return true;
    } catch (error) {
      this.statusText = '相机启动失败';
      hilog.error(DOMAIN, 'BeautyDemo', 'Camera start failed: %{public}s', JSON.stringify(error));
      return false;
    } finally {
      this.isStarting = false;
    }
  }

  private async switchCamera(): Promise<void> {
    if (this.isStarting) {
      return;
    }
    this.statusText = '正在切换镜头';
    await this.startCamera(!this.isFront);
  }

  private applyPreset(index: number): void {
    const preset = this.presets[index];
    this.selectedPreset = index;
    this.spiderMaskEnabled = false;
    this.smooth = preset.smooth;
    this.whiten = preset.whiten;
    this.rosy = preset.rosy;
    this.statusText = `已应用${preset.name}滤镜`;
    this.updateRealtimeEffects();
  }

  private applySpiderMask(): void {
    this.selectedPreset = -2;
    this.spiderMaskEnabled = true;
    this.realtimeEffectsEnabled = true;
    this.statusText = '赛博面具已开启';
    this.updateRealtimeEffects();
  }

  private updateRealtimeEffects(): void {
    try {
      const beautyEnabled = this.realtimeEffectsEnabled && !this.spiderMaskEnabled;
      beautyPipeline.setParameters(beautyEnabled, this.smooth, this.whiten, this.rosy);
      beautyPipeline.setReshapeParameters(
        beautyEnabled ? this.slimFace : 0,
        beautyEnabled ? this.bigEye : 0);
      beautyPipeline.setSpiderMaskEnabled(this.realtimeEffectsEnabled && this.spiderMaskEnabled);
    } catch (error) {
      hilog.warn(DOMAIN, 'BeautyDemo', 'Native effect update failed: %{public}s', JSON.stringify(error));
    }
  }

  private toggleSystemBeauty(enabled: boolean): void {
    try {
      const applied = this.cameraController.setControlCenterEnabled(enabled);
      this.systemBeautyEnabled = applied && enabled;
      if (!applied) {
        this.statusText = '当前设备不支持系统美颜';
      }
    } catch (error) {
      this.systemBeautyEnabled = false;
      this.statusText = '系统美颜切换失败';
      hilog.warn(DOMAIN, 'BeautyDemo', 'Beauty toggle failed: %{public}s', JSON.stringify(error));
    }
  }

  @Builder
  private backendOption(backend: InferenceBackend) {
    Text(backend)
      .fontSize(12)
      .fontWeight(this.inferenceBackend === backend ? FontWeight.Medium : FontWeight.Regular)
      .fontColor(this.inferenceBackend === backend ? '#101114' : '#B8BBC2')
      .textAlign(TextAlign.Center)
      .width(68)
      .height(32)
      .borderRadius(5)
      .backgroundColor(this.inferenceBackend === backend ? '#F4F5F7' : Color.Transparent)
      .onClick(() => this.selectInferenceBackend(backend));
  }

  @Builder
  private backendSelector() {
    Scroll() {
      Row() {
        this.backendOption(InferenceBackend.MNN_USER_0);
        this.backendOption(InferenceBackend.MNN_USER_1);
        this.backendOption(InferenceBackend.NPU);
        this.backendOption(InferenceBackend.CPU);
        this.backendOption(InferenceBackend.GPU);
      }
      .width(346)
      .padding(3)
    }
    .scrollable(ScrollDirection.Horizontal)
    .scrollBar(BarState.Off)
    .edgeEffect(EdgeEffect.None)
    .layoutWeight(1)
    .margin({ left: 8 })
    .borderRadius(7)
    .backgroundColor('#24262B');
  }

  @Builder
  private presetSelector() {
    Row() {
      ForEach(this.presets, (preset: BeautyPreset, index: number) => {
        Text(preset.name)
          .fontSize(14)
          .fontWeight(this.selectedPreset === index ? FontWeight.Medium : FontWeight.Regular)
          .fontColor(this.selectedPreset === index ? '#101114' : '#B8BBC2')
          .textAlign(TextAlign.Center)
          .layoutWeight(1)
          .height(34)
          .borderRadius(6)
          .backgroundColor(this.selectedPreset === index ? '#F4F5F7' : Color.Transparent)
          .onClick(() => this.applyPreset(index));
      }, (preset: BeautyPreset) => preset.name)
      Text('赛博面具')
        .fontSize(14)
        .fontWeight(this.spiderMaskEnabled ? FontWeight.Medium : FontWeight.Regular)
        .fontColor(this.spiderMaskEnabled ? Color.White : '#B8BBC2')
        .textAlign(TextAlign.Center)
        .layoutWeight(1)
        .height(34)
        .borderRadius(6)
        .backgroundColor(this.spiderMaskEnabled ? '#B7192B' : Color.Transparent)
        .onClick(() => this.applySpiderMask());
    }
    .width('100%')
    .padding(3)
    .borderRadius(8)
    .backgroundColor('#24262B');
  }

  build() {
    Column() {
      Stack({ alignContent: Alignment.Bottom }) {
        XComponent({
          id: 'beauty-preview',
          type: XComponentType.TEXTURE,
          controller: this.xComponentController
        })
          .width('100%')
          .height('100%')
          .renderFit(RenderFit.RESIZE_COVER)
          .onLoad(() => {
            this.surfaceId = this.xComponentController.getXComponentSurfaceId();
            this.xComponentController.setXComponentSurfaceRotation({ lock: true });
            this.cameraSurfaceId = beautyPipeline.create(this.surfaceId);
            this.loadBeautyLuts();
            this.loadFaceMeshTexture();
            const nnrtDevices: string = beautyPipeline.getNpuDevices();
            hilog.warn(DOMAIN, 'BeautyDemo', 'NNRT devices: %{public}s', nnrtDevices);
            this.probeNpuFaceModel();
            this.updateRealtimeEffects();
            this.startCamera();
          })
          .onDestroy(() => {
            this.stopFaceDetection();
            this.surfaceId = '';
            this.cameraSurfaceId = '';
            this.cameraController.release();
            beautyPipeline.release();
          });

        Column() {
          Row() {
            Column({ space: 2 }) {
              Text('美颜实验室')
                .fontSize(20)
                .fontWeight(FontWeight.Bold)
                .fontColor(Color.White);
              Text(this.statusText)
                .fontSize(12)
                .fontColor('#D9FFFFFF');
              Text(this.inferenceStatus)
                .fontSize(11)
                .fontColor('#B8FFFFFF')
                .maxLines(2)
                .textOverflow({ overflow: TextOverflow.Ellipsis });
            }
            .alignItems(HorizontalAlign.Start);
            Blank();
            Text(this.detectedFaceCount > 0 ? `人脸 ${this.detectedFaceCount} · 关键点 ${this.detectedLandmarkCount}` :
              (this.realtimeEffectsEnabled ? '实时滤镜 ON' : '实时滤镜 OFF'))
              .fontSize(12)
              .fontColor(Color.White)
              .padding({ left: 10, right: 10, top: 6, bottom: 6 })
              .borderRadius(6)
              .backgroundColor(this.realtimeEffectsEnabled ? '#D94C647C' : '#80000000');
          }
          .width('100%')
          .padding({ left: 18, right: 18, top: 16 });
          Blank();
          Row() {
            Button(this.isFront ? '切到后摄' : '切到前摄')
              .width(88)
              .height(40)
              .fontSize(13)
              .fontColor(Color.White)
              .backgroundColor('#80000000')
              .borderRadius(20)
              .onClick(() => this.switchCamera());
          }
          .width('100%')
          .padding({ left: 18, right: 18, bottom: 16 })
          .alignItems(VerticalAlign.Center);
        }
        .width('100%')
        .height('100%');
      }
      .width('100%')
      .aspectRatio(4 / 5)
      .backgroundColor('#191B20');

      Column({ space: 8 }) {
        Row() {
          Text('实时调节')
            .fontSize(17)
            .fontWeight(FontWeight.Medium)
            .fontColor('#F4F5F7');
          Blank();
          Text(this.spiderMaskEnabled ? 'GPU 赛博面具' : 'GPU 局部美颜')
            .fontSize(13)
            .fontColor('#B8BBC2');
          Toggle({ type: ToggleType.Switch, isOn: this.realtimeEffectsEnabled })
            .selectedColor('#FF647C')
            .margin({ left: 8 })
            .onChange((enabled: boolean) => {
              this.realtimeEffectsEnabled = enabled;
              this.statusText = enabled ? '实时滤镜已开启' : '实时滤镜已关闭';
              this.updateRealtimeEffects();
            });
          Text('标记')
            .fontSize(13)
            .fontColor('#B8BBC2')
            .margin({ left: 12 });
          Toggle({ type: ToggleType.Switch, isOn: this.debugOverlayEnabled })
            .selectedColor('#44C97A')
            .margin({ left: 6 })
            .onChange((enabled: boolean) => {
              this.debugOverlayEnabled = enabled;
              beautyPipeline.setDebugOverlay(enabled);
            });
        }
        .width('100%')
        .height(34);

        Row() {
          Text('推理后端')
            .width(52)
            .fontSize(13)
            .fontColor('#D9DBE0');
          this.backendSelector();
        }
        .width('100%')
        .height(38);

        Row() {
          Text('静态图')
            .width(52)
            .fontSize(13)
            .fontColor('#D9DBE0');
          Toggle({ type: ToggleType.Switch, isOn: this.staticImageTestRunning })
            .selectedColor('#44C97A')
            .onChange((enabled: boolean) => this.toggleStaticImageTest(enabled));
          Text(this.staticImageTestRunning ? `固定输入循环 ${this.staticImageTestCount} 次` : '固定输入循环')
            .fontSize(12)
            .fontColor('#B8BBC2')
            .margin({ left: 10 });
          Blank();
          Text('统一输入')
            .fontSize(11)
            .fontColor('#70747D');
        }
        .width('100%')
        .height(30);

        this.presetSelector();
        Row({ space: 12 }) {
          Text('柔光').width(42).fontSize(14).fontColor('#D9DBE0');
          Slider({ value: this.smooth, min: 0, max: 100, step: 1, style: SliderStyle.InSet })
            .layoutWeight(1)
            .selectedColor('#FF647C')
            .trackColor('#353840')
            .blockColor('#FFFFFF')
            .onChange((value: number) => {
              this.smooth = value;
              this.selectedPreset = -1;
              this.updateRealtimeEffects();
            });
          Text(Math.round(this.smooth).toString())
            .width(30).fontSize(12).fontColor('#8F939B').textAlign(TextAlign.End);
        }
        .width('100%')
        .height(38)
        .enabled(!this.spiderMaskEnabled)
        .opacity(this.spiderMaskEnabled ? 0.32 : 1);

        Row({ space: 18 }) {
          Column({ space: 2 }) {
            Row() {
              Text('瘦脸').fontSize(12).fontColor('#D9DBE0');
              Blank();
              Text(Math.round(this.slimFace).toString()).fontSize(11).fontColor('#8F939B');
            }
            .width('100%');
            Slider({ value: this.slimFace, min: 0, max: 100, step: 1, style: SliderStyle.InSet })
              .width('100%')
              .selectedColor('#FF647C')
              .trackColor('#353840')
              .blockColor('#FFFFFF')
              .onChange((value: number) => {
                this.slimFace = value;
                this.updateRealtimeEffects();
              });
          }
          .layoutWeight(1);

          Column({ space: 2 }) {
            Row() {
              Text('大眼').fontSize(12).fontColor('#D9DBE0');
              Blank();
              Text(Math.round(this.bigEye).toString()).fontSize(11).fontColor('#8F939B');
            }
            .width('100%');
            Slider({ value: this.bigEye, min: 0, max: 100, step: 1, style: SliderStyle.InSet })
              .width('100%')
              .selectedColor('#FF647C')
              .trackColor('#353840')
              .blockColor('#FFFFFF')
              .onChange((value: number) => {
                this.bigEye = value;
                this.updateRealtimeEffects();
              });
          }
          .layoutWeight(1);
        }
        .width('100%')
        .height(50)
        .enabled(!this.spiderMaskEnabled)
        .opacity(this.spiderMaskEnabled ? 0.32 : 1);

        Row({ space: 12 }) {
          Text('美白').width(42).fontSize(14).fontColor('#D9DBE0');
          Slider({ value: this.whiten, min: 0, max: 100, step: 1, style: SliderStyle.InSet })
            .layoutWeight(1)
            .selectedColor('#FF647C')
            .trackColor('#353840')
            .blockColor('#FFFFFF')
            .onChange((value: number) => {
              this.whiten = value;
              this.selectedPreset = -1;
              this.updateRealtimeEffects();
            });
          Text(Math.round(this.whiten).toString())
            .width(30).fontSize(12).fontColor('#8F939B').textAlign(TextAlign.End);
        }
        .width('100%')
        .height(38)
        .enabled(!this.spiderMaskEnabled)
        .opacity(this.spiderMaskEnabled ? 0.32 : 1);

        Row({ space: 12 }) {
          Text('气色').width(42).fontSize(14).fontColor('#D9DBE0');
          Slider({ value: this.rosy, min: 0, max: 100, step: 1, style: SliderStyle.InSet })
            .layoutWeight(1)
            .selectedColor('#FF647C')
            .trackColor('#353840')
            .blockColor('#FFFFFF')
            .onChange((value: number) => {
              this.rosy = value;
              this.selectedPreset = -1;
              this.updateRealtimeEffects();
            });
          Text(Math.round(this.rosy).toString())
            .width(30).fontSize(12).fontColor('#8F939B').textAlign(TextAlign.End);
        }
        .width('100%')
        .height(38)
        .enabled(!this.spiderMaskEnabled)
        .opacity(this.spiderMaskEnabled ? 0.32 : 1);
      }
      .width('100%')
      .padding({ left: 18, right: 18, top: 12, bottom: 14 })
      .backgroundColor('#17191D')
      .border({ width: { top: 1 }, color: '#303239' });
    }
    .width('100%')
    .height('100%')
    .backgroundColor('#101114');
  }
}
