export interface FaceFrame {
  data: ArrayBuffer;
  xScale: number;
  yScale: number;
}

export interface GpuFaceResult {
  scores: ArrayBuffer;
  boxes: ArrayBuffer;
  elapsedMs: number;
}

export interface GpuLandmarkResult {
  landmarks: ArrayBuffer;
  elapsedMs: number;
}

export interface BeautyPipeline {
  create(outputSurfaceId: string): string;
  setParameters(enabled: boolean, smooth: number, whiten: number, rosy: number): void;
  setReshapeParameters(slimFace: number, bigEye: number): void;
  setSpiderMaskEnabled(enabled: boolean): void;
  setFaceRegion(hasFace: boolean, left: number, top: number, width: number, height: number): void;
  setLandmarks(landmarks: ArrayBuffer): void;
  setFaceMesh(mesh: ArrayBuffer): void;
  setFaceMeshTexture(rgba: ArrayBuffer): void;
  setDebugOverlay(enabled: boolean): void;
  setBeautyLuts(gray: ArrayBuffer, origin: ArrayBuffer, skin: ArrayBuffer, light: ArrayBuffer): void;
  getNpuDevices(): string;
  runCannProfiler(model: ArrayBuffer, outputDir: string, repeatCount: number): string;
  runTinyLlmProfiling(outputDir: string, repeatCount: number): string;
  runTinyGpt2Profiling(model: ArrayBuffer, inputEmbeddings: ArrayBuffer, outputDir: string, repeatCount: number): string;
  consumeFaceFrame(): FaceFrame | null;
  initializeGpuInference(faceModel: ArrayBuffer, landmarkModel: ArrayBuffer): string;
  runGpuFace(input: ArrayBuffer): GpuFaceResult;
  runGpuLandmarks(input: ArrayBuffer): GpuLandmarkResult;
  initializeMnnNpuInference(faceModel: ArrayBuffer, landmarkModel: ArrayBuffer, mode: number): string;
  runMnnNpuFace(input: ArrayBuffer): GpuFaceResult;
  runMnnNpuLandmarks(input: ArrayBuffer): GpuLandmarkResult;
  release(): void;
}

const beautyPipeline: BeautyPipeline;
export default beautyPipeline;
