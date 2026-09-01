import { camera } from '@kit.CameraKit';
import { common } from '@kit.AbilityKit';
import { display } from '@kit.ArkUI';
import { hilog } from '@kit.PerformanceAnalysisKit';
import { BusinessError } from '@kit.BasicServicesKit';

const DOMAIN = 0xB001;

export interface CameraStartResult {
  controlCenterSupported: boolean;
  beautyEffectSupported: boolean;
  previewWidth: number;
  previewHeight: number;
}

export class BeautyCameraController {
  private manager: camera.CameraManager | undefined;
  private input: camera.CameraInput | undefined;
  private previewOutput: camera.PreviewOutput | undefined;
  private metadataOutput: camera.MetadataOutput | undefined;
  private session: camera.VideoSession | undefined;
  private controlCenterEnabled: boolean = false;
  private hasLoggedFaceRegion: boolean = false;

  async start(context: common.UIAbilityContext, surfaceId: string,
    position: camera.CameraPosition, onFaces?: (faces: Array<camera.Rect>) => void): Promise<CameraStartResult> {
    await this.release();
    this.hasLoggedFaceRegion = false;

    this.manager = camera.getCameraManager(context);
    const devices = this.manager.getSupportedCameras();
    let selected: camera.CameraDevice | undefined = devices.find((device: camera.CameraDevice) => {
      return device.cameraPosition === position;
    });
    if (!selected) {
      throw new Error(`No camera device is available for position ${position}`);
    }

    const modes = this.manager.getSupportedSceneModes(selected);
    if (modes.indexOf(camera.SceneMode.NORMAL_VIDEO) < 0) {
      throw new Error('The selected camera does not support normal video mode');
    }

    const capability = this.manager.getSupportedOutputCapability(selected, camera.SceneMode.NORMAL_VIDEO);
    const profile = this.pickPreviewProfile(capability.previewProfiles);

    this.input = this.manager.createCameraInput(selected);
    await this.input.open();
    this.previewOutput = this.manager.createPreviewOutput(profile, surfaceId);
    this.session = this.manager.createSession<camera.VideoSession>(camera.SceneMode.NORMAL_VIDEO);

    const supportsFaceMetadata = capability.supportedMetadataObjectTypes.indexOf(
      camera.MetadataObjectType.FACE_DETECTION) >= 0;
    if (supportsFaceMetadata) {
      this.metadataOutput = this.manager.createMetadataOutput([camera.MetadataObjectType.FACE_DETECTION]);
      this.metadataOutput.on('metadataObjectsAvailable', (error: BusinessError,
        objects: Array<camera.MetadataObject>) => {
        if (error) {
          hilog.warn(DOMAIN, 'BeautyDemo', 'Face metadata failed: %{public}d', error.code);
          return;
        }
        if (onFaces) {
          const faces = objects.map((object: camera.MetadataObject) => object.boundingBox);
          if (!this.hasLoggedFaceRegion && faces.length > 0) {
            const face = faces[0];
            hilog.warn(DOMAIN, 'BeautyDemo', 'First face region: %{public}s', JSON.stringify(face));
            this.hasLoggedFaceRegion = true;
          }
          onFaces(faces);
        }
      });
    }

    this.session.beginConfig();
    this.session.addInput(this.input);
    this.session.addOutput(this.previewOutput);
    if (this.metadataOutput) {
      this.session.addOutput(this.metadataOutput);
    }
    await this.session.commitConfig();

    const displayRotation = display.getDefaultDisplaySync().rotation;
    const previewRotation = this.previewOutput.getPreviewRotation(displayRotation);
    this.previewOutput.setPreviewRotation(previewRotation, true);

    let controlCenterSupported = false;
    let beautyEffectSupported = false;
    try {
      controlCenterSupported = this.session.isControlCenterSupported();
      if (controlCenterSupported) {
        const effects = this.session.getSupportedEffectTypes();
        beautyEffectSupported = effects.indexOf(camera.ControlCenterEffectType.BEAUTY) >= 0;
        this.session.enableControlCenter(false);
        this.controlCenterEnabled = false;
      }
    } catch (error) {
      hilog.warn(DOMAIN, 'BeautyDemo', 'Control center unavailable: %{public}s', JSON.stringify(error));
    }

    await this.session.start();
    if (this.metadataOutput) {
      await this.metadataOutput.start();
    }
    hilog.info(DOMAIN, 'BeautyDemo', 'Camera started: %{public}dx%{public}d',
      profile.size.width, profile.size.height);
    return {
      controlCenterSupported,
      beautyEffectSupported,
      previewWidth: profile.size.width,
      previewHeight: profile.size.height
    };
  }

  setControlCenterEnabled(enabled: boolean): boolean {
    if (!this.session || !this.session.isControlCenterSupported()) {
      return false;
    }
    const effects = this.session.getSupportedEffectTypes();
    if (effects.indexOf(camera.ControlCenterEffectType.BEAUTY) < 0) {
      return false;
    }
    this.session.enableControlCenter(enabled);
    this.controlCenterEnabled = enabled;
    return true;
  }

  isControlCenterEnabled(): boolean {
    return this.controlCenterEnabled;
  }

  async release(): Promise<void> {
    const session = this.session;
    const previewOutput = this.previewOutput;
    const metadataOutput = this.metadataOutput;
    const input = this.input;
    this.session = undefined;
    this.previewOutput = undefined;
    this.metadataOutput = undefined;
    this.input = undefined;
    this.controlCenterEnabled = false;

    if (metadataOutput) {
      try {
        metadataOutput.off('metadataObjectsAvailable');
        await metadataOutput.stop();
      } catch (error) {
        hilog.debug(DOMAIN, 'BeautyDemo', 'Metadata stop skipped: %{public}s', JSON.stringify(error));
      }
    }
    if (session) {
      try {
        await session.stop();
      } catch (error) {
        hilog.debug(DOMAIN, 'BeautyDemo', 'Session stop skipped: %{public}s', JSON.stringify(error));
      }
      try {
        await session.release();
      } catch (error) {
        hilog.warn(DOMAIN, 'BeautyDemo', 'Session release failed: %{public}s', JSON.stringify(error));
      }
    }
    if (previewOutput) {
      try {
        await previewOutput.release();
      } catch (error) {
        hilog.warn(DOMAIN, 'BeautyDemo', 'Preview release failed: %{public}s', JSON.stringify(error));
      }
    }
    if (metadataOutput) {
      try {
        await metadataOutput.release();
      } catch (error) {
        hilog.warn(DOMAIN, 'BeautyDemo', 'Metadata release failed: %{public}s', JSON.stringify(error));
      }
    }
    if (input) {
      try {
        await input.close();
      } catch (error) {
        hilog.warn(DOMAIN, 'BeautyDemo', 'Input close failed: %{public}s', JSON.stringify(error));
      }
    }
  }

  private pickPreviewProfile(profiles: Array<camera.Profile>): camera.Profile {
    if (profiles.length === 0) {
      throw new Error('No preview profile is available');
    }
    const targetArea = 1280 * 960;
    let selected = profiles[0];
    let selectedScore = Number.MAX_VALUE;
    profiles.forEach((profile: camera.Profile) => {
      const area = profile.size.width * profile.size.height;
      const ratio = Math.max(profile.size.width, profile.size.height) /
        Math.min(profile.size.width, profile.size.height);
      const score = Math.abs(area - targetArea) + Math.abs(ratio - 4 / 3) * targetArea;
      if (score < selectedScore) {
        selected = profile;
        selectedScore = score;
      }
    });
    return selected;
  }
}
