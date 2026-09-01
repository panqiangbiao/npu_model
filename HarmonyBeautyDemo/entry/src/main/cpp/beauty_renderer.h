#ifndef HARMONY_BEAUTY_RENDERER_H
#define HARMONY_BEAUTY_RENDERER_H

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <EGL/egl.h>
#include <GLES3/gl3.h>
#include <native_image/native_image.h>
#include <native_window/external_window.h>

class BeautyRenderer {
public:
    BeautyRenderer() = default;
    ~BeautyRenderer();

    bool Start(uint64_t outputSurfaceId, uint64_t &inputSurfaceId, std::string &error);
    void Stop();
    void SetParameters(bool enabled, float smooth, float whiten, float rosy);
    void SetReshapeParameters(float slimFace, float bigEye);
    void SetSpiderMaskEnabled(bool enabled);
    void SetFaceRegion(bool hasFace, float left, float top, float width, float height);
    void SetLandmarks(const std::vector<float> &landmarks);
    void SetFaceMesh(const std::vector<float> &mesh);
    void SetFaceMeshTexture(const std::vector<uint8_t> &rgba);
    void SetDebugOverlay(bool enabled);
    void SetBeautyLuts(const std::vector<uint8_t> &gray, const std::vector<uint8_t> &origin,
        const std::vector<uint8_t> &skin, const std::vector<uint8_t> &light);
    bool ConsumeFaceInput(std::vector<float> &input, float &xScale, float &yScale);

private:
    struct Parameters {
        bool enabled = true;
        float smooth = 0.16f;
        float whiten = 0.10f;
        float rosy = 0.08f;
        float slimFace = 0.18f;
        float bigEye = 0.12f;
        bool spiderMaskEnabled = false;
        bool hasFace = false;
        float faceLeft = 0.0f;
        float faceTop = 0.0f;
        float faceWidth = 1.0f;
        float faceHeight = 1.0f;
        bool debugOverlay = false;
    };

    static void OnFrameAvailable(void *context);
    void NotifyFrame();
    void RenderLoop();
    bool InitializeGl(std::string &error);
    void RenderFrame();
    void CaptureFaceInput(const float *transform);
    void UpdateSkinMask(const std::vector<float> &landmarks);
    void RenderSmoothTexture(const float *transform);
    void UploadBeautyLuts();
    void DestroyGl();
    GLuint CompileShader(GLenum type, const char *source);
    GLuint CreateProgram();
    GLuint CreateFaceProgram();
    GLuint CreateLandmarkProgram();
    GLuint CreateFaceMeshProgram();
    GLuint CreateBlurProgram();

    uint64_t outputSurfaceId_ = 0;
    uint64_t inputSurfaceId_ = 0;
    std::thread renderThread_;
    std::mutex stateMutex_;
    std::condition_variable stateCv_;
    bool starting_ = false;
    bool initialized_ = false;
    bool running_ = false;
    bool frameAvailable_ = false;
    std::string startupError_;

    std::mutex parameterMutex_;
    Parameters parameters_;
    std::vector<float> targetLandmarks_;
    std::vector<float> landmarks_;
    bool targetLandmarksValid_ = false;
    float landmarkBlend_ = 0.0f;
    bool skinMaskDirty_ = true;
    std::vector<float> targetFaceMesh_;
    std::vector<float> faceMesh_;
    bool targetFaceMeshValid_ = false;
    float faceMeshBlend_ = 0.0f;

    std::mutex faceMeshTextureMutex_;
    std::vector<uint8_t> faceMeshTextureData_;
    bool faceMeshTexturePending_ = false;

    std::mutex lutMutex_;
    std::vector<uint8_t> grayLutData_;
    std::vector<uint8_t> originLutData_;
    std::vector<uint8_t> skinLutData_;
    std::vector<uint8_t> lightLutData_;
    bool lutUploadPending_ = false;
    bool beautyLutsReady_ = false;

    std::mutex faceInputMutex_;
    std::vector<float> faceInput_;
    bool faceInputReady_ = false;
    float faceInputXScale_ = 1.0f;
    float faceInputYScale_ = 1.0f;
    uint32_t frameCounter_ = 0;

    OHNativeWindow *outputWindow_ = nullptr;
    OH_NativeImage *nativeImage_ = nullptr;
    EGLDisplay eglDisplay_ = EGL_NO_DISPLAY;
    EGLContext eglContext_ = EGL_NO_CONTEXT;
    EGLSurface eglSurface_ = EGL_NO_SURFACE;
    GLuint externalTexture_ = 0;
    GLuint program_ = 0;
    GLuint faceProgram_ = 0;
    GLuint landmarkProgram_ = 0;
    GLuint faceMeshProgram_ = 0;
    GLuint blurProgram_ = 0;
    GLuint faceFramebuffer_ = 0;
    GLuint faceTexture_ = 0;
    GLuint skinMaskTexture_ = 0;
    GLuint beautySourceTexture_ = 0;
    GLuint beautyBlurTexture_ = 0;
    GLuint beautySourceFramebuffer_ = 0;
    GLuint beautyBlurFramebuffer_ = 0;
    GLuint grayLutTexture_ = 0;
    GLuint originLutTexture_ = 0;
    GLuint skinLutTexture_ = 0;
    GLuint lightLutTexture_ = 0;
    GLuint vertexArray_ = 0;
    GLuint vertexBuffer_ = 0;
    GLuint landmarkVertexArray_ = 0;
    GLuint landmarkVertexBuffer_ = 0;
    GLuint faceMeshVertexArray_ = 0;
    GLuint faceMeshVertexBuffer_ = 0;
    GLuint faceMeshIndexBuffer_ = 0;
    GLuint faceMeshTexture_ = 0;
    GLint transformLocation_ = -1;
    GLint texelLocation_ = -1;
    GLint smoothLocation_ = -1;
    GLint whitenLocation_ = -1;
    GLint rosyLocation_ = -1;
    GLint hasFaceLocation_ = -1;
    GLint faceRegionLocation_ = -1;
    GLint skinMaskLocation_ = -1;
    GLint hasSkinMaskLocation_ = -1;
    GLint smoothTextureLocation_ = -1;
    GLint debugOverlayLocation_ = -1;
    GLint grayLutLocation_ = -1;
    GLint originLutLocation_ = -1;
    GLint skinLutLocation_ = -1;
    GLint lightLutLocation_ = -1;
    GLint hasLutsLocation_ = -1;
    GLint landmarksLocation_ = -1;
    GLint hasLandmarksLocation_ = -1;
    GLint slimFaceLocation_ = -1;
    GLint bigEyeLocation_ = -1;
    GLint spiderMaskLocation_ = -1;
    GLint aspectRatioLocation_ = -1;
    GLint faceTransformLocation_ = -1;
    GLint faceScaleLocation_ = -1;
    GLint faceMeshOpacityLocation_ = -1;
    int outputWidth_ = 1;
    int outputHeight_ = 1;
    int beautyWidth_ = 1;
    int beautyHeight_ = 1;
};

#endif
