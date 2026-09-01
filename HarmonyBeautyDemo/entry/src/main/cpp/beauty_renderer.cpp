#include "beauty_renderer.h"
#include "beauty_trace.h"
#include "face_mesh_data.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cmath>

#include <GLES2/gl2ext.h>
#include <hilog/log.h>

#undef LOG_DOMAIN
#undef LOG_TAG
#define LOG_DOMAIN 0xB001
#define LOG_TAG "BeautyRenderer"

namespace {
constexpr char VERTEX_SHADER[] = R"(#version 300 es
layout(location = 0) in vec2 aPosition;
layout(location = 1) in vec2 aTexCoord;
uniform highp mat4 uTexMatrix;
out vec2 vTexCoord;
out vec2 vDisplayCoord;
void main()
{
    gl_Position = vec4(aPosition, 0.0, 1.0);
    vTexCoord = (uTexMatrix * vec4(aTexCoord, 0.0, 1.0)).xy;
    vDisplayCoord = vec2(aTexCoord.x, 1.0 - aTexCoord.y);
}
)";

constexpr char FRAGMENT_SHADER[] = R"(#version 300 es
#extension GL_OES_EGL_image_external_essl3 : require
precision mediump float;
uniform samplerExternalOES uCamera;
uniform highp mat4 uTexMatrix;
uniform float uSmooth;
uniform float uWhiten;
uniform float uRosy;
uniform float uHasFace;
uniform vec4 uFaceRegion;
uniform sampler2D uSkinMask;
uniform float uHasSkinMask;
uniform sampler2D uSmoothTexture;
uniform float uShowDebug;
uniform sampler2D uLookupGray;
uniform sampler2D uLookupOrigin;
uniform sampler2D uLookupSkin;
uniform sampler2D uLookupLight;
uniform float uHasLuts;
uniform vec2 uLandmarks[68];
uniform float uHasLandmarks;
uniform float uSlimFace;
uniform float uBigEye;
uniform float uAspectRatio;
uniform float uSpiderMask;
in vec2 vTexCoord;
in vec2 vDisplayCoord;
out vec4 fragColor;

float skinWeight(vec3 color)
{
    float maximum = max(color.r, max(color.g, color.b));
    float minimum = min(color.r, min(color.g, color.b));
    float rgbRule = smoothstep(0.02, 0.10, color.r - color.g) *
        smoothstep(-0.02, 0.08, color.g - color.b) * smoothstep(0.08, 0.28, maximum - minimum);
    float y = dot(color, vec3(0.299, 0.587, 0.114));
    float cb = (color.b - y) * 0.564 + 0.5;
    float cr = (color.r - y) * 0.713 + 0.5;
    float chromaRule = smoothstep(0.29, 0.36, cb) * (1.0 - smoothstep(0.53, 0.60, cb)) *
        smoothstep(0.49, 0.55, cr) * (1.0 - smoothstep(0.70, 0.77, cr));
    return clamp(max(rgbRule, chromaRule) * smoothstep(0.08, 0.28, y), 0.0, 1.0);
}

vec3 rgbToYcbcr(vec3 color)
{
    float y = dot(color, vec3(0.299, 0.587, 0.114));
    return vec3(y, (color.b - y) * 0.564, (color.r - y) * 0.713);
}

vec3 ycbcrToRgb(vec3 color)
{
    return vec3(
        color.x + 1.403 * color.z,
        color.x - 0.344 * color.y - 0.714 * color.z,
        color.x + 1.773 * color.y);
}

vec3 sampleLut16(sampler2D lookupTexture, vec3 color)
{
    float blue = clamp(color.b, 0.0, 1.0) * 15.0;
    float floorBlue = floor(blue);
    float ceilBlue = ceil(blue);
    vec2 quad1 = vec2(floorBlue - floor(floorBlue * 0.25) * 4.0, floor(floorBlue * 0.25));
    vec2 quad2 = vec2(ceilBlue - floor(ceilBlue * 0.25) * 4.0, floor(ceilBlue * 0.25));
    vec2 offset = clamp(color.rg, 0.0, 1.0) * 0.234375 + 0.0078125;
    vec3 first = texture(lookupTexture, quad1 * 0.25 + offset).rgb;
    vec3 second = texture(lookupTexture, quad2 * 0.25 + offset).rgb;
    return mix(first, second, fract(blue));
}

vec3 sampleLut64(sampler2D lookupTexture, vec3 color)
{
    float blue = clamp(color.b, 0.0, 1.0) * 63.0;
    float floorBlue = floor(blue);
    float ceilBlue = ceil(blue);
    vec2 quad1 = vec2(floorBlue - floor(floorBlue / 8.0) * 8.0, floor(floorBlue / 8.0));
    vec2 quad2 = vec2(ceilBlue - floor(ceilBlue / 8.0) * 8.0, floor(ceilBlue / 8.0));
    vec2 firstUv = quad1 / 8.0 + vec2(0.5 / 512.0) +
        (vec2(1.0 / 8.0 - 1.0 / 512.0) * clamp(color.rg, 0.0, 1.0));
    vec2 secondUv = quad2 / 8.0 + vec2(0.5 / 512.0) +
        (vec2(1.0 / 8.0 - 1.0 / 512.0) * clamp(color.rg, 0.0, 1.0));
    return mix(texture(lookupTexture, firstUv).rgb, texture(lookupTexture, secondUv).rgb, fract(blue));
}

vec3 gpupixelWhiten(vec3 inputColor, float intensity)
{
    const float levelRangeInv = 1.02657;
    const float levelBlack = 0.0258820;
    const float alpha = 0.7;
    vec3 leveled = clamp((inputColor - vec3(levelBlack)) * levelRangeInv, 0.0, 1.0);
    vec3 grayMapped = vec3(
        texture(uLookupGray, vec2(leveled.r, 0.5)).r,
        texture(uLookupGray, vec2(leveled.g, 0.5)).g,
        texture(uLookupGray, vec2(leveled.b, 0.5)).b);
    vec3 texel = mix(inputColor, mix(leveled, grayMapped, 0.5), alpha);
    texel = mix(sampleLut16(uLookupOrigin, texel), leveled, alpha);
    vec3 skinTone = sampleLut16(uLookupSkin, texel);
    vec3 lightTone = sampleLut64(uLookupLight, skinTone);
    return mix(inputColor, lightTone, intensity);
}

vec2 metricPosition(vec2 point)
{
    return vec2(point.x, point.y / max(uAspectRatio, 0.001));
}

vec2 curveWarp(vec2 coordinate, vec2 origin, vec2 target, float amount)
{
    vec2 direction = (target - origin) * amount;
    float radius = max(distance(metricPosition(target), metricPosition(origin)), 0.001);
    float falloff = clamp(1.0 - distance(metricPosition(coordinate), metricPosition(origin)) / radius, 0.0, 1.0);
    return coordinate - direction * falloff * falloff;
}

vec2 enlargeEye(vec2 coordinate, vec2 center, float radius, float amount)
{
    float normalizedDistance = distance(metricPosition(coordinate), metricPosition(center)) / max(radius, 0.001);
    if (normalizedDistance >= 1.0) return coordinate;
    float scale = 1.0 - (1.0 - normalizedDistance * normalizedDistance) * amount;
    return center + (coordinate - center) * clamp(scale, 0.72, 1.0);
}

vec2 reshapeCoordinate(vec2 coordinate)
{
    if (uHasLandmarks < 0.5) return coordinate;
    const int jawIndices[6] = int[6](2, 4, 6, 14, 12, 10);
    vec2 nose = uLandmarks[30];
    for (int index = 0; index < 6; ++index) {
        vec2 origin = uLandmarks[jawIndices[index]];
        vec2 target = vec2(mix(origin.x, nose.x, 0.74), mix(origin.y, nose.y, 0.24));
        coordinate = curveWarp(coordinate, origin, target, uSlimFace * 0.10);
    }
    vec2 leftEyeCenter = (uLandmarks[36] + uLandmarks[39]) * 0.5;
    vec2 rightEyeCenter = (uLandmarks[42] + uLandmarks[45]) * 0.5;
    float leftRadius = distance(metricPosition(uLandmarks[36]), metricPosition(uLandmarks[39])) * 1.45;
    float rightRadius = distance(metricPosition(uLandmarks[42]), metricPosition(uLandmarks[45])) * 1.45;
    coordinate = enlargeEye(coordinate, leftEyeCenter, leftRadius, uBigEye * 0.22);
    coordinate = enlargeEye(coordinate, rightEyeCenter, rightRadius, uBigEye * 0.22);
    return clamp(coordinate, 0.0, 1.0);
}

float faceWeight(vec2 uv)
{
    if (uHasSkinMask > 0.5) {
        return texture(uSkinMask, uv).r;
    }
    if (uHasFace < 0.5) {
        return 1.0;
    }
    vec2 center = uFaceRegion.xy + uFaceRegion.zw * 0.5;
    vec2 radius = max(uFaceRegion.zw * vec2(0.62, 0.72), vec2(0.001));
    float distanceFromCenter = length((uv - center) / radius);
    return 1.0 - smoothstep(0.78, 1.04, distanceFromCenter);
}

float faceBoxBorder(vec2 uv)
{
    if (uHasFace < 0.5 || uShowDebug < 0.5) {
        return 0.0;
    }
    vec2 minimum = uFaceRegion.xy;
    vec2 maximum = minimum + uFaceRegion.zw;
    vec2 borderWidth = vec2(0.006, 0.004);
    float outer = step(minimum.x, uv.x) * step(minimum.y, uv.y) *
        step(uv.x, maximum.x) * step(uv.y, maximum.y);
    vec2 innerMinimum = minimum + borderWidth;
    vec2 innerMaximum = maximum - borderWidth;
    float inner = step(innerMinimum.x, uv.x) * step(innerMinimum.y, uv.y) *
        step(uv.x, innerMaximum.x) * step(uv.y, innerMaximum.y);
    return outer * (1.0 - inner);
}

float spiderEye(vec2 uv, vec2 outerCorner, vec2 innerCorner)
{
    vec2 outerMetric = metricPosition(outerCorner);
    vec2 innerMetric = metricPosition(innerCorner);
    vec2 center = (outerMetric + innerMetric) * 0.5;
    vec2 axisX = normalize(innerMetric - outerMetric);
    vec2 axisY = vec2(-axisX.y, axisX.x);
    vec2 delta = metricPosition(uv) - center;
    float halfWidth = max(distance(outerMetric, innerMetric) * 0.88, 0.001);
    float halfHeight = halfWidth * 0.72;
    vec2 lens = vec2(dot(delta, axisX) / halfWidth, dot(delta, axisY) / halfHeight);
    float pointedEllipse = pow(abs(lens.x), 1.45) + lens.y * lens.y;
    return 1.0 - smoothstep(0.76, 0.98, pointedEllipse);
}

vec3 applySpiderMask(vec3 color, vec2 uv, float opacity)
{
    vec2 leftEye = (metricPosition(uLandmarks[36]) + metricPosition(uLandmarks[39])) * 0.5;
    vec2 rightEye = (metricPosition(uLandmarks[42]) + metricPosition(uLandmarks[45])) * 0.5;
    vec2 axisX = normalize(rightEye - leftEye);
    vec2 axisY = vec2(-axisX.y, axisX.x);
    vec2 chin = metricPosition(uLandmarks[8]);
    vec2 nose = metricPosition(uLandmarks[30]);
    if (dot(axisY, chin - nose) < 0.0) axisY = -axisY;

    vec2 brow = (metricPosition(uLandmarks[19]) + metricPosition(uLandmarks[24])) * 0.5;
    vec2 forehead = brow + (brow - nose) * 1.18;
    vec2 center = (forehead + chin) * 0.5;
    float halfWidth = max(distance(metricPosition(uLandmarks[0]), metricPosition(uLandmarks[16])) * 0.55, 0.001);
    float halfHeight = max(distance(forehead, chin) * 0.52, 0.001);
    vec2 delta = metricPosition(uv) - center;
    vec2 local = vec2(dot(delta, axisX) / halfWidth, dot(delta, axisY) / halfHeight);
    float lowerTaper = mix(1.0, 0.78, smoothstep(0.15, 1.0, local.y));
    float faceShape = length(vec2(local.x / lowerTaper, local.y));
    float faceMask = 1.0 - smoothstep(0.91, 1.02, faceShape);

    float leftLens = spiderEye(uv, uLandmarks[36], uLandmarks[39]);
    float rightLens = spiderEye(uv, uLandmarks[45], uLandmarks[42]);
    float lens = max(leftLens, rightLens) * faceMask;
    float leftBorder = spiderEye(uv, uLandmarks[36] + (uLandmarks[36] - uLandmarks[39]) * 0.12,
        uLandmarks[39] + (uLandmarks[39] - uLandmarks[36]) * 0.12);
    float rightBorder = spiderEye(uv, uLandmarks[45] + (uLandmarks[45] - uLandmarks[42]) * 0.12,
        uLandmarks[42] + (uLandmarks[42] - uLandmarks[45]) * 0.12);
    float lensBorder = max(leftBorder, rightBorder) * faceMask;

    float radius = length(local);
    float angle = atan(local.y, local.x);
    float radialWeb = 1.0 - smoothstep(0.018, 0.055, abs(sin(angle * 6.0)));
    float ringWeb = 1.0 - smoothstep(0.018, 0.060, abs(sin(radius * 13.0)));
    float web = max(radialWeb, ringWeb) * faceMask * (1.0 - lensBorder);

    vec3 red = mix(vec3(0.34, 0.008, 0.015), vec3(0.84, 0.025, 0.045),
        clamp(1.12 - radius * 0.58, 0.0, 1.0));
    vec3 maskColor = mix(red, vec3(0.018, 0.020, 0.025), clamp(web * 0.92 + lensBorder * 0.96, 0.0, 1.0));
    maskColor = mix(maskColor, vec3(0.88, 0.93, 0.96), lens);
    return mix(color, maskColor, faceMask * opacity);
}

void main()
{
    vec2 reshapedDisplayCoord = reshapeCoordinate(vDisplayCoord);
    vec2 reshapedCameraCoord = (uTexMatrix *
        vec4(reshapedDisplayCoord.x, 1.0 - reshapedDisplayCoord.y, 0.0, 1.0)).xy;
    vec3 base = texture(uCamera, reshapedCameraCoord).rgb;
    vec3 lowFrequency = texture(uSmoothTexture,
        vec2(reshapedDisplayCoord.x, 1.0 - reshapedDisplayCoord.y)).rgb;
    vec3 detail = base - lowFrequency;
    float meanVariance = dot(detail * detail, vec3(2.6667));
    float skinGate = clamp((min(base.r, lowFrequency.r - 0.1) - 0.2) * 4.0, 0.0, 1.0);
    float smoothWeight = (1.0 - meanVariance / (meanVariance + 0.1)) * skinGate * uSmooth;
    vec3 color = mix(base, lowFrequency, clamp(smoothWeight, 0.0, 1.0));
    color += detail * (0.08 + uSmooth * 0.08);
    if (uHasLuts > 0.5 && uWhiten > 0.0) {
        color = gpupixelWhiten(clamp(color, 0.0, 1.0), uWhiten * 0.5);
    }
    vec3 rosyTone = rgbToYcbcr(color);
    float rosyMask = skinWeight(base) * uRosy;
    rosyTone.z += rosyMask * 0.018;
    rosyTone.y += rosyMask * 0.006;
    color = ycbcrToRgb(rosyTone);
    if (uSpiderMask > 0.0 && uHasLandmarks > 0.5) {
        color = applySpiderMask(color, vDisplayCoord, uSpiderMask);
    }
    color = mix(color, vec3(0.08, 1.0, 0.30), faceBoxBorder(vDisplayCoord) * 0.95);
    fragColor = vec4(clamp(color, 0.0, 1.0), 1.0);
}
)";

constexpr char FACE_VERTEX_SHADER[] = R"(#version 300 es
layout(location = 0) in vec2 aPosition;
layout(location = 1) in vec2 aTexCoord;
uniform highp mat4 uTexMatrix;
uniform vec2 uContentScale;
out vec2 vTexCoord;
void main()
{
    gl_Position = vec4(aPosition * uContentScale, 0.0, 1.0);
    vTexCoord = (uTexMatrix * vec4(aTexCoord, 0.0, 1.0)).xy;
}
)";

constexpr char FACE_FRAGMENT_SHADER[] = R"(#version 300 es
#extension GL_OES_EGL_image_external_essl3 : require
precision mediump float;
uniform samplerExternalOES uCamera;
in vec2 vTexCoord;
out vec4 fragColor;
void main()
{
    fragColor = vec4(texture(uCamera, vTexCoord).rgb, 1.0);
}
)";

constexpr char FACE_MESH_VERTEX_SHADER[] = R"(#version 300 es
layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec2 aTexCoord;
out vec2 vTexCoord;
void main()
{
    gl_Position = vec4(aPosition.x * 2.0 - 1.0, 1.0 - aPosition.y * 2.0, 0.0, 1.0);
    vTexCoord = aTexCoord;
}
)";

constexpr char FACE_MESH_FRAGMENT_SHADER[] = R"(#version 300 es
precision mediump float;
uniform sampler2D uMaskTexture;
uniform float uOpacity;
in vec2 vTexCoord;
out vec4 fragColor;
void main()
{
    vec4 mask = texture(uMaskTexture, vTexCoord);
    float maximum = max(mask.r, max(mask.g, mask.b));
    float minimum = min(mask.r, min(mask.g, mask.b));
    float chroma = maximum - minimum;
    float luminance = dot(mask.rgb, vec3(0.2126, 0.7152, 0.0722));
    float neutral = 1.0 - smoothstep(0.035, 0.12, chroma);
    float lightBackground = smoothstep(0.35, 0.64, luminance);
    float alpha = mask.a * (1.0 - neutral * lightBackground);
    fragColor = vec4(mask.rgb, alpha * uOpacity);
}
)";

constexpr char LANDMARK_VERTEX_SHADER[] = R"(#version 300 es
layout(location = 0) in vec2 aPosition;
void main()
{
    gl_Position = vec4(aPosition, 0.0, 1.0);
    gl_PointSize = 10.0;
}
)";

constexpr char LANDMARK_FRAGMENT_SHADER[] = R"(#version 300 es
precision mediump float;
out vec4 fragColor;
void main()
{
    vec2 offset = gl_PointCoord - vec2(0.5);
    if (dot(offset, offset) > 0.25) discard;
    fragColor = vec4(0.08, 1.0, 0.30, 1.0);
}
)";

constexpr char BLUR_VERTEX_SHADER[] = R"(#version 300 es
layout(location = 0) in vec2 aPosition;
layout(location = 1) in vec2 aTexCoord;
out vec2 vTexCoord;
void main()
{
    gl_Position = vec4(aPosition, 0.0, 1.0);
    vTexCoord = aTexCoord;
}
)";

constexpr char BLUR_FRAGMENT_SHADER[] = R"(#version 300 es
precision mediump float;
uniform sampler2D uTexture;
uniform vec2 uDirection;
in vec2 vTexCoord;
out vec4 fragColor;
void main()
{
    vec3 color = texture(uTexture, vTexCoord).rgb * 0.227027;
    color += texture(uTexture, vTexCoord + uDirection * 1.384615).rgb * 0.316216;
    color += texture(uTexture, vTexCoord - uDirection * 1.384615).rgb * 0.316216;
    color += texture(uTexture, vTexCoord + uDirection * 3.230769).rgb * 0.070270;
    color += texture(uTexture, vTexCoord - uDirection * 3.230769).rgb * 0.070270;
    fragColor = vec4(color, 1.0);
}
)";

constexpr int FACE_INPUT_WIDTH = 320;
constexpr int FACE_INPUT_HEIGHT = 240;
constexpr uint32_t FACE_CAPTURE_INTERVAL = 8;
constexpr int SKIN_MASK_WIDTH = 192;
constexpr int SKIN_MASK_HEIGHT = 256;

struct MaskPoint {
    float x;
    float y;
};

struct MaskBounds {
    float left = 1.0f;
    float top = 1.0f;
    float right = 0.0f;
    float bottom = 0.0f;
};

float SmoothStep(float edge0, float edge1, float value)
{
    const float t = std::clamp((value - edge0) / std::max(edge1 - edge0, 0.00001f), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

bool PointInPolygon(const std::vector<MaskPoint> &polygon, float x, float y)
{
    bool inside = false;
    for (size_t current = 0, previous = polygon.size() - 1; current < polygon.size(); previous = current++) {
        const MaskPoint &a = polygon[current];
        const MaskPoint &b = polygon[previous];
        const bool crosses = ((a.y > y) != (b.y > y)) &&
            (x < (b.x - a.x) * (y - a.y) / (b.y - a.y + 0.000001f) + a.x);
        if (crosses) inside = !inside;
    }
    return inside;
}

float DistanceToSegment(float x, float y, const MaskPoint &a, const MaskPoint &b)
{
    const float dx = b.x - a.x;
    const float dy = b.y - a.y;
    const float lengthSquared = dx * dx + dy * dy;
    const float t = lengthSquared > 0.0f ?
        std::clamp(((x - a.x) * dx + (y - a.y) * dy) / lengthSquared, 0.0f, 1.0f) : 0.0f;
    const float px = a.x + t * dx;
    const float py = a.y + t * dy;
    return std::sqrt((x - px) * (x - px) + (y - py) * (y - py));
}

MaskBounds BoundsFor(const std::vector<float> &landmarks, int first, int last)
{
    MaskBounds bounds;
    for (int index = first; index <= last; ++index) {
        bounds.left = std::min(bounds.left, landmarks[index * 2]);
        bounds.top = std::min(bounds.top, landmarks[index * 2 + 1]);
        bounds.right = std::max(bounds.right, landmarks[index * 2]);
        bounds.bottom = std::max(bounds.bottom, landmarks[index * 2 + 1]);
    }
    return bounds;
}

float OutsideFeatureEllipse(float x, float y, const MaskBounds &bounds, float xPadding, float yPadding)
{
    const float centerX = (bounds.left + bounds.right) * 0.5f;
    const float centerY = (bounds.top + bounds.bottom) * 0.5f;
    const float radiusX = std::max((bounds.right - bounds.left) * 0.5f + xPadding, 0.006f);
    const float radiusY = std::max((bounds.bottom - bounds.top) * 0.5f + yPadding, 0.006f);
    const float distance = std::sqrt(((x - centerX) / radiusX) * ((x - centerX) / radiusX) +
        ((y - centerY) / radiusY) * ((y - centerY) / radiusY));
    return SmoothStep(0.82f, 1.20f, distance);
}

constexpr std::array<float, 16> QUAD = {
    -1.0f, -1.0f, 0.0f, 0.0f,
     1.0f, -1.0f, 1.0f, 0.0f,
    -1.0f,  1.0f, 0.0f, 1.0f,
     1.0f,  1.0f, 1.0f, 1.0f,
};
}

BeautyRenderer::~BeautyRenderer()
{
    Stop();
}

bool BeautyRenderer::Start(uint64_t outputSurfaceId, uint64_t &inputSurfaceId, std::string &error)
{
    Stop();
    {
        std::lock_guard<std::mutex> lock(stateMutex_);
        outputSurfaceId_ = outputSurfaceId;
        inputSurfaceId_ = 0;
        startupError_.clear();
        starting_ = true;
        initialized_ = false;
        running_ = true;
        frameAvailable_ = false;
    }
    renderThread_ = std::thread(&BeautyRenderer::RenderLoop, this);

    std::unique_lock<std::mutex> lock(stateMutex_);
    const bool ready = stateCv_.wait_for(lock, std::chrono::seconds(5), [this] { return !starting_; });
    if (!ready || !initialized_) {
        error = ready ? startupError_ : "Timed out while creating GPU pipeline";
        lock.unlock();
        Stop();
        return false;
    }
    inputSurfaceId = inputSurfaceId_;
    return true;
}

void BeautyRenderer::Stop()
{
    {
        std::lock_guard<std::mutex> lock(stateMutex_);
        running_ = false;
        frameAvailable_ = true;
    }
    stateCv_.notify_all();
    if (renderThread_.joinable()) {
        renderThread_.join();
    }
}

void BeautyRenderer::SetParameters(bool enabled, float smooth, float whiten, float rosy)
{
    std::lock_guard<std::mutex> lock(parameterMutex_);
    parameters_.enabled = enabled;
    parameters_.smooth = std::clamp(smooth / 100.0f, 0.0f, 1.0f);
    parameters_.whiten = std::clamp(whiten / 100.0f, 0.0f, 1.0f);
    parameters_.rosy = std::clamp(rosy / 100.0f, 0.0f, 1.0f);
}

void BeautyRenderer::SetReshapeParameters(float slimFace, float bigEye)
{
    std::lock_guard<std::mutex> lock(parameterMutex_);
    parameters_.slimFace = std::clamp(slimFace / 100.0f, 0.0f, 1.0f);
    parameters_.bigEye = std::clamp(bigEye / 100.0f, 0.0f, 1.0f);
}

void BeautyRenderer::SetSpiderMaskEnabled(bool enabled)
{
    std::lock_guard<std::mutex> lock(parameterMutex_);
    parameters_.spiderMaskEnabled = enabled;
}

void BeautyRenderer::SetFaceRegion(bool hasFace, float left, float top, float width, float height)
{
    BeautyTrace::Scope trace("Native/Output/SetFaceRegion");
    std::lock_guard<std::mutex> lock(parameterMutex_);
    parameters_.hasFace = hasFace && left >= 0.0f && top >= 0.0f && width > 0.0f && height > 0.0f &&
        left <= 1.0f && top <= 1.0f && width <= 1.0f && height <= 1.0f;
    parameters_.faceLeft = left;
    parameters_.faceTop = top;
    parameters_.faceWidth = width;
    parameters_.faceHeight = height;
}

void BeautyRenderer::SetLandmarks(const std::vector<float> &landmarks)
{
    BeautyTrace::Scope trace("Native/Output/SetLandmarks");
    std::lock_guard<std::mutex> lock(parameterMutex_);
    targetLandmarksValid_ = landmarks.size() == 136;
    if (!targetLandmarksValid_) return;
    targetLandmarks_.clear();
    targetLandmarks_.reserve(landmarks.size());
    for (float coordinate : landmarks) {
        targetLandmarks_.push_back(std::clamp(coordinate, 0.0f, 1.0f));
    }
    if (landmarks_.size() != 136) {
        landmarks_ = targetLandmarks_;
        landmarkBlend_ = 0.0f;
    }
    skinMaskDirty_ = true;
    static bool logged = false;
    if (!logged) {
        OH_LOG_INFO(LOG_APP, "Received 68 landmarks, first=(%{public}.3f,%{public}.3f)",
            targetLandmarks_[0], targetLandmarks_[1]);
        logged = true;
    }
}

void BeautyRenderer::SetFaceMesh(const std::vector<float> &mesh)
{
    std::lock_guard<std::mutex> lock(parameterMutex_);
    targetFaceMeshValid_ = mesh.size() == 468 * 3;
    if (!targetFaceMeshValid_) return;
    targetFaceMesh_ = mesh;
    if (faceMesh_.size() != mesh.size()) {
        faceMesh_ = mesh;
        faceMeshBlend_ = 0.0f;
    }
}

void BeautyRenderer::SetFaceMeshTexture(const std::vector<uint8_t> &rgba)
{
    if (rgba.size() != 1024 * 1024 * 4) return;
    std::lock_guard<std::mutex> lock(faceMeshTextureMutex_);
    faceMeshTextureData_ = rgba;
    faceMeshTexturePending_ = true;
}

void BeautyRenderer::SetDebugOverlay(bool enabled)
{
    std::lock_guard<std::mutex> lock(parameterMutex_);
    parameters_.debugOverlay = enabled;
}

void BeautyRenderer::SetBeautyLuts(const std::vector<uint8_t> &gray, const std::vector<uint8_t> &origin,
    const std::vector<uint8_t> &skin, const std::vector<uint8_t> &light)
{
    if (gray.size() != 256 * 4 || origin.size() != 64 * 64 * 3 || skin.size() != 64 * 64 * 3 ||
        light.size() != 512 * 512 * 3) {
        OH_LOG_ERROR(LOG_APP, "Invalid GPUPixel LUT sizes: %{public}zu/%{public}zu/%{public}zu/%{public}zu",
            gray.size(), origin.size(), skin.size(), light.size());
        return;
    }
    std::lock_guard<std::mutex> lock(lutMutex_);
    grayLutData_ = gray;
    originLutData_ = origin;
    skinLutData_ = skin;
    lightLutData_ = light;
    lutUploadPending_ = true;
}

bool BeautyRenderer::ConsumeFaceInput(std::vector<float> &input, float &xScale, float &yScale)
{
    BeautyTrace::Scope trace("Native/Input/ConsumeFaceFrame");
    std::lock_guard<std::mutex> lock(faceInputMutex_);
    if (!faceInputReady_) return false;
    input = std::move(faceInput_);
    xScale = faceInputXScale_;
    yScale = faceInputYScale_;
    faceInputReady_ = false;
    return true;
}

void BeautyRenderer::OnFrameAvailable(void *context)
{
    static_cast<BeautyRenderer *>(context)->NotifyFrame();
}

void BeautyRenderer::NotifyFrame()
{
    {
        std::lock_guard<std::mutex> lock(stateMutex_);
        frameAvailable_ = true;
    }
    stateCv_.notify_one();
}

void BeautyRenderer::RenderLoop()
{
    std::string error;
    const bool initialized = InitializeGl(error);
    {
        std::lock_guard<std::mutex> lock(stateMutex_);
        initialized_ = initialized;
        startupError_ = error;
        starting_ = false;
        if (!initialized) running_ = false;
    }
    stateCv_.notify_all();
    if (!initialized) {
        OH_LOG_ERROR(LOG_APP, "GPU pipeline initialization failed: %{public}s", error.c_str());
        DestroyGl();
        return;
    }

    OH_LOG_INFO(LOG_APP, "GPU beauty pipeline started, input=%{public}" PRIu64 ", output=%{public}dx%{public}d",
        inputSurfaceId_, outputWidth_, outputHeight_);
    while (true) {
        std::unique_lock<std::mutex> lock(stateMutex_);
        stateCv_.wait(lock, [this] { return frameAvailable_ || !running_; });
        if (!running_) break;
        frameAvailable_ = false;
        lock.unlock();
        RenderFrame();
    }
    DestroyGl();
}

bool BeautyRenderer::InitializeGl(std::string &error)
{
    if (OH_NativeWindow_CreateNativeWindowFromSurfaceId(outputSurfaceId_, &outputWindow_) != 0 ||
        outputWindow_ == nullptr) {
        error = "Failed to open XComponent surface";
        return false;
    }
    eglDisplay_ = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (eglDisplay_ == EGL_NO_DISPLAY || !eglInitialize(eglDisplay_, nullptr, nullptr)) {
        error = "Failed to initialize EGL";
        return false;
    }
    const EGLint configAttributes[] = {
        EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
        EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
        EGL_NONE
    };
    EGLConfig config = nullptr;
    EGLint configCount = 0;
    if (!eglChooseConfig(eglDisplay_, configAttributes, &config, 1, &configCount) || configCount == 0) {
        error = "Failed to select EGL configuration";
        return false;
    }
    const EGLint contextAttributes[] = { EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE };
    eglContext_ = eglCreateContext(eglDisplay_, config, EGL_NO_CONTEXT, contextAttributes);
    const EGLNativeWindowType nativeWindow = reinterpret_cast<EGLNativeWindowType>(outputWindow_);
    eglSurface_ = eglCreateWindowSurface(eglDisplay_, config, nativeWindow, nullptr);
    if (eglContext_ == EGL_NO_CONTEXT || eglSurface_ == EGL_NO_SURFACE ||
        !eglMakeCurrent(eglDisplay_, eglSurface_, eglSurface_, eglContext_)) {
        error = "Failed to create EGL rendering context";
        return false;
    }
    eglQuerySurface(eglDisplay_, eglSurface_, EGL_WIDTH, &outputWidth_);
    eglQuerySurface(eglDisplay_, eglSurface_, EGL_HEIGHT, &outputHeight_);
    eglSwapInterval(eglDisplay_, 1);

    glGenTextures(1, &externalTexture_);
    glBindTexture(GL_TEXTURE_EXTERNAL_OES, externalTexture_);
    glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    nativeImage_ = OH_NativeImage_Create(externalTexture_, GL_TEXTURE_EXTERNAL_OES);
    if (nativeImage_ == nullptr || OH_NativeImage_GetSurfaceId(nativeImage_, &inputSurfaceId_) != 0) {
        error = "Failed to create camera input surface";
        return false;
    }
    OH_NativeImage_SetDropBufferMode(nativeImage_, true);
    OH_OnFrameAvailableListener listener { this, OnFrameAvailable };
    if (OH_NativeImage_SetOnFrameAvailableListener(nativeImage_, listener) != 0) {
        error = "Failed to register camera frame listener";
        return false;
    }

    program_ = CreateProgram();
    faceProgram_ = CreateFaceProgram();
    landmarkProgram_ = CreateLandmarkProgram();
    faceMeshProgram_ = CreateFaceMeshProgram();
    blurProgram_ = CreateBlurProgram();
    if (program_ == 0 || faceProgram_ == 0 || landmarkProgram_ == 0 || faceMeshProgram_ == 0 || blurProgram_ == 0) {
        error = "Failed to compile beauty shader";
        return false;
    }
    transformLocation_ = glGetUniformLocation(program_, "uTexMatrix");
    texelLocation_ = glGetUniformLocation(program_, "uTexel");
    smoothLocation_ = glGetUniformLocation(program_, "uSmooth");
    whitenLocation_ = glGetUniformLocation(program_, "uWhiten");
    rosyLocation_ = glGetUniformLocation(program_, "uRosy");
    hasFaceLocation_ = glGetUniformLocation(program_, "uHasFace");
    faceRegionLocation_ = glGetUniformLocation(program_, "uFaceRegion");
    skinMaskLocation_ = glGetUniformLocation(program_, "uSkinMask");
    hasSkinMaskLocation_ = glGetUniformLocation(program_, "uHasSkinMask");
    smoothTextureLocation_ = glGetUniformLocation(program_, "uSmoothTexture");
    debugOverlayLocation_ = glGetUniformLocation(program_, "uShowDebug");
    grayLutLocation_ = glGetUniformLocation(program_, "uLookupGray");
    originLutLocation_ = glGetUniformLocation(program_, "uLookupOrigin");
    skinLutLocation_ = glGetUniformLocation(program_, "uLookupSkin");
    lightLutLocation_ = glGetUniformLocation(program_, "uLookupLight");
    hasLutsLocation_ = glGetUniformLocation(program_, "uHasLuts");
    landmarksLocation_ = glGetUniformLocation(program_, "uLandmarks");
    hasLandmarksLocation_ = glGetUniformLocation(program_, "uHasLandmarks");
    slimFaceLocation_ = glGetUniformLocation(program_, "uSlimFace");
    bigEyeLocation_ = glGetUniformLocation(program_, "uBigEye");
    spiderMaskLocation_ = glGetUniformLocation(program_, "uSpiderMask");
    aspectRatioLocation_ = glGetUniformLocation(program_, "uAspectRatio");
    faceTransformLocation_ = glGetUniformLocation(faceProgram_, "uTexMatrix");
    faceScaleLocation_ = glGetUniformLocation(faceProgram_, "uContentScale");
    faceMeshOpacityLocation_ = glGetUniformLocation(faceMeshProgram_, "uOpacity");

    glGenTextures(1, &faceTexture_);
    glBindTexture(GL_TEXTURE_2D, faceTexture_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, FACE_INPUT_WIDTH, FACE_INPUT_HEIGHT, 0,
        GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glGenFramebuffers(1, &faceFramebuffer_);
    glBindFramebuffer(GL_FRAMEBUFFER, faceFramebuffer_);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, faceTexture_, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        error = "Failed to create face input framebuffer";
        return false;
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    std::vector<uint8_t> initialMask(SKIN_MASK_WIDTH * SKIN_MASK_HEIGHT, 255);
    glGenTextures(1, &skinMaskTexture_);
    glBindTexture(GL_TEXTURE_2D, skinMaskTexture_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, SKIN_MASK_WIDTH, SKIN_MASK_HEIGHT, 0,
        GL_RED, GL_UNSIGNED_BYTE, initialMask.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    beautyWidth_ = std::max(1, outputWidth_ / 2);
    beautyHeight_ = std::max(1, outputHeight_ / 2);
    glGenTextures(1, &beautySourceTexture_);
    glBindTexture(GL_TEXTURE_2D, beautySourceTexture_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, beautyWidth_, beautyHeight_, 0,
        GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glGenFramebuffers(1, &beautySourceFramebuffer_);
    glBindFramebuffer(GL_FRAMEBUFFER, beautySourceFramebuffer_);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, beautySourceTexture_, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        error = "Failed to create beauty source framebuffer";
        return false;
    }
    glGenTextures(1, &beautyBlurTexture_);
    glBindTexture(GL_TEXTURE_2D, beautyBlurTexture_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, beautyWidth_, beautyHeight_, 0,
        GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glGenFramebuffers(1, &beautyBlurFramebuffer_);
    glBindFramebuffer(GL_FRAMEBUFFER, beautyBlurFramebuffer_);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, beautyBlurTexture_, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        error = "Failed to create beauty blur framebuffer";
        return false;
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    const auto createLutTexture = [](GLuint &texture) {
        constexpr uint8_t WHITE_PIXEL[4] = { 255, 255, 255, 255 };
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, WHITE_PIXEL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    };
    createLutTexture(grayLutTexture_);
    createLutTexture(originLutTexture_);
    createLutTexture(skinLutTexture_);
    createLutTexture(lightLutTexture_);
    glGenVertexArrays(1, &vertexArray_);
    glGenBuffers(1, &vertexBuffer_);
    glBindVertexArray(vertexArray_);
    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(QUAD), QUAD.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), nullptr);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
        reinterpret_cast<const void *>(2 * sizeof(float)));
    glBindVertexArray(0);
    glGenVertexArrays(1, &landmarkVertexArray_);
    glGenBuffers(1, &landmarkVertexBuffer_);
    glBindVertexArray(landmarkVertexArray_);
    glBindBuffer(GL_ARRAY_BUFFER, landmarkVertexBuffer_);
    glBufferData(GL_ARRAY_BUFFER, 136 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), nullptr);
    glBindVertexArray(0);

    glGenVertexArrays(1, &faceMeshVertexArray_);
    glGenBuffers(1, &faceMeshVertexBuffer_);
    glGenBuffers(1, &faceMeshIndexBuffer_);
    glBindVertexArray(faceMeshVertexArray_);
    glBindBuffer(GL_ARRAY_BUFFER, faceMeshVertexBuffer_);
    glBufferData(GL_ARRAY_BUFFER, 468 * 5 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), nullptr);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float),
        reinterpret_cast<const void *>(3 * sizeof(float)));
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, faceMeshIndexBuffer_);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, FACE_MESH_INDICES.size() * sizeof(uint16_t),
        FACE_MESH_INDICES.data(), GL_STATIC_DRAW);
    glBindVertexArray(0);

    constexpr uint8_t RED_PIXEL[4] = { 190, 12, 30, 255 };
    glGenTextures(1, &faceMeshTexture_);
    glBindTexture(GL_TEXTURE_2D, faceMeshTexture_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, RED_PIXEL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    if (glGetError() != GL_NO_ERROR) {
        error = "Failed to initialize OpenGL resources";
        return false;
    }
    return true;
}

void BeautyRenderer::RenderFrame()
{
    BeautyTrace::Scope trace("Native/Render/Frame");
    if (OH_NativeImage_UpdateSurfaceImage(nativeImage_) != 0) return;
    float transform[16] = {0.0f};
    OH_NativeImage_GetTransformMatrixV2(nativeImage_, transform);
    Parameters parameters;
    std::vector<float> landmarks;
    std::vector<float> faceMesh;
    float landmarkBlend = 0.0f;
    float faceMeshBlend = 0.0f;
    bool updateSkinMask = false;
    {
        std::lock_guard<std::mutex> lock(parameterMutex_);
        parameters = parameters_;
        if (targetLandmarksValid_ && targetLandmarks_.size() == 136) {
            float squaredMotion = 0.0f;
            for (size_t index = 0; index < landmarks_.size(); index += 2) {
                const float dx = targetLandmarks_[index] - landmarks_[index];
                const float dy = targetLandmarks_[index + 1] - landmarks_[index + 1];
                squaredMotion += dx * dx + dy * dy;
            }
            const float motion = std::sqrt(squaredMotion / 68.0f);
            const float smoothing = std::clamp(0.24f + motion * 4.0f, 0.24f, 0.60f);
            for (size_t index = 0; index < landmarks_.size(); ++index) {
                landmarks_[index] += (targetLandmarks_[index] - landmarks_[index]) * smoothing;
            }
            landmarkBlend_ = std::min(1.0f, landmarkBlend_ + 0.20f);
        } else {
            landmarkBlend_ = std::max(0.0f, landmarkBlend_ - 0.08f);
            if (landmarkBlend_ == 0.0f && !landmarks_.empty()) {
                landmarks_.clear();
                skinMaskDirty_ = true;
            }
        }
        landmarks = landmarks_;
        landmarkBlend = landmarkBlend_;
        if (targetFaceMeshValid_ && targetFaceMesh_.size() == 468 * 3) {
            float squaredMotion = 0.0f;
            for (size_t index = 0; index < faceMesh_.size(); index += 3) {
                const float dx = targetFaceMesh_[index] - faceMesh_[index];
                const float dy = targetFaceMesh_[index + 1] - faceMesh_[index + 1];
                squaredMotion += dx * dx + dy * dy;
            }
            const float motion = std::sqrt(squaredMotion / 468.0f);
            const float smoothing = std::clamp(0.28f + motion * 4.0f, 0.28f, 0.65f);
            for (size_t index = 0; index < faceMesh_.size(); ++index) {
                faceMesh_[index] += (targetFaceMesh_[index] - faceMesh_[index]) * smoothing;
            }
            faceMeshBlend_ = std::min(1.0f, faceMeshBlend_ + 0.20f);
        } else {
            faceMeshBlend_ = std::max(0.0f, faceMeshBlend_ - 0.06f);
            if (faceMeshBlend_ == 0.0f) faceMesh_.clear();
        }
        faceMesh = faceMesh_;
        faceMeshBlend = faceMeshBlend_;
        updateSkinMask = skinMaskDirty_;
        skinMaskDirty_ = false;
    }
    if (updateSkinMask) UpdateSkinMask(landmarks);
    {
        std::lock_guard<std::mutex> lock(faceMeshTextureMutex_);
        if (faceMeshTexturePending_) {
            glBindTexture(GL_TEXTURE_2D, faceMeshTexture_);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1024, 1024, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                faceMeshTextureData_.data());
            faceMeshTexturePending_ = false;
        }
    }
    {
        BeautyTrace::Scope renderEffects("Native/Render/BeautyEffects");
        UploadBeautyLuts();
        RenderSmoothTexture(transform);
    }
    const float enabled = parameters.enabled ? 1.0f : 0.0f;
    glViewport(0, 0, outputWidth_, outputHeight_);
    glUseProgram(program_);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_EXTERNAL_OES, externalTexture_);
    glUniform1i(glGetUniformLocation(program_, "uCamera"), 0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, skinMaskTexture_);
    glUniform1i(skinMaskLocation_, 1);
    glUniform1f(hasSkinMaskLocation_, landmarks.size() == 136 && landmarkBlend > 0.0f ? 1.0f : 0.0f);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, beautySourceTexture_);
    glUniform1i(smoothTextureLocation_, 2);
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, grayLutTexture_);
    glUniform1i(grayLutLocation_, 3);
    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_2D, originLutTexture_);
    glUniform1i(originLutLocation_, 4);
    glActiveTexture(GL_TEXTURE5);
    glBindTexture(GL_TEXTURE_2D, skinLutTexture_);
    glUniform1i(skinLutLocation_, 5);
    glActiveTexture(GL_TEXTURE6);
    glBindTexture(GL_TEXTURE_2D, lightLutTexture_);
    glUniform1i(lightLutLocation_, 6);
    glUniform1f(hasLutsLocation_, beautyLutsReady_ ? 1.0f : 0.0f);
    glActiveTexture(GL_TEXTURE0);
    glUniformMatrix4fv(transformLocation_, 1, GL_FALSE, transform);
    glUniform1f(smoothLocation_, parameters.smooth * enabled);
    glUniform1f(whitenLocation_, parameters.whiten * enabled);
    glUniform1f(rosyLocation_, parameters.rosy * enabled);
    glUniform1f(hasLandmarksLocation_, landmarks.size() == 136 && landmarkBlend > 0.0f ? 1.0f : 0.0f);
    glUniform1f(slimFaceLocation_, parameters.slimFace * enabled * landmarkBlend);
    glUniform1f(bigEyeLocation_, parameters.bigEye * enabled * landmarkBlend);
    glUniform1f(spiderMaskLocation_, parameters.spiderMaskEnabled && faceMesh.size() != 468 * 3 ? landmarkBlend : 0.0f);
    glUniform1f(aspectRatioLocation_, static_cast<float>(outputWidth_) / static_cast<float>(outputHeight_));
    if (landmarks.size() == 136) glUniform2fv(landmarksLocation_, 68, landmarks.data());
    glUniform1f(hasFaceLocation_, parameters.hasFace ? 1.0f : 0.0f);
    glUniform1f(debugOverlayLocation_, parameters.debugOverlay ? 1.0f : 0.0f);
    glUniform4f(faceRegionLocation_, parameters.faceLeft, parameters.faceTop,
        parameters.faceWidth, parameters.faceHeight);
    glBindVertexArray(vertexArray_);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    if (parameters.spiderMaskEnabled && faceMesh.size() == 468 * 3 && faceMeshBlend > 0.0f) {
        std::vector<float> vertices(468 * 5);
        for (size_t index = 0; index < 468; ++index) {
            vertices[index * 5] = faceMesh[index * 3];
            vertices[index * 5 + 1] = faceMesh[index * 3 + 1];
            vertices[index * 5 + 2] = faceMesh[index * 3 + 2];
            vertices[index * 5 + 3] = FACE_MESH_UVS[index * 2];
            vertices[index * 5 + 4] = FACE_MESH_UVS[index * 2 + 1];
        }
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glUseProgram(faceMeshProgram_);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, faceMeshTexture_);
        glUniform1i(glGetUniformLocation(faceMeshProgram_, "uMaskTexture"), 0);
        glUniform1f(faceMeshOpacityLocation_, faceMeshBlend);
        glBindVertexArray(faceMeshVertexArray_);
        glBindBuffer(GL_ARRAY_BUFFER, faceMeshVertexBuffer_);
        glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.size() * sizeof(float), vertices.data());
        glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(FACE_MESH_INDICES.size()), GL_UNSIGNED_SHORT, nullptr);
        glDisable(GL_BLEND);
    }
    if (parameters.debugOverlay && landmarks.size() == 136) {
        for (size_t index = 0; index < landmarks.size(); index += 2) {
            landmarks[index] = landmarks[index] * 2.0f - 1.0f;
            landmarks[index + 1] = 1.0f - landmarks[index + 1] * 2.0f;
        }
        glUseProgram(landmarkProgram_);
        glBindVertexArray(landmarkVertexArray_);
        glBindBuffer(GL_ARRAY_BUFFER, landmarkVertexBuffer_);
        glBufferSubData(GL_ARRAY_BUFFER, 0, landmarks.size() * sizeof(float), landmarks.data());
        glDrawArrays(GL_POINTS, 0, 68);
        static bool logged = false;
        if (!logged) {
            OH_LOG_INFO(LOG_APP, "Drew 68 landmark points, glError=%{public}u", glGetError());
            logged = true;
        }
    }
    if (++frameCounter_ % FACE_CAPTURE_INTERVAL == 0) {
        CaptureFaceInput(transform);
    }
    {
        BeautyTrace::Scope swap("Native/Render/EglSwapBuffers");
        eglSwapBuffers(eglDisplay_, eglSurface_);
    }
}

void BeautyRenderer::UploadBeautyLuts()
{
    std::vector<uint8_t> gray;
    std::vector<uint8_t> origin;
    std::vector<uint8_t> skin;
    std::vector<uint8_t> light;
    {
        std::lock_guard<std::mutex> lock(lutMutex_);
        if (!lutUploadPending_) return;
        gray.swap(grayLutData_);
        origin.swap(originLutData_);
        skin.swap(skinLutData_);
        light.swap(lightLutData_);
        lutUploadPending_ = false;
    }
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glBindTexture(GL_TEXTURE_2D, grayLutTexture_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 256, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, gray.data());
    glBindTexture(GL_TEXTURE_2D, originLutTexture_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, 64, 64, 0, GL_RGB, GL_UNSIGNED_BYTE, origin.data());
    glBindTexture(GL_TEXTURE_2D, skinLutTexture_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, 64, 64, 0, GL_RGB, GL_UNSIGNED_BYTE, skin.data());
    glBindTexture(GL_TEXTURE_2D, lightLutTexture_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, 512, 512, 0, GL_RGB, GL_UNSIGNED_BYTE, light.data());
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    beautyLutsReady_ = glGetError() == GL_NO_ERROR;
    OH_LOG_INFO(LOG_APP, "GPUPixel beauty LUT upload %{public}s", beautyLutsReady_ ? "ready" : "failed");
}

void BeautyRenderer::RenderSmoothTexture(const float *transform)
{
    glBindFramebuffer(GL_FRAMEBUFFER, beautySourceFramebuffer_);
    glViewport(0, 0, beautyWidth_, beautyHeight_);
    glUseProgram(faceProgram_);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_EXTERNAL_OES, externalTexture_);
    glUniform1i(glGetUniformLocation(faceProgram_, "uCamera"), 0);
    glUniformMatrix4fv(faceTransformLocation_, 1, GL_FALSE, transform);
    glUniform2f(faceScaleLocation_, 1.0f, 1.0f);
    glBindVertexArray(vertexArray_);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    glBindFramebuffer(GL_FRAMEBUFFER, beautyBlurFramebuffer_);
    glUseProgram(blurProgram_);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, beautySourceTexture_);
    glUniform1i(glGetUniformLocation(blurProgram_, "uTexture"), 0);
    glUniform2f(glGetUniformLocation(blurProgram_, "uDirection"),
        1.0f / static_cast<float>(beautyWidth_), 0.0f);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    glBindFramebuffer(GL_FRAMEBUFFER, beautySourceFramebuffer_);
    glBindTexture(GL_TEXTURE_2D, beautyBlurTexture_);
    glUniform2f(glGetUniformLocation(blurProgram_, "uDirection"),
        0.0f, 1.0f / static_cast<float>(beautyHeight_));
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void BeautyRenderer::UpdateSkinMask(const std::vector<float> &landmarks)
{
    std::vector<uint8_t> mask(SKIN_MASK_WIDTH * SKIN_MASK_HEIGHT, 0);
    size_t coveredPixels = 0;
    if (landmarks.size() == 136) {
        float faceLeft = 1.0f;
        float faceRight = 0.0f;
        float faceTop = 1.0f;
        float faceBottom = 0.0f;
        for (int index = 0; index <= 16; ++index) {
            faceLeft = std::min(faceLeft, landmarks[index * 2]);
            faceRight = std::max(faceRight, landmarks[index * 2]);
            faceBottom = std::max(faceBottom, landmarks[index * 2 + 1]);
        }
        for (int index = 17; index <= 26; ++index) {
            faceTop = std::min(faceTop, landmarks[index * 2 + 1]);
        }
        const float faceWidth = std::max(faceRight - faceLeft, 0.01f);
        const float faceHeight = std::max(faceBottom - faceTop, 0.01f);
        const float foreheadTop = std::clamp(faceTop - faceHeight * 0.42f, 0.0f, 1.0f);
        std::vector<MaskPoint> facePolygon;
        facePolygon.reserve(22);
        for (int index = 0; index <= 16; ++index) {
            facePolygon.push_back({ landmarks[index * 2], landmarks[index * 2 + 1] });
        }
        facePolygon.push_back({ std::clamp(faceRight + faceWidth * 0.02f, 0.0f, 1.0f),
            std::clamp(faceTop - faceHeight * 0.12f, 0.0f, 1.0f) });
        facePolygon.push_back({ std::clamp(landmarks[26 * 2] + faceWidth * 0.08f, 0.0f, 1.0f), foreheadTop });
        facePolygon.push_back({ (landmarks[21 * 2] + landmarks[22 * 2]) * 0.5f,
            std::clamp(foreheadTop - faceHeight * 0.04f, 0.0f, 1.0f) });
        facePolygon.push_back({ std::clamp(landmarks[17 * 2] - faceWidth * 0.08f, 0.0f, 1.0f), foreheadTop });
        facePolygon.push_back({ std::clamp(faceLeft - faceWidth * 0.02f, 0.0f, 1.0f),
            std::clamp(faceTop - faceHeight * 0.12f, 0.0f, 1.0f) });

        const MaskBounds leftEye = BoundsFor(landmarks, 36, 41);
        const MaskBounds rightEye = BoundsFor(landmarks, 42, 47);
        const MaskBounds mouth = BoundsFor(landmarks, 48, 59);

        for (int row = 0; row < SKIN_MASK_HEIGHT; ++row) {
            const float y = (static_cast<float>(row) + 0.5f) / SKIN_MASK_HEIGHT;
            for (int column = 0; column < SKIN_MASK_WIDTH; ++column) {
                const float x = (static_cast<float>(column) + 0.5f) / SKIN_MASK_WIDTH;
                if (!PointInPolygon(facePolygon, x, y)) continue;
                float edgeDistance = 1.0f;
                for (size_t index = 0; index < facePolygon.size(); ++index) {
                    edgeDistance = std::min(edgeDistance, DistanceToSegment(x, y, facePolygon[index],
                        facePolygon[(index + 1) % facePolygon.size()]));
                }
                float weight = SmoothStep(0.0f, 0.040f, edgeDistance);
                weight *= OutsideFeatureEllipse(x, y, leftEye, faceWidth * 0.010f, faceHeight * 0.010f);
                weight *= OutsideFeatureEllipse(x, y, rightEye, faceWidth * 0.010f, faceHeight * 0.010f);
                weight *= OutsideFeatureEllipse(x, y, mouth, faceWidth * 0.025f, faceHeight * 0.025f);
                const uint8_t maskValue = static_cast<uint8_t>(std::round(weight * 255.0f));
                mask[row * SKIN_MASK_WIDTH + column] = maskValue;
                if (maskValue > 32) ++coveredPixels;
            }
        }
    }
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, skinMaskTexture_);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, SKIN_MASK_WIDTH, SKIN_MASK_HEIGHT,
        GL_RED, GL_UNSIGNED_BYTE, mask.data());
    glActiveTexture(GL_TEXTURE0);
    static bool logged = false;
    if (!logged && landmarks.size() == 136) {
        const float coverage = static_cast<float>(coveredPixels) /
            static_cast<float>(SKIN_MASK_WIDTH * SKIN_MASK_HEIGHT) * 100.0f;
        OH_LOG_INFO(LOG_APP, "Skin mask uploaded, covered=%{public}.2f%%", coverage);
        logged = true;
    }
}

void BeautyRenderer::CaptureFaceInput(const float *transform)
{
    BeautyTrace::Scope trace("Native/Input/CaptureFaceFrame");
    {
        std::lock_guard<std::mutex> lock(faceInputMutex_);
        if (faceInputReady_) return;
    }

    const float outputAspect = static_cast<float>(std::max(outputWidth_, 1)) /
        static_cast<float>(std::max(outputHeight_, 1));
    constexpr float inputAspect = static_cast<float>(FACE_INPUT_WIDTH) / static_cast<float>(FACE_INPUT_HEIGHT);
    const float xScale = std::min(1.0f, outputAspect / inputAspect);
    const float yScale = std::min(1.0f, inputAspect / outputAspect);

    {
        BeautyTrace::Scope draw("Native/Input/AnalysisFboDraw");
        glBindFramebuffer(GL_FRAMEBUFFER, faceFramebuffer_);
        glViewport(0, 0, FACE_INPUT_WIDTH, FACE_INPUT_HEIGHT);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(faceProgram_);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_EXTERNAL_OES, externalTexture_);
        glUniform1i(glGetUniformLocation(faceProgram_, "uCamera"), 0);
        glUniformMatrix4fv(faceTransformLocation_, 1, GL_FALSE, transform);
        glUniform2f(faceScaleLocation_, xScale, yScale);
        glBindVertexArray(vertexArray_);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    }

    std::vector<uint8_t> rgba(FACE_INPUT_WIDTH * FACE_INPUT_HEIGHT * 4);
    {
        BeautyTrace::Scope readback("Native/Input/GlReadPixels");
        glReadPixels(0, 0, FACE_INPUT_WIDTH, FACE_INPUT_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    const size_t planeSize = FACE_INPUT_WIDTH * FACE_INPUT_HEIGHT;
    std::vector<float> input(planeSize * 3);
    {
        BeautyTrace::Scope convert("Native/Input/RgbaToNchw");
        for (int y = 0; y < FACE_INPUT_HEIGHT; ++y) {
            const int sourceY = FACE_INPUT_HEIGHT - 1 - y;
            for (int x = 0; x < FACE_INPUT_WIDTH; ++x) {
                const size_t source = (sourceY * FACE_INPUT_WIDTH + x) * 4;
                const size_t target = y * FACE_INPUT_WIDTH + x;
                input[target] = (static_cast<float>(rgba[source]) - 127.0f) / 128.0f;
                input[planeSize + target] = (static_cast<float>(rgba[source + 1]) - 127.0f) / 128.0f;
                input[planeSize * 2 + target] = (static_cast<float>(rgba[source + 2]) - 127.0f) / 128.0f;
            }
        }
    }

    BeautyTrace::Scope publish("Native/Input/PublishFaceFrame");
    std::lock_guard<std::mutex> lock(faceInputMutex_);
    if (!faceInputReady_) {
        faceInput_ = std::move(input);
        faceInputXScale_ = xScale;
        faceInputYScale_ = yScale;
        faceInputReady_ = true;
        BeautyTrace::Count("State/FaceInputReady", 1);
    }
}

void BeautyRenderer::DestroyGl()
{
    if (nativeImage_ != nullptr) {
        OH_NativeImage_UnsetOnFrameAvailableListener(nativeImage_);
        OH_NativeImage_Destroy(&nativeImage_);
    }
    if (eglDisplay_ != EGL_NO_DISPLAY && eglContext_ != EGL_NO_CONTEXT) {
        eglMakeCurrent(eglDisplay_, eglSurface_, eglSurface_, eglContext_);
        if (vertexBuffer_ != 0) glDeleteBuffers(1, &vertexBuffer_);
        if (vertexArray_ != 0) glDeleteVertexArrays(1, &vertexArray_);
        if (landmarkVertexBuffer_ != 0) glDeleteBuffers(1, &landmarkVertexBuffer_);
        if (landmarkVertexArray_ != 0) glDeleteVertexArrays(1, &landmarkVertexArray_);
        if (faceMeshVertexBuffer_ != 0) glDeleteBuffers(1, &faceMeshVertexBuffer_);
        if (faceMeshIndexBuffer_ != 0) glDeleteBuffers(1, &faceMeshIndexBuffer_);
        if (faceMeshVertexArray_ != 0) glDeleteVertexArrays(1, &faceMeshVertexArray_);
        if (faceFramebuffer_ != 0) glDeleteFramebuffers(1, &faceFramebuffer_);
        if (beautySourceFramebuffer_ != 0) glDeleteFramebuffers(1, &beautySourceFramebuffer_);
        if (beautyBlurFramebuffer_ != 0) glDeleteFramebuffers(1, &beautyBlurFramebuffer_);
        if (faceTexture_ != 0) glDeleteTextures(1, &faceTexture_);
        if (skinMaskTexture_ != 0) glDeleteTextures(1, &skinMaskTexture_);
        if (beautySourceTexture_ != 0) glDeleteTextures(1, &beautySourceTexture_);
        if (beautyBlurTexture_ != 0) glDeleteTextures(1, &beautyBlurTexture_);
        if (grayLutTexture_ != 0) glDeleteTextures(1, &grayLutTexture_);
        if (originLutTexture_ != 0) glDeleteTextures(1, &originLutTexture_);
        if (skinLutTexture_ != 0) glDeleteTextures(1, &skinLutTexture_);
        if (lightLutTexture_ != 0) glDeleteTextures(1, &lightLutTexture_);
        if (faceMeshTexture_ != 0) glDeleteTextures(1, &faceMeshTexture_);
        if (faceProgram_ != 0) glDeleteProgram(faceProgram_);
        if (landmarkProgram_ != 0) glDeleteProgram(landmarkProgram_);
        if (faceMeshProgram_ != 0) glDeleteProgram(faceMeshProgram_);
        if (blurProgram_ != 0) glDeleteProgram(blurProgram_);
        if (program_ != 0) glDeleteProgram(program_);
        if (externalTexture_ != 0) glDeleteTextures(1, &externalTexture_);
        eglMakeCurrent(eglDisplay_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    }
    if (eglDisplay_ != EGL_NO_DISPLAY && eglSurface_ != EGL_NO_SURFACE) {
        eglDestroySurface(eglDisplay_, eglSurface_);
    }
    if (eglDisplay_ != EGL_NO_DISPLAY && eglContext_ != EGL_NO_CONTEXT) {
        eglDestroyContext(eglDisplay_, eglContext_);
    }
    if (eglDisplay_ != EGL_NO_DISPLAY) eglTerminate(eglDisplay_);
    if (outputWindow_ != nullptr) OH_NativeWindow_DestroyNativeWindow(outputWindow_);
    outputWindow_ = nullptr;
    eglDisplay_ = EGL_NO_DISPLAY;
    eglContext_ = EGL_NO_CONTEXT;
    eglSurface_ = EGL_NO_SURFACE;
    externalTexture_ = 0;
    program_ = 0;
    faceProgram_ = 0;
    landmarkProgram_ = 0;
    faceMeshProgram_ = 0;
    blurProgram_ = 0;
    faceFramebuffer_ = 0;
    beautySourceFramebuffer_ = 0;
    beautyBlurFramebuffer_ = 0;
    faceTexture_ = 0;
    skinMaskTexture_ = 0;
    beautySourceTexture_ = 0;
    beautyBlurTexture_ = 0;
    grayLutTexture_ = 0;
    originLutTexture_ = 0;
    skinLutTexture_ = 0;
    lightLutTexture_ = 0;
    faceMeshTexture_ = 0;
    beautyLutsReady_ = false;
    vertexArray_ = 0;
    vertexBuffer_ = 0;
    landmarkVertexArray_ = 0;
    landmarkVertexBuffer_ = 0;
    faceMeshVertexArray_ = 0;
    faceMeshVertexBuffer_ = 0;
    faceMeshIndexBuffer_ = 0;
}

GLuint BeautyRenderer::CompileShader(GLenum type, const char *source)
{
    const GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    GLint compiled = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (compiled == GL_FALSE) {
        char log[1024] = {0};
        glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
        OH_LOG_ERROR(LOG_APP, "Shader compilation failed: %{public}s", log);
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

GLuint BeautyRenderer::CreateProgram()
{
    const GLuint vertexShader = CompileShader(GL_VERTEX_SHADER, VERTEX_SHADER);
    const GLuint fragmentShader = CompileShader(GL_FRAGMENT_SHADER, FRAGMENT_SHADER);
    if (vertexShader == 0 || fragmentShader == 0) {
        if (vertexShader != 0) glDeleteShader(vertexShader);
        if (fragmentShader != 0) glDeleteShader(fragmentShader);
        return 0;
    }
    const GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    GLint linked = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (linked == GL_FALSE) {
        char log[1024] = {0};
        glGetProgramInfoLog(program, sizeof(log), nullptr, log);
        OH_LOG_ERROR(LOG_APP, "Shader link failed: %{public}s", log);
        glDeleteProgram(program);
        return 0;
    }
    return program;
}

GLuint BeautyRenderer::CreateFaceProgram()
{
    const GLuint vertexShader = CompileShader(GL_VERTEX_SHADER, FACE_VERTEX_SHADER);
    const GLuint fragmentShader = CompileShader(GL_FRAGMENT_SHADER, FACE_FRAGMENT_SHADER);
    if (vertexShader == 0 || fragmentShader == 0) {
        if (vertexShader != 0) glDeleteShader(vertexShader);
        if (fragmentShader != 0) glDeleteShader(fragmentShader);
        return 0;
    }
    const GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    GLint linked = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (linked == GL_FALSE) {
        char log[1024] = {0};
        glGetProgramInfoLog(program, sizeof(log), nullptr, log);
        OH_LOG_ERROR(LOG_APP, "Face shader link failed: %{public}s", log);
        glDeleteProgram(program);
        return 0;
    }
    return program;
}

GLuint BeautyRenderer::CreateLandmarkProgram()
{
    const GLuint vertexShader = CompileShader(GL_VERTEX_SHADER, LANDMARK_VERTEX_SHADER);
    const GLuint fragmentShader = CompileShader(GL_FRAGMENT_SHADER, LANDMARK_FRAGMENT_SHADER);
    if (vertexShader == 0 || fragmentShader == 0) {
        if (vertexShader != 0) glDeleteShader(vertexShader);
        if (fragmentShader != 0) glDeleteShader(fragmentShader);
        return 0;
    }
    const GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    GLint linked = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (linked == GL_FALSE) {
        char log[1024] = {0};
        glGetProgramInfoLog(program, sizeof(log), nullptr, log);
        OH_LOG_ERROR(LOG_APP, "Landmark shader link failed: %{public}s", log);
        glDeleteProgram(program);
        return 0;
    }
    return program;
}

GLuint BeautyRenderer::CreateFaceMeshProgram()
{
    const GLuint vertexShader = CompileShader(GL_VERTEX_SHADER, FACE_MESH_VERTEX_SHADER);
    const GLuint fragmentShader = CompileShader(GL_FRAGMENT_SHADER, FACE_MESH_FRAGMENT_SHADER);
    if (vertexShader == 0 || fragmentShader == 0) {
        if (vertexShader != 0) glDeleteShader(vertexShader);
        if (fragmentShader != 0) glDeleteShader(fragmentShader);
        return 0;
    }
    const GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    GLint linked = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (linked == GL_FALSE) {
        char log[1024] = {0};
        glGetProgramInfoLog(program, sizeof(log), nullptr, log);
        OH_LOG_ERROR(LOG_APP, "Face mesh shader link failed: %{public}s", log);
        glDeleteProgram(program);
        return 0;
    }
    return program;
}

GLuint BeautyRenderer::CreateBlurProgram()
{
    const GLuint vertexShader = CompileShader(GL_VERTEX_SHADER, BLUR_VERTEX_SHADER);
    const GLuint fragmentShader = CompileShader(GL_FRAGMENT_SHADER, BLUR_FRAGMENT_SHADER);
    if (vertexShader == 0 || fragmentShader == 0) {
        if (vertexShader != 0) glDeleteShader(vertexShader);
        if (fragmentShader != 0) glDeleteShader(fragmentShader);
        return 0;
    }
    const GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    GLint linked = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (linked == GL_FALSE) {
        char log[1024] = {0};
        glGetProgramInfoLog(program, sizeof(log), nullptr, log);
        OH_LOG_ERROR(LOG_APP, "Blur shader link failed: %{public}s", log);
        glDeleteProgram(program);
        return 0;
    }
    return program;
}
