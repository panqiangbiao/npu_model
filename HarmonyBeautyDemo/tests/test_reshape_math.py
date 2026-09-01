import math


def metric(point, aspect):
    return point[0], point[1] / aspect


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def curve_warp(coordinate, origin, target, amount, aspect):
    radius = max(distance(metric(target, aspect), metric(origin, aspect)), 0.001)
    falloff = max(0.0, min(1.0,
        1.0 - distance(metric(coordinate, aspect), metric(origin, aspect)) / radius))
    return (
        coordinate[0] - (target[0] - origin[0]) * amount * falloff * falloff,
        coordinate[1] - (target[1] - origin[1]) * amount * falloff * falloff,
    )


def enlarge_eye(coordinate, center, radius, amount, aspect):
    normalized = distance(metric(coordinate, aspect), metric(center, aspect)) / radius
    if normalized >= 1.0:
        return coordinate
    scale = max(0.72, min(1.0, 1.0 - (1.0 - normalized * normalized) * amount))
    return (
        center[0] + (coordinate[0] - center[0]) * scale,
        center[1] + (coordinate[1] - center[1]) * scale,
    )


def smoothing_factor(motion):
    return max(0.24, min(0.60, 0.24 + motion * 4.0))


def test_slim_face_samples_outside_the_jaw():
    jaw = (0.25, 0.62)
    target = (0.48, 0.58)
    sampled = curve_warp(jaw, jaw, target, 0.05, 0.75)
    assert sampled[0] < jaw[0]


def test_big_eye_samples_toward_the_eye_center():
    center = (0.40, 0.42)
    coordinate = (0.44, 0.42)
    sampled = enlarge_eye(coordinate, center, 0.12, 0.10, 0.75)
    assert center[0] < sampled[0] < coordinate[0]


def test_warps_are_identity_outside_their_radius():
    coordinate = (0.90, 0.90)
    assert curve_warp(coordinate, (0.25, 0.62), (0.48, 0.58), 0.05, 0.75) == coordinate
    assert enlarge_eye(coordinate, (0.40, 0.42), 0.12, 0.10, 0.75) == coordinate


def test_landmark_smoothing_is_stable_but_catches_large_motion():
    assert smoothing_factor(0.0) == 0.24
    assert smoothing_factor(0.02) < smoothing_factor(0.08)
    assert smoothing_factor(0.20) == 0.60


def test_lost_face_strength_fades_instead_of_snapping():
    blend = 1.0
    blend = max(0.0, blend - 0.08)
    assert 0.0 < blend < 1.0
    for _ in range(12):
        blend = max(0.0, blend - 0.08)
    assert blend == 0.0


if __name__ == "__main__":
    test_slim_face_samples_outside_the_jaw()
    test_big_eye_samples_toward_the_eye_center()
    test_warps_are_identity_outside_their_radius()
    test_landmark_smoothing_is_stable_but_catches_large_motion()
    test_lost_face_strength_fades_instead_of_snapping()
    print("reshape math tests passed")
