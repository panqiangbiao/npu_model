"""Generate native mesh tables and an RGBA mask texture from MediaPipe data."""

from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "npu_face_mesh"
OBJ_PATH = MODEL_DIR / "canonical_face_model.obj"
HEADER_PATH = ROOT / "entry" / "src" / "main" / "cpp" / "face_mesh_data.h"
TEXTURE_PATH = ROOT / "entry" / "src" / "main" / "resources" / "rawfile" / "spider_face_mesh.rgba"
CYBER_TEXTURE_SOURCE = MODEL_DIR / "cyber_hero_face_mesh_v2.png"


def parse_obj() -> tuple[list[tuple[float, float]], list[int]]:
    texture_coordinates: list[tuple[float, float]] = []
    faces: list[list[tuple[int, int]]] = []
    for line in OBJ_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("vt "):
            _, u, v = line.split()
            texture_coordinates.append((float(u), float(v)))
        elif line.startswith("f "):
            face: list[tuple[int, int]] = []
            for token in line.split()[1:]:
                vertex, texture = token.split("/")[:2]
                face.append((int(vertex) - 1, int(texture) - 1))
            faces.append(face)

    vertex_uv: list[tuple[float, float] | None] = [None] * 468
    indices: list[int] = []
    for face in faces:
        for vertex, texture in face:
            uv = texture_coordinates[texture]
            if vertex_uv[vertex] is not None and vertex_uv[vertex] != uv:
                raise ValueError(f"vertex {vertex} has multiple UV coordinates")
            vertex_uv[vertex] = uv
            indices.append(vertex)
    if any(uv is None for uv in vertex_uv):
        raise ValueError("canonical model does not map every landmark")
    return [uv for uv in vertex_uv if uv is not None], indices


def write_header(uvs: list[tuple[float, float]], indices: list[int]) -> None:
    uv_values = [value for uv in uvs for value in uv]

    def rows(values: list[float | int], width: int, formatter) -> str:
        return "\n".join(
            "    " + ", ".join(formatter(value) for value in values[start:start + width]) + ","
            for start in range(0, len(values), width)
        )

    header = f"""#ifndef HARMONY_FACE_MESH_DATA_H
#define HARMONY_FACE_MESH_DATA_H

#include <array>
#include <cstdint>

// Generated from MediaPipe's Apache-2.0 canonical_face_model.obj.
inline constexpr std::array<float, {len(uv_values)}> FACE_MESH_UVS = {{
{rows(uv_values, 8, lambda value: f'{value:.7f}f')}
}};

inline constexpr std::array<uint16_t, {len(indices)}> FACE_MESH_INDICES = {{
{rows(indices, 18, str)}
}};

#endif
"""
    HEADER_PATH.write_text(header, encoding="ascii")


def write_texture() -> None:
    size = 1024
    if not CYBER_TEXTURE_SOURCE.exists():
        raise FileNotFoundError(f"missing mask texture: {CYBER_TEXTURE_SOURCE}")
    image = Image.open(CYBER_TEXTURE_SOURCE).convert("RGBA")
    image = image.resize((size, size), Image.Resampling.LANCZOS)

    TEXTURE_PATH.write_bytes(image.tobytes())
    image.save(MODEL_DIR / "cyber_hero_face_mesh_preview.png")


def main() -> None:
    uvs, indices = parse_obj()
    write_header(uvs, indices)
    write_texture()
    print(f"generated {len(uvs)} vertices, {len(indices) // 3} triangles")
    print(f"texture={TEXTURE_PATH} bytes={TEXTURE_PATH.stat().st_size}")


if __name__ == "__main__":
    main()
