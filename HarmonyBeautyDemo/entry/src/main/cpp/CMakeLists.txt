cmake_minimum_required(VERSION 3.5.0)
project(beauty_pipeline)

set(CMAKE_CXX_STANDARD 17)

add_library(beauty_pipeline SHARED napi_init.cpp beauty_renderer.cpp gpu_face_inference.cpp)

target_include_directories(beauty_pipeline PRIVATE third_party/mnn/include)
target_link_directories(beauty_pipeline PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../libs/${OHOS_ARCH})

target_link_libraries(beauty_pipeline PUBLIC
    libace_napi.z.so
    libhilog_ndk.z.so
    libnative_window.so
    libnative_image.so
    libneural_network_core.so
    libneural_network_runtime.so
    libhitrace_ndk.z.so
    libEGL.so
    libGLESv3.so
    hiai
    hiai_ir
    hiai_ir_build
    MNN_CL
    MNN)
