cmake_minimum_required(VERSION 3.10)
project(yolo26depth-rknn-cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# RKNN
find_library(RKNN_RT NAMES rknnrt librknnrt.so
    PATHS /usr/lib /usr/local/lib /lib
    NO_DEFAULT_PATH
)
if(NOT RKNN_RT)
    set(RKNN_RT -lrknnrt)
endif()

# OpenCV via pkg-config
find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENCV REQUIRED opencv4)

# RGA (RK3588 hardware accelerator)
pkg_check_modules(RGA librga QUIET)
if(RGA_FOUND)
    message(STATUS "RGA found: ${RGA_LIBRARIES}")
    add_definitions(-DHAS_RGA)
else()
    message(WARNING "librga not found, RGA benchmarks disabled")
endif()

# Include paths for RKNN API
find_path(RKNN_API_DIR rknn_api.h
    PATHS /usr/include /usr/local/include /home/cat/rknn_model_zoo/3rdparty/rknpu2/include
    PATH_SUFFIXES rknn
)
if(NOT RKNN_API_DIR)
    set(RKNN_API_DIR /home/cat/rknn_model_zoo/3rdparty/rknpu2/include)
endif()

# Sources
set(COMMON_SOURCES
    src/rknn.cpp
    src/preprocess.cpp
    src/postprocess.cpp
    src/pointcloud.cpp
)

set(COMMON_HEADERS
    src/rknn.hpp
    src/preprocess.hpp
    src/postprocess.hpp
    src/pointcloud.hpp
)

# Main executable
add_executable(yolo26depth-rknn
    src/main.cpp
    ${COMMON_SOURCES}
    ${COMMON_HEADERS}
)
target_link_libraries(yolo26depth-rknn PRIVATE
    ${RKNN_RT}
    ${OPENCV_LIBRARIES}
)
target_include_directories(yolo26depth-rknn PRIVATE
    ${RKNN_API_DIR}
    ${OPENCV_INCLUDE_DIRS}
)

# Benchmark pool
add_executable(benchmark_pool
    src/benchmark_pool.cpp
    ${COMMON_SOURCES}
    ${COMMON_HEADERS}
)
target_link_libraries(benchmark_pool PRIVATE
    ${RKNN_RT}
    ${OPENCV_LIBRARIES}
)
target_include_directories(benchmark_pool PRIVATE
    ${RKNN_API_DIR}
    ${OPENCV_INCLUDE_DIRS}
)

# RGA benchmarks (only if librga is available)
if(RGA_FOUND)
    # Preprocess benchmark
    add_executable(bench_rga
        src/bench_rga.cpp
        src/rga_preprocess.cpp
        src/rga_preprocess.hpp
    )
    target_link_libraries(bench_rga PRIVATE
        ${RGA_LIBRARIES}
        ${OPENCV_LIBRARIES}
    )
    target_include_directories(bench_rga PRIVATE
        ${RGA_INCLUDE_DIRS}
        ${OPENCV_INCLUDE_DIRS}
    )

    # Full pipeline benchmark (pre + infer + post)
    add_executable(bench_full_pool_rga
        src/bench_full_pool_rga.cpp
        src/rga_preprocess.cpp
        src/rga_preprocess.hpp
        ${COMMON_SOURCES}
        ${COMMON_HEADERS}
    )
    target_link_libraries(bench_full_pool_rga PRIVATE
        ${RKNN_RT}
        ${OPENCV_LIBRARIES}
        ${RGA_LIBRARIES}
    )
    target_include_directories(bench_full_pool_rga PRIVATE
        ${RKNN_API_DIR}
        ${OPENCV_INCLUDE_DIRS}
        ${RGA_INCLUDE_DIRS}
    )
endif()
