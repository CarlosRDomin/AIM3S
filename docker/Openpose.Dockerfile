FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04
# 10.1-cudnn7-devel-ubuntu18.04 (10.1/devel/cudnn7/Dockerfile)

# ------------------------------------------------------------------
# openpose
# ------------------------------------------------------------------

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=10.0"
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install git -yy && \
    git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git \ 
    /software/openpose && \
    apt-get install sudo libopencv-dev cmake -yy && \
    mkdir -p /software/openpose/build && cd /software/openpose/build && \
    chmod +x ../scripts/ubuntu/install_deps.sh && \
    ../scripts/ubuntu/install_deps.sh && \
    cmake -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D BUILD_PYTHON=ON \
    -D BUILD_CAFFE=ON \
    -D BUILD_EXAMPLES=ON \
    -D GPU_MODE=CUDA \
    -D CMAKE_BUILD_TYPE=Release \
    -D DOWNLOAD_BODY_25_MODEL=ON \
    -D DOWNLOAD_HAND_MODEL=OFF \
    -D DOWNLOAD_FACE_MODEL=OFF .. && \
    make -j"$(nproc)" && \
    make install