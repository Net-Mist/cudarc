FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu24.04

# ENV PKG_CONFIG_PATH=/opt/conda/lib/pkgconfig/
ENV CUDA_COMPUTE_CAP=89
ENV DEBIAN_FRONTEND=noninteractive

# ENV LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
# ENV LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:$LIBRARY_PATH


# Install Rust and other useful tools
# libclang-dev is for bindgen
# clang is required in the build.rs script
RUN apt-get update && \
    apt upgrade -y && \
    apt-get install -y curl unzip pkg-config git libclang-dev clang && \
    curl https://sh.rustup.rs -sSf | sh -s -- -y && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# # install nsys-cli
# RUN apt update && \
#     apt upgrade -y && \
#     apt install -y --no-install-recommends gnupg && \
#     echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64 /" | tee /etc/apt/sources.list.d/nvidia-devtools.list  && \
#     apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub && \
#     apt update && \
#     apt install -y nsight-systems-cli && \
#     apt clean && \
#     rm -rf /var/lib/apt/lists/*

# # install nsys-ui
# RUN apt update && \
#     apt upgrade -y && \
#     apt install -y --no-install-recommends gnupg libglx-dev libsm6 libxkbfile1 nsight-systems && \
#     apt clean && \
#     rm -rf /var/lib/apt/lists/*

    
RUN /root/.cargo/bin/cargo install bindgen-cli --version 0.69.4
