ARG BASE_IMAGE
FROM ${BASE_IMAGE} as base

ARG ENV
ARG BUILD_CONTEXT
ARG CUDA_VER
ARG TRT_MAJOR_VER
ARG TRT_VER
ARG USE_NIGHTLY

ENV BUILD_CONTEXT=${BUILD_CONTEXT}
ENV CUDA_VER=${CUDA_VER}
ENV TRT_MAJOR_VER=${TRT_MAJOR_VER}
ENV TRT_VER=${TRT_VER}
ENV ENV=${ENV}
ARG USE_NIGHTLY=${USE_NIGHTLY}

SHELL ["/bin/bash", "-c"]

WORKDIR /tmp

COPY --from=common /requirements.${BUILD_CONTEXT}.* .

# 1. apt installs
COPY --from=common /install_apt_deps.sh .
RUN if [ -f install_apt_deps.sh ]; then bash install_apt_deps.sh; fi

# 2. pip installs
COPY --from=common /install_pip_deps.sh .
RUN if [ -f install_pip_deps.sh ]; then bash install_pip_deps.sh; fi

# 3. Any from-source build required.
COPY --from=common /install_misc_deps.sh .
RUN if [ -f install_misc_deps.sh ]; then bash install_misc_deps.sh; fi

# 4. install TensorRT for dev mode
COPY install_tensorrt.sh .
RUN if [ -f install_tensorrt.sh ]; then bash install_tensorrt.sh; fi

# 5. (hacky) install loadgen for SDXL run in automated L0 
COPY install_mlperf_loadgen.sh .
RUN if [ -f install_mlperf_loadgen.sh ]; then bash install_mlperf_loadgen.sh; fi

# 6. switch between compiler for building faster transformer repo
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 9
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 9
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 11
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 11

WORKDIR /work
