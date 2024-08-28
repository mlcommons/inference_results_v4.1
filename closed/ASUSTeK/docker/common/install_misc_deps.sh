#!/bin/bash

set -e

install_gflags(){
    local VERSION=$1

    cd /tmp
    # -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=ON -DBUILD_gflags_LIB=ON .. \
    git clone -b v${VERSION} https://github.com/gflags/gflags.git
    cd gflags
    mkdir build && cd build
    cmake -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=ON -DBUILD_gflags_LIB=ON ..
    make -j
    make install
    cd /tmp && rm -rf gflags
}

install_glog(){
    local VERSION=$1

    cd /tmp
    git clone -b v${VERSION} https://github.com/google/glog.git
    cd glog
    cmake -H. -Bbuild -G "Unix Makefiles" -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=ON
    cmake --build build
    cmake --build build --target install
    cd /tmp && rm -rf glog

}

install_CUB(){
    local ARCH=$1

    # Install CUB, needed by NMS OPT plugin
    cd /tmp
    wget https://github.com/NVlabs/cub/archive/1.8.0.zip -O cub-1.8.0.zip
    unzip cub-1.8.0.zip
    mv cub-1.8.0/cub /usr/include/${ARCH}-linux-gnu/
    rm -rf cub-1.8.0.zip cub-1.8.0
}

install_nlohmann_json_x86_64(){
    # Install nlohmann/json for LLM config parsing
    cd /tmp
    git clone -b v3.11.2 https://github.com/nlohmann/json.git
    cp -r json/single_include/nlohmann /usr/include/x86_64-linux-gnu/
    rm -rf json
}

install_nlohmann_json_aarch64(){
    apt install -y nlohmann-json3-dev
}

install_nvrtc_dev(){
    local ARCH=$1
    # remove the default nvrtc package, do a fresh install
    apt-get remove --purge -y --allow-change-held-packages cuda-nvrtc-dev*

    cd /tmp
    curl -fsSLO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/${ARCH}/cuda-keyring_1.0-1_all.deb
    dpkg -i cuda-keyring_1.0-1_all.deb
    apt update
    apt-get install -y --no-install-recommends cuda-nvrtc-dev-12-4=12.4.127-1

    # Create symlink for TRTLLM
    sudo ln -s /usr/local/cuda-12.4/lib64/libnvrtc_static.a /usr/local/libnvrtc_static.a
    sudo ln -s /usr/local/cuda-12.4/lib64/libnvrtc-builtins_static.a /usr/local/libnvrtc-builtins_static.a
}

install_boost_1_78(){
    apt purge libboost-all-dev \
      && apt autoremove -y libboost1.74-dev \
      && sudo rm -rf /usr/lib/libboost_* /usr/include/boost \
      && wget -O /tmp/boost.tar.gz \
          https://archives.boost.io/release/1.80.0/source/boost_1_80_0.tar.gz \
      && (cd /tmp && tar xzf boost.tar.gz) \
      && mv /tmp/boost_1_80_0/boost /usr/include/boost
}

install_cupva_orin(){
    # install cupva
    wget -nv https://repo.download.nvidia.com/jetson/common/pool/main/c/cupva-2.0-l4t/cupva-2.0-l4t_2.0.0_arm64.deb
    dpkg -i cupva-2.0-l4t_2.0.0_arm64.deb
    apt update
    apt install cupva-2.0-l4t

    wget -nv https://developer.nvidia.com/downloads/embedded/l4t/cupva-algos-gen2-2.0.0-cupva_algo_dlops.deb
    dpkg --add-architecture amd64
    dpkg -i cupva-algos-gen2-2.0.0-cupva_algo_dlops.deb
}

install_mxeval_deps() {
  # Needed by moe submission_checker
  # Ref: https://github.com/mlcommons/inference/blob/master/language/mixtral-8x7b/Dockerfile.eval

  # php
  # NOTE(vir): using gettext instead of php-gettext
  add-apt-repository -y ppa:ondrej/php
  apt update
  apt install -y --no-install-recommends php8.0 gettext php-bcmath php-cgi php-common php-curl php-fpm \
      php-gd php-intl php-json php-mbstring php-pear php-xml php-zip php8.0-common php8.0-mysql

  # java
  apt install -y --no-install-recommends openjdk-8-jdk

  # golang
  cd /usr/local
  local GO_TAR=go1.19.1.linux-amd64.tar.gz
  sudo wget https://go.dev/dl/$GO_TAR
  sudo tar -xzvf $GO_TAR
  sudo rm $GO_TAR
  echo 'export PATH="$PATH:/usr/local/go/bin"' >> $HOME/.bashrc

  # swift
  cd /usr/local
  local UBUNTU_MAJOR_VER=$(lsb_release -r -s | cut -d. -f1)
  sudo wget https://download.swift.org/swift-5.7-release/ubuntu${UBUNTU_MAJOR_VER}04/swift-5.7-RELEASE/swift-5.7-RELEASE-ubuntu${UBUNTU_MAJOR_VER}.04.tar.gz
  sudo tar -xzvf swift-5.7-RELEASE-ubuntu${UBUNTU_MAJOR_VER}.04.tar.gz
  sudo rm swift-5.7-RELEASE-ubuntu${UBUNTU_MAJOR_VER}.04.tar.gz

  # scala
  apt install -y --no-install-recommends scala

  # C#
  apt install -y --no-install-recommends dotnet6

  # perl
  sudo PERL_MM_USE_DEFAULT=1 perl -MCPAN -e 'install Data::Compare'
}

install_dlrm_build_deps() {
    # taken from v4.0 submission Dockerfile.aarch64
    # need to build fbgemm and torchrec from source as wheels are not available for grace-hopper.
    # NOTE(vir): we use 0.6.0 which since 0.3.2 from reference impl is not available on arm.
    # TODO(vir): file MR with mlcommons to update dlrm reference impl torchrec to >= 0.6.0.

    # build and install fbgemm==0.6.0
    cd /tmp
    export CUDA_BIN_PATH="/usr/local/cuda/"
    export CUDACXX="/usr/local/cuda/bin/nvcc"
    export CUDNN_INCLUDE_DIR="/usr/local/cuda/include/"
    export CUDNN_LIBRARY="/usr/local/cuda/lib64/"
    export TORCH_CUDA_ARCH_LIST="Ampere Ada Hopper"
    export _GLIBCXX_USE_CXX11_ABI=1
    git clone --branch v0.6.0 --recursive https://github.com/pytorch/FBGEMM.git
    cd FBGEMM/fbgemm_gpu
    git submodule sync
    git submodule update --init --recursive
    python setup.py install
    cd /tmp && rm -rf /tmp/FBGEMM

    # build and install torchrec==0.6.0
    cd /tmp
    export CUDA_NVCC_EXECUTABLE="/usr/local/cuda/bin/nvcc"
    export CUDA_HOME="/usr/local/cuda"
    export CUDNN_INCLUDE_PATH="/usr/local/cuda/include/"
    export CUDNN_LIBRARY_PATH="/usr/local/cuda/lib64/"
    export LIBRARY_PATH="$LIBRARY_PATH:/usr/local/cuda/lib64"
    export USE_CUDA=1 USE_CUDNN=1
    export TORCH_CUDA_ARCH_LIST="Ampere Ada Hopper"
    export _GLIBCXX_USE_CXX11_ABI=1
    git clone --branch v0.6.0 --recursive https://github.com/pytorch/torchrec
    cd torchrec
    python setup.py bdist_wheel
    cd dist
    python3 -m pip install torchrec-0.6.0-py3-none-any.whl
    cd /tmp && rm -rf /tmp/torchrec
}

case ${BUILD_CONTEXT} in
  x86_64)
    install_gflags 2.2.1
    install_glog 0.6.0
    install_CUB x86_64
    install_nlohmann_json_x86_64
    install_boost_1_78
    install_nvrtc_dev x86_64
    install_mxeval_deps
    ;;
  aarch64-GraceHopper)
    install_gflags 2.2.2
    install_glog 0.6.0
    install_CUB aarch64
    install_nlohmann_json_aarch64
    install_boost_1_78
    # install_nvrtc_dev sbsa
    install_mxeval_deps
    install_dlrm_build_deps
    ;;
  aarch64-Orin)
    install_CUB aarch64
    install_nlohmann_json_aarch64
    install_cupva_orin
    ;;
  *)
    echo "Supported BUILD_CONTEXT are only x86_64, aarch64-GraceHopper, and aarch64-Orin."
    exit 1
    ;;
esac
