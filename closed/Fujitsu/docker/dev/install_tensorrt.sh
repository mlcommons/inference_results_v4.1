#!/bin/bash

set -e

install_deb_pkg() {
    wget $TRT_DEB_URL/$1 -O $1 && dpkg -i $1 && rm $1;
}


install_tensorrt_x86_64_nightly(){
    apt remove -y tensorrt-dev libnvparsers-dev
    rm -rf /usr/local/lib/python3.10/dist-packages/tensorrt*
    export TRT_NIGHTLY_URL=https://urm.nvidia.com/artifactory/sw-tensorrt-generic/cicd/main/L1_Nightly/336/trt_build_x86_64_centos7_cuda12.4_full_optimized.tar
    cd /tmp
    wget ${TRT_NIGHTLY_URL} --user tensorrt-read-only --password "Tensorrt@123" -O TRT.tar
    tar -xf TRT.tar
    cd source/install/x86_64-gnu
    mkdir trt
    tar -xvzf cuda-${CUDA_VER}/release_tarfile/TensorRT-*.*.x86_64-gnu.cuda-${CUDA_VER}.tar.gz -C trt --strip-components 1
    tar -xvzf TensorRT-*.*.x86_64-gnu.cuda-${CUDA_VER}.internal.tar.gz -C trt --strip-components 1
    cp -rv trt/lib/* /usr/lib/x86_64-linux-gnu/
    cp -rv trt/include/* /usr/include/x86_64-linux-gnu/
    cp -rv trt/bin/* /usr/bin/
    python3 -m pip install trt/python/tensorrt-*-cp310-none-linux_x86_64.whl
    cd ../../..
    rm -rf source
    rm -f TRT.tar
}

install_tensorrt_x86_64(){
    apt remove -y tensorrt-dev libnvparsers-dev
    rm -rf /usr/local/lib/python3.10/dist-packages/tensorrt*
    # Path to install TRT RC 10.2.0.19
    export TRT_DEB_URL=http://cuda-repo/release-candidates/Libraries/TensorRT/v10.2/${TRT_VER}-1ee99adb/12.5-r555/Ubuntu22_04-x64-manylinux_2_17/deb/
    install_deb_pkg libnvinfer${TRT_MAJOR_VER}_${TRT_VER}-1+cuda12.5_amd64.deb
    install_deb_pkg libnvinfer-headers-dev_${TRT_VER}-1+cuda12.5_amd64.deb
    install_deb_pkg libnvinfer-dev_${TRT_VER}-1+cuda12.5_amd64.deb
    install_deb_pkg libnvinfer-headers-plugin-dev_${TRT_VER}-1+cuda12.5_amd64.deb
    install_deb_pkg libnvinfer-lean${TRT_MAJOR_VER}_${TRT_VER}-1+cuda12.5_amd64.deb
    install_deb_pkg libnvinfer-lean-dev_${TRT_VER}-1+cuda12.5_amd64.deb
    install_deb_pkg libnvinfer-dispatch${TRT_MAJOR_VER}_${TRT_VER}-1+cuda12.5_amd64.deb
    install_deb_pkg libnvinfer-dispatch-dev_${TRT_VER}-1+cuda12.5_amd64.deb
    install_deb_pkg libnvinfer-plugin${TRT_MAJOR_VER}_${TRT_VER}-1+cuda12.5_amd64.deb
    install_deb_pkg libnvinfer-plugin-dev_${TRT_VER}-1+cuda12.5_amd64.deb
    install_deb_pkg libnvinfer-vc-plugin${TRT_MAJOR_VER}_${TRT_VER}-1+cuda12.5_amd64.deb
    install_deb_pkg libnvinfer-vc-plugin-dev_${TRT_VER}-1+cuda12.5_amd64.deb
    install_deb_pkg libnvonnxparsers${TRT_MAJOR_VER}_${TRT_VER}-1+cuda12.5_amd64.deb
    install_deb_pkg libnvonnxparsers-dev_${TRT_VER}-1+cuda12.5_amd64.deb
    install_deb_pkg python3-libnvinfer_${TRT_VER}-1+cuda12.5_amd64.deb
    install_deb_pkg python3-libnvinfer-lean_${TRT_VER}-1+cuda12.5_amd64.deb
    install_deb_pkg python3-libnvinfer-dispatch_${TRT_VER}-1+cuda12.5_amd64.deb
    install_deb_pkg python3-libnvinfer-dev_${TRT_VER}-1+cuda12.5_amd64.deb
    install_deb_pkg libnvinfer-bin_${TRT_VER}-1+cuda12.5_amd64.deb
    ln -sf /usr/src/tensorrt/bin/trtexec /usr/bin/trtexec
    unset -f install_deb_pkg
}


install_tensorrt_aarch64(){
    apt remove -y tensorrt-dev libnvparsers-dev
    rm -rf /usr/local/lib/python3.10/dist-packages/tensorrt*
    # Path to install TRT RC 10.2.0.19
    CUDA_VER=12.5
    export TRT_DEB_URL=http://cuda-repo/release-candidates/Libraries/TensorRT/v10.2/${TRT_VER}-1ee99adb/12.5-r555/Ubuntu22_04-aarch64-manylinux_2_31/deb/
    install_deb_pkg libnvinfer${TRT_MAJOR_VER}_${TRT_VER}-1+cuda${CUDA_VER}_arm64.deb
    install_deb_pkg libnvinfer-headers-dev_${TRT_VER}-1+cuda${CUDA_VER}_arm64.deb
    install_deb_pkg libnvinfer-dev_${TRT_VER}-1+cuda${CUDA_VER}_arm64.deb
    install_deb_pkg libnvinfer-headers-plugin-dev_${TRT_VER}-1+cuda${CUDA_VER}_arm64.deb
    install_deb_pkg libnvinfer-lean${TRT_MAJOR_VER}_${TRT_VER}-1+cuda${CUDA_VER}_arm64.deb
    install_deb_pkg libnvinfer-lean-dev_${TRT_VER}-1+cuda${CUDA_VER}_arm64.deb
    install_deb_pkg libnvinfer-dispatch${TRT_MAJOR_VER}_${TRT_VER}-1+cuda${CUDA_VER}_arm64.deb
    install_deb_pkg libnvinfer-dispatch-dev_${TRT_VER}-1+cuda${CUDA_VER}_arm64.deb
    install_deb_pkg libnvinfer-plugin${TRT_MAJOR_VER}_${TRT_VER}-1+cuda${CUDA_VER}_arm64.deb
    install_deb_pkg libnvinfer-plugin-dev_${TRT_VER}-1+cuda${CUDA_VER}_arm64.deb
    install_deb_pkg libnvinfer-vc-plugin${TRT_MAJOR_VER}_${TRT_VER}-1+cuda${CUDA_VER}_arm64.deb
    install_deb_pkg libnvinfer-vc-plugin-dev_${TRT_VER}-1+cuda${CUDA_VER}_arm64.deb
    install_deb_pkg libnvonnxparsers${TRT_MAJOR_VER}_${TRT_VER}-1+cuda${CUDA_VER}_arm64.deb
    install_deb_pkg libnvonnxparsers-dev_${TRT_VER}-1+cuda${CUDA_VER}_arm64.deb
    install_deb_pkg python3-libnvinfer_${TRT_VER}-1+cuda${CUDA_VER}_arm64.deb
    install_deb_pkg python3-libnvinfer-lean_${TRT_VER}-1+cuda${CUDA_VER}_arm64.deb
    install_deb_pkg python3-libnvinfer-dispatch_${TRT_VER}-1+cuda${CUDA_VER}_arm64.deb
    install_deb_pkg python3-libnvinfer-dev_${TRT_VER}-1+cuda${CUDA_VER}_arm64.deb
    install_deb_pkg libnvinfer-bin_${TRT_VER}-1+cuda${CUDA_VER}_arm64.deb
    ln -sf /usr/src/tensorrt/bin/trtexec /usr/bin/trtexec
    unset -f wget_deb_pkg
}


install_tensorrt_orin(){
  # remove the trt 8.6 in jetpack base image
  apt remove -y nvidia-tensorrt-dev nvidia-tensorrt tensorrt-libs tensorrt
  apt autoremove -y
  export TRT_DEB_URL=http://cuda-repo/release-candidates/Libraries/TensorRT/v10.1/${TRT_VER}-fc0b6037/12.4-r550/l4t-aarch64/deb/
  CUDA_VER=12.4
  install_deb_pkg libnvinfer${TRT_MAJOR_VER}_${TRT_VER}-1+cuda${CUDA_VER}_arm64.deb
  install_deb_pkg libnvinfer-headers-dev_${TRT_VER}-1+cuda${CUDA_VER}_arm64.deb
  install_deb_pkg libnvinfer-dev_${TRT_VER}-1+cuda${CUDA_VER}_arm64.deb
  install_deb_pkg libnvinfer-headers-plugin-dev_${TRT_VER}-1+cuda${CUDA_VER}_arm64.deb
  install_deb_pkg libnvinfer-lean${TRT_MAJOR_VER}_${TRT_VER}-1+cuda${CUDA_VER}_arm64.deb
  install_deb_pkg libnvinfer-lean-dev_${TRT_VER}-1+cuda${CUDA_VER}_arm64.deb
  install_deb_pkg libnvinfer-dispatch${TRT_MAJOR_VER}_${TRT_VER}-1+cuda${CUDA_VER}_arm64.deb
  install_deb_pkg libnvinfer-dispatch-dev_${TRT_VER}-1+cuda${CUDA_VER}_arm64.deb
  install_deb_pkg libnvinfer-plugin${TRT_MAJOR_VER}_${TRT_VER}-1+cuda${CUDA_VER}_arm64.deb
  install_deb_pkg libnvinfer-plugin-dev_${TRT_VER}-1+cuda${CUDA_VER}_arm64.deb
  install_deb_pkg libnvinfer-vc-plugin${TRT_MAJOR_VER}_${TRT_VER}-1+cuda${CUDA_VER}_arm64.deb
  install_deb_pkg libnvinfer-vc-plugin-dev_${TRT_VER}-1+cuda${CUDA_VER}_arm64.deb
  install_deb_pkg libnvonnxparsers${TRT_MAJOR_VER}_${TRT_VER}-1+cuda${CUDA_VER}_arm64.deb
  install_deb_pkg libnvonnxparsers-dev_${TRT_VER}-1+cuda${CUDA_VER}_arm64.deb
  install_deb_pkg python3-libnvinfer_${TRT_VER}-1+cuda${CUDA_VER}_arm64.deb
  install_deb_pkg python3-libnvinfer-lean_${TRT_VER}-1+cuda${CUDA_VER}_arm64.deb
  install_deb_pkg python3-libnvinfer-dispatch_${TRT_VER}-1+cuda${CUDA_VER}_arm64.deb
  install_deb_pkg python3-libnvinfer-dev_${TRT_VER}-1+cuda${CUDA_VER}_arm64.deb
  install_deb_pkg libnvinfer-bin_${TRT_VER}-1+cuda${CUDA_VER}_arm64.deb
  ln -sf /usr/src/tensorrt/bin/trtexec /usr/bin/trtexec
  unset -f install_deb_pkg

}

case $BUILD_CONTEXT in
  x86_64)
    if [ "$USE_NIGHTLY" = "1" ]; then
      install_tensorrt_x86_64_nightly
    else
      install_tensorrt_x86_64
    fi
    ;;
  aarch64-GraceHopper)
    install_tensorrt_aarch64
    ;;
  aarch64-Orin)
    install_tensorrt_orin
    ;;
  *)
    echo "Supported BUILD_CONTEXT are only x86_64, aarch64-GraceHopper, and aarch64-Orin."
    exit 1
    ;;
esac
