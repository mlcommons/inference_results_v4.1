install_loadgen_x86_64(){
    cd /tmp
    git clone https://gitlab-master.nvidia.com/zihaok/mlperf-loadgen-whl.git
    python3 -m pip install mlperf-loadgen-whl/mlcommons_loadgen-${MLPERF_VER#v}-cp310-cp310-linux_x86_64.whl
    rm -rf mlperf-loadgen-whl
}

install_loadgen_orin(){
    cd /tmp
    git clone https://gitlab-master.nvidia.com/zihaok/mlperf-loadgen-whl.git
    python3 -m pip install mlperf-loadgen-whl/mlcommons_loadgen-${MLPERF_VER#v}-cp310-cp310-linux_aarch64.whl
    rm -rf mlperf-loadgen-whl
}

case $BUILD_CONTEXT in
  x86_64)
    install_loadgen_x86_64
    ;;
  aarch64-GraceHopper)
    echo "L0 doesn't contain grace hopper system so we don't install loadgen for SDXL to pass L0."
    ;;
  aarch64-Orin)
    install_loadgen_orin
    ;;
  *)
    echo "Supported BUILD_CONTEXT are only x86_64, aarch64-GraceHopper, and aarch64-Orin."
    exit 1
    ;;
esac