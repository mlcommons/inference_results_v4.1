number_threads=`nproc --all`
export number_cores=`lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l`
export NUM_SOCKETS=`grep physical.id /proc/cpuinfo | sort -u | wc -l`
export CPUS_PER_SOCKET=$((number_cores/NUM_SOCKETS))

export CPUS_PER_PROCESS=${CPUS_PER_SOCKET}  # which determine how much processes will be used
export CPUS_PER_INSTANCE=2  # instance-per-process number=CPUS_PER_PROCESS/CPUS_PER_INSTANCE
                             # total-instance = instance-per-process * process-per-socket
export CPUS_FOR_LOADGEN=1   # number of cpus for loadgen
                            # finally used in our code is max(CPUS_FOR_LOADGEN, left cores for instances)
export BATCH_SIZE=200
export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
