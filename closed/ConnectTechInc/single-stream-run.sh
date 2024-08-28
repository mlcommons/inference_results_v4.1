
#echo "QUANTIZING" | tee -a mlperf.log
#python build/TRTLLM/examples/quantization/quantize.py --model_dir ${MLPERF_SCRATCH_PATH}/models/GPTJ-6B/checkpo
#int-final/ --dtype float16 --qformat int4_awq --output_dir ${MLPERF_SCRATCH_PATH}/models/GPTJ-6B/orin-w4a16-awq --calib_size 512


echo "GENERATING_ENGINES singlestream" | tee -a mlperf.log
make generate_engines RUN_ARGS="--benchmarks=gptj --scenarios=singlestream --config_ver=high_accuracy --test_mode=AccuracyOnly" 2>&1| tee -a mlperf.log
echo "Running harness singlestream Accuracy" | tee -a mlperf.log
make run_harness RUN_ARGS="--benchmarks=gptj --scenarios=singlestream --config_ver=high_accuracy --test_mode=AccuracyOnly" 2>&1 | tee -a mlperf.log
echo "Running harness singlestream performance" | tee -a mlperf.log
make run_harness RUN_ARGS="--benchmarks=gptj --scenarios=singlestream --config_ver=high_accuracy --test_mode=PerformanceOnly" 2>&1 | tee -a mlperf.log
