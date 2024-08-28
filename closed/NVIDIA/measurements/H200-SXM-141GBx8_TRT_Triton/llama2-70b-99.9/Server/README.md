To run this benchmark, first follow the setup steps in `closed/NVIDIA/README.md`. Then to run the harness:

```
make run_harness RUN_ARGS="--benchmarks=llama2-70b --scenarios=Server --harness_type=triton --test_mode=AccuracyOnly"
make run_harness RUN_ARGS="--benchmarks=llama2-70b --scenarios=Server --harness_type=triton --test_mode=PerformanceOnly"
```

For more details, please refer to `closed/NVIDIA/README.md`.