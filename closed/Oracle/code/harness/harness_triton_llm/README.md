# MLPerf Triton LLM harness.
## Targeted workloads
This harness currently only supports Llama2-70B

## Software stack
| Software                | Version                                     |
|-------------------------|---------------------------------------------|
| Triton Inference Server | r24.06 + cherry-pick [ea3f365](https://github.com/triton-inference-server/server/commit/ea3f365c9784660dbe8f533671e977ded94a419c)                |
| Triton TRTLLM backend   | main (06/25) - compatible with TRTLLM 06/25 |

## Steps to run Llama2-70B
0. Enter the container and install TRTLLM. From `closed/NVIDIA`:
```
make prebuild
```
Then, from inside container:

```
make clone_trt_llm && make build_trt_llm
```


1. Run the following steps to install the required Triton software.

```
make clone_triton && make build_triton
```
- This should be required only once, as all compilations are binary files, not required if you re-enter the container
- There is a known issue here: You may get an error from git asking who you are. This is due to a cherry-pick in `clone_triton` of a [commit](https://github.com/triton-inference-server/server/commit/ea3f365c9784660dbe8f533671e977ded94a419c) comprising a gRPC hotfix by the Triton team - just run `git config --global user.name "Your Name"; git config --global user.email "youremail@nvidia.com"` to do a successful cherry-pick.

2. Generate the engines.

```
make generate_engines RUN_ARGS="--benchmarks=llama2 --scenarios=Offline[,Server] --harness_type=triton"
```

3. Run the harness.

```
make run_harness RUN_ARGS="--benchmarks=llama2 --scenarios=Offline[,Server] --harness_type=triton"
```


## Performance:

H200x1 Server (For triton):

| target_qps | completed_qps | tokens/sec | TTFT (ms)  | TPOT (ms)  |
|------------|---------------|------------|------------|------------|
| 12.5       | 12.45         | 3660.08    | 582.212653 | 199.225168 |

H200x1 Offline:
|        | tokens/sec     | QPS   |
|--------|----------------|-------|
| custom | 3757.47 | 12.78 |
| triton | 3752.66 | 12.77 |
| comparison | 99.87% | 99.87% |

H200x8 Offline: (from 06/09, needs fresh runs)
|        | tokens/sec     | QPS   |
|--------|----------------|-------|
| custom | 28941.8 | 98.4532 |
| triton | 28278.6 | 96.1968 |
| comparison | 97.71% | 97.71% |
