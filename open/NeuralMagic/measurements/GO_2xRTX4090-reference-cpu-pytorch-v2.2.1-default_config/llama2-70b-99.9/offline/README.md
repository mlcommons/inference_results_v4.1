This experiment is generated using the [MLCommons Collective Mind automation framework (CM)](https://github.com/mlcommons/ck).

*Check [CM MLPerf docs](https://mlcommons.github.io/inference) for more details.*

## Host platform

* OS version: Linux-6.2.0-39-generic-x86_64-with-glibc2.37
* CPU version: x86_64
* Python version: 3.11.4 (main, Dec  7 2023, 15:43:41) [GCC 12.3.0]
* MLCommons CM version: 2.3.1

## CM Run Command

See [CM installation guide](https://github.com/mlcommons/ck/blob/master/docs/installation.md).

```bash
pip install -U cmind

cm rm cache -f

cm pull repo gateoverflow@cm4mlops --checkout=cab74c5faac1d74acfa010481375939fdd32930c

cm run script \
	--tags=run-mlperf,inference,_full,_r4.1 \
	--model=llama2-70b-99 \
	--implementation=reference \
	--division=open \
	--category=datacenter \
	--device=cpu \
	--quiet \
	--api_server=http://localhost:8000 \
	--adr.mlperf-implementation.tags=_repo.https://github.com/gateoverflow/inference_temp,_branch.vllm \
	--vllm_model_name=nm-testing/Llama-2-70B-Chat-GPTQ \
	--server_target_qps=0.6 \
	--test_query_count=10 \
	--execution_mode=valid
```
*Note that if you want to use the [latest automation recipes](https://access.cknowledge.org/playground/?action=scripts) for MLPerf (CM scripts),
 you should simply reload gateoverflow@cm4mlops without checkout and clean CM cache as follows:*

```bash
cm rm repo gateoverflow@cm4mlops
cm pull repo gateoverflow@cm4mlops
cm rm cache -f

```

## Results

Platform: GO_2xRTX4090-reference-cpu-pytorch-v2.2.1-default_config

### Accuracy Results 
`ROUGE1`: `44.0047`, Required accuracy for closed division `>= 44.38677`
`ROUGE2`: `21.7276`, Required accuracy for closed division `>= 22.01316`
`ROUGEL`: `28.4008`, Required accuracy for closed division `>= 28.58758`
`TOKENS_PER_SAMPLE`: `291.9`, Required accuracy for closed division `>= 265.005` and `<= 323.895`

### Performance Results 
`Samples per second`: `424.895`
