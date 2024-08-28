This experiment is generated using the [MLCommons Collective Mind automation framework (CM)](https://github.com/mlcommons/ck).

*Check [CM MLPerf docs](https://mlcommons.github.io/inference) for more details.*

## Host platform

* OS version: Linux-6.5.0-41-generic-x86_64-with-glibc2.35
* CPU version: x86_64
* Python version: 3.10.12 (main, Mar 22 2024, 16:50:05) [GCC 11.4.0]
* MLCommons CM version: 2.3.1

## CM Run Command

See [CM installation guide](https://github.com/mlcommons/ck/blob/master/docs/installation.md).

```bash
pip install -U cmind

cm rm cache -f

cm pull repo gateoverflow@cm4mlops --checkout=8a580d7323472728957ec25a9ed3e2d607ddcce3

cm run script \
	--tags=run-mlperf,inference,_full,_r4.1,_all-scenarios \
	--model=llama2-70b-99 \
	--implementation=reference \
	--division=open \
	--category=datacenter \
	--device=cpu \
	--quiet \
	--api_server=http://localhost:8000 \
	--adr.mlperf-implementation.tags=_repo.https://github.com/neuralmagic/inference,_branch.vllm \
	--vllm_model_name=neuralmagic/Llama-2-7b-chat-hf-FP8 \
	--execution_mode=valid \
	--max_test_duration=400 \
	--test_query_count=500 \
	--offline_target_qps=2 \
	--scenario=Offline \
	--server_target_qps=5 \
	--num_workers=1
```
*Note that if you want to use the [latest automation recipes](https://access.cknowledge.org/playground/?action=scripts) for MLPerf (CM scripts),
 you should simply reload gateoverflow@cm4mlops without checkout and clean CM cache as follows:*

```bash
cm rm repo gateoverflow@cm4mlops
cm pull repo gateoverflow@cm4mlops
cm rm cache -f

```

## Results

Platform: phoenix_Amd_Am5-reference-cpu-pytorch-v2.2.1-default_config

### Accuracy Results 
`ROUGE1`: `41.7123`, Required accuracy for closed division `>= 44.38677`
`ROUGE2`: `19.6143`, Required accuracy for closed division `>= 22.01316`
`ROUGEL`: `26.4603`, Required accuracy for closed division `>= 28.58758`
`TOKENS_PER_SAMPLE`: `296.6`, Required accuracy for closed division `>= 265.005` and `<= 323.895`

### Performance Results 
`Samples per second`: `1337.46`
