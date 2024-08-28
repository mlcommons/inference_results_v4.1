This experiment is generated using the [MLCommons Collective Mind automation framework (CM)](https://github.com/mlcommons/cm4mlops).

*Check [CM MLPerf docs](https://docs.mlcommons.org/inference) for more details.*

## Host platform

* OS version: Linux-5.15.0-94-generic-x86_64-with-glibc2.35
* CPU version: x86_64
* Python version: 3.10.12 (main, Nov 20 2023, 15:14:05) [GCC 11.4.0]
* MLCommons CM version: 2.3.3

## CM Run Command

See [CM installation guide](https://docs.mlcommons.org/inference/install/).

```bash
pip install -U cmind

cm rm cache -f

cm pull repo mlcommons@cm4mlops --checkout=6d87fa795ef001d7d76fe10217d2e2fa5e9b9742

cm run script \
	--tags=run-mlperf,inference,_full,_compliance \
	--model=llama2-70b-99 \
	--implementation=reference \
	--device=cpu \
	--quiet \
	--api_server=http://localhost:8000 \
	--adr.mlperf-implementation.tags=_repo.https://github.com/neuralmagic/inference,_branch.vllm \
	--vllm_model_name=nm-testing/Llama-2-70b-chat-hf-FP8 \
	--test_query_count=1 \
	--server_target_qps=1 \
	--num_workers=1 \
	--scenario=Offline \
	--max_test_duration=2000 \
	--execution_mode=valid \
	--offline_target_qps=3 \
	--division=closed \
	-category=datacenter
```
*Note that if you want to use the [latest automation recipes](https://docs.mlcommons.org/inference) for MLPerf (CM scripts),
 you should simply reload mlcommons@cm4mlops without checkout and clean CM cache as follows:*

```bash
cm rm repo mlcommons@cm4mlops
cm pull repo mlcommons@cm4mlops
cm rm cache -f

```

## Results

Platform: vLLM_8xL40S-reference-cpu-pytorch-v2.3.1-default_config

### Accuracy Results 
`ROUGE1`: `44.3329`, Required accuracy for closed division `>= 43.98689`
`ROUGE2`: `22.0197`, Required accuracy for closed division `>= 21.81485`
`ROUGEL`: `28.598`, Required accuracy for closed division `>= 28.33004`
`TOKENS_PER_SAMPLE`: `300.5`, Required accuracy for closed division `>= 265.005` and `<= 323.895`

### Performance Results 
`Samples per second`: `948.198`
