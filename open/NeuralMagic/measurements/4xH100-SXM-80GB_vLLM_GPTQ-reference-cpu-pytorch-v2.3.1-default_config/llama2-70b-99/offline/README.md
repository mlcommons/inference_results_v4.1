This experiment is generated using the [MLCommons Collective Mind automation framework (CM)](https://github.com/mlcommons/ck).

*Check [CM MLPerf docs](https://mlcommons.github.io/inference) for more details.*

## Host platform

* OS version: Linux-6.5.0-35-generic-x86_64-with-glibc2.35
* CPU version: x86_64
* Python version: 3.10.12 (main, Nov 20 2023, 15:14:05) [GCC 11.4.0]
* MLCommons CM version: 2.3.4

## CM Run Command

See [CM installation guide](https://github.com/mlcommons/ck/blob/master/docs/installation.md).

```bash
pip install -U cmind

cm rm cache -f

cm pull repo mlcommons@cm4mlops --checkout=3955da1f609bc9c74a9e05fba3cdf41f78d8f633

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
	--vllm_model_name=nm-testing/Llama-2-70B-Chat-GPTQ \
	--rerun \
	--execution_mode=valid \
	--scenario=Server \
	--server_target_qps=4
```
*Note that if you want to use the [latest automation recipes](https://access.cknowledge.org/playground/?action=scripts) for MLPerf (CM scripts),
 you should simply reload mlcommons@cm4mlops without checkout and clean CM cache as follows:*

```bash
cm rm repo mlcommons@cm4mlops
cm pull repo mlcommons@cm4mlops
cm rm cache -f

```

## Results

Platform: 4xH100-SXM-80GB_vLLM_GPTQ-reference-cpu-pytorch-v2.3.1-default_config

### Accuracy Results 
`ROUGE1`: `44.2003`, Required accuracy for closed division `>= 43.98689`
`ROUGE2`: `21.8589`, Required accuracy for closed division `>= 21.81485`
`ROUGEL`: `28.4858`, Required accuracy for closed division `>= 28.33004`
`TOKENS_PER_SAMPLE`: `586.9`, Required accuracy for closed division `>= 265.005`

### Performance Results 
`Samples per second`: `1577.11`
