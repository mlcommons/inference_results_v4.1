This experiment is generated using the [MLCommons Collective Mind automation framework (CM)](https://github.com/mlcommons/ck).

*Check [CM MLPerf docs](https://mlcommons.github.io/inference) for more details.*

## Host platform

* OS version: Linux-6.5.0-44-generic-x86_64-with-glibc2.35
* CPU version: x86_64
* Python version: 3.10.12 (main, Mar 22 2024, 16:50:05) [GCC 11.4.0]
* MLCommons CM version: 2.3.3

## CM Run Command

See [CM installation guide](https://github.com/mlcommons/ck/blob/master/docs/installation.md).

```bash
pip install -U cmind

cm rm cache -f

cm pull repo anandhu-eng@cm4mlops --checkout=b341b316e6536fa590c936520301d93b42b0b858

cm run script \
	--tags=run,mlperf,inference,generate-run-cmds \
	--adr.python.version_min=3.8 \
	--adr.compiler.tags=gcc \
	--implementation=reference \
	--model=bert-99 \
	--precision=int8 \
	--backend=deepsparse \
	--device=cpu \
	--scenario=Offline \
	--execution_mode=valid \
	--adr.mlperf-inference-implementation.max_batchsize=128 \
	--env.CM_MLPERF_NEURALMAGIC_MODEL_ZOO_STUB=zoo:nlp/question_answering/mobilebert-none/pytorch/huggingface/squad/base_quant-none \
	--quiet
```
*Note that if you want to use the [latest automation recipes](https://access.cknowledge.org/playground/?action=scripts) for MLPerf (CM scripts),
 you should simply reload anandhu-eng@cm4mlops without checkout and clean CM cache as follows:*

```bash
cm rm repo anandhu-eng@cm4mlops
cm pull repo anandhu-eng@cm4mlops
cm rm cache -f

```

## Results

Platform: ASUS_Vivobook-reference-cpu-deepsparse_v1.8.0-default_config

### Accuracy Results 
`F1`: `90.80559`, Required accuracy for closed division `>= 89.96526`

### Performance Results 
`Samples per second`: `17.251`
