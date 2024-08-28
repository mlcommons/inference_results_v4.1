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
	--tags=run,mlperf,inference,generate-run-cmds,_submission \
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
	--env.CM_MLPERF_NEURALMAGIC_MODEL_ZOO_STUB=zoo:nlp/question_answering/obert-large/pytorch/huggingface/squad/pruned95-none-vnni \
	--quiet
```
*Note that if you want to use the [latest automation recipes](https://access.cknowledge.org/playground/?action=scripts) for MLPerf (CM scripts),
 you should simply reload gateoverflow@cm4mlops without checkout and clean CM cache as follows:*

```bash
cm rm repo gateoverflow@cm4mlops
cm pull repo gateoverflow@cm4mlops
cm rm cache -f

```

## Results

Platform: phoenix_Amd_Am5-reference-cpu-deepsparse-vdefault-default_config

### Accuracy Results 
`F1`: `90.17833`, Required accuracy for closed division `>= 89.96526`

### Performance Results 
`Samples per second`: `38.4994`
