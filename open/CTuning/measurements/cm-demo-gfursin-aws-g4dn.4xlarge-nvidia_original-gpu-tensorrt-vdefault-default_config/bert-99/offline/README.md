# Collective Mind demo

This open submission is used to demonstrate [Collective Mind automation framework (MLCommons CM)](https://arxiv.org/abs/2406.16791)
with the latest [CM4MLops and CM4MLPerf automation recipes](https://access.cknowledge.org/playground/?action=scripts)
being developed as a community effort to support our open educational initiatives, reproducibility studies, artifact evaluation and optimization challenges 
in collaboration with ACM, IEEE and MLCommons: [cTuning.org/ae](https://cTuning.org/ae) .

## Host platform

* OS version: Linux-6.5.0-1023-aws-x86_64-with-glibc2.29
* CPU version: x86_64
* Python version: 3.8.10 (default, Mar 25 2024, 10:42:49) 
[GCC 9.4.0]
* MLCommons CM version: 2.3.4

## CM Run Command

See the [CM installation guide](https://access.cknowledge.org/playground/?action=install).

```bash

pip install -U cmind

cm rm cache -f

cm pull repo mlcommons@cm4mlops --checkout=ef2193e4d104b0c248f22fae4d27671798a4c53e

cm run script \
	--tags=run-mlperf,inference,_r4.1 \
	--model=bert-99 \
	--implementation=nvidia \
	--framework=tensorrt \
	--category=datacenter \
	--scenario=Offline \
	--execution_mode=valid \
	--device=cuda \
	--quiet \
	--docker_cm_repo=mlcommons@cm4mlops \
	--docker_cm_repo_flags=--branch=mlperf-inference
```

*Note that if you want to use the [latest automation recipes](https://access.cknowledge.org/playground/?action=scripts) for MLPerf (CM4MLOps/CM4MLPerf scripts),
 you should simply reload mlcommons@cm4mlops without checkout and clean CM cache as follows:*


```bash
cm rm repo mlcommons@cm4mlops
cm pull repo mlcommons@cm4mlops --branch=dev
cm rm cache -f

```

## Results

Platform: cm-demo-gfursin-aws-g4dn.4xlarge-nvidia_original-gpu-tensorrt-vdefault-default_config

### Accuracy Results 
`F1`: `90.21495`, Required accuracy for closed division `>= 89.96526`

### Performance Results 
`Samples per second`: `381.124`

## Future work

Learn more about our community initiatives to co-design more efficient and cost-effective AI/ML systems with Collective Mind, 
virtualized MLOps, MLPerf, Collective Knowledge Playground and reproducible optimization tournaments 
from our [white paper](https://arxiv.org/abs/2406.16791).
