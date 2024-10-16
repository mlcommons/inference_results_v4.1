# Collective Mind demo

This open submission is used to demonstrate [Collective Mind automation framework (MLCommons CM)](https://arxiv.org/abs/2406.16791)
with the latest [CM4MLops and CM4MLPerf automation recipes](https://access.cknowledge.org/playground/?action=scripts)
being developed as a community effort to support our open educational initiatives, reproducibility studies, artifact evaluation and optimization challenges 
in collaboration with ACM, IEEE and MLCommons: [cTuning.org/ae](https://cTuning.org/ae) .

## Host platform

* OS version: Linux-5.15.0-116-generic-x86_64-with-glibc2.35
* CPU version: x86_64
* Python version: 3.10.12 (main, Mar 22 2024, 16:50:05) [GCC 11.4.0]
* MLCommons CM version: 2.3.4

## CM Run Command

See the [CM installation guide](https://access.cknowledge.org/playground/?action=install).

```bash
pip install -U cmind

cm rm cache -f

cm pull repo mlcommons@cm4mlops --checkout=3955da1f609bc9c74a9e05fba3cdf41f78d8f633

cm run script \
	--tags=run-mlperf,inference,_r4.1 \
	--model=sdxl \
	--implementation=reference \
	--framework=pytorch \
	--category=datacenter \
	--scenario=Offline \
	--execution_mode=valid \
	--device=cuda \
	--quiet
```

*Note that if you want to use the [latest automation recipes](https://access.cknowledge.org/playground/?action=scripts) for MLPerf (CM4MLOps/CM4MLPerf scripts),
 you should simply reload mlcommons@cm4mlops without checkout and clean CM cache as follows:*


```bash
cm rm repo mlcommons@cm4mlops
cm pull repo mlcommons@cm4mlops --branch=dev
cm rm cache -f

```

## Results

Platform: cm-demo-gfursin-scaleway-L4-1-24G-reference-gpu-pytorch-v2.3.1-default_config

### Accuracy Results 
`CLIP_SCORE`: `31.75054`, Required accuracy for closed division `>= 31.68632` and `<= 31.81332`
`FID_SCORE`: `23.46805`, Required accuracy for closed division `>= 23.01086` and `<= 23.95008`

### Performance Results 
`Samples per second`: `0.125716`

## Future work

Learn more about our community initiatives to co-design more efficient and cost-effective AI/ML systems with Collective Mind, 
virtualized MLOps, MLPerf, Collective Knowledge Playground and reproducible optimization tournaments 
from our [white paper](https://arxiv.org/abs/2406.16791).
