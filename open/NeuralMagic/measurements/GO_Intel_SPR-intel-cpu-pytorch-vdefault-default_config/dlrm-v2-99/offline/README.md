This experiment is generated using the [MLCommons Collective Mind automation framework (CM)](https://github.com/mlcommons/ck).

*Check [CM MLPerf docs](https://mlcommons.github.io/inference) for more details.*

## Host platform

* OS version: Linux-6.2.0-39-generic-x86_64-with-glibc2.35
* CPU version: x86_64
* Python version: 3.10.12 (main, Mar 22 2024, 16:50:05) [GCC 11.4.0]
* MLCommons CM version: 2.3.3

## CM Run Command

See [CM installation guide](https://github.com/mlcommons/ck/blob/master/docs/installation.md).

```bash
pip install -U cmind

cm rm cache -f

cm pull repo gateoverflow@cm4mlops --checkout=3f30d2bfea6d04a525bea782ca4ed4e2c399b673

cm run script \
	--tags=run-mlperf,inference,_r4.0 \
	--model=dlrm-v2-99 \
	--implementation=intel \
	--quiet \
	--execution_mode=valid \
	--offline_target_qps=870
```
*Note that if you want to use the [latest automation recipes](https://access.cknowledge.org/playground/?action=scripts) for MLPerf (CM scripts),
 you should simply reload gateoverflow@cm4mlops without checkout and clean CM cache as follows:*

```bash
cm rm repo gateoverflow@cm4mlops
cm pull repo gateoverflow@cm4mlops
cm rm cache -f

```

## Results

Platform: GO_2xRTX4090-intel-cpu-pytorch-vdefault-default_config

### Accuracy Results 
`AUC`: `80.199`, Required accuracy for closed division `>= 79.5069`

### Performance Results 
`Samples per second`: `822.557`
