# BERT Inference on CPU - Supplemental GNR Config
The contents here are intended to supplement the default workload [README](README.md), not be used in isolation.

## Workload modifications
Modify user_default.conf with the following contents:
```
[default]
number_cores = 256
bert.Offline.target_qps = 3600
bert.Server.target_qps = 2435
```
