# RetinaNet Inference on CPU - Supplemental GNR Config
The contents here are intended to supplement the default workload [README](README.md), not be used in isolation.

## Workload modifications
Modify user_default.conf with the following contents:
```
[default]
number_cores = 256
dlrm.Server.target_qps = 17750.0
dlrm.Offline.target_qps = 18500.0
```
