# ResNet-50 Inference on CPU - Supplemental GNR Config
The contents here are intended to supplement the default workload [README](README.md), not be used in isolation.

## Workload modifications
Modify user_default.conf with the following contents:

### For Offline runs:
```
[default]
number_cores = 256
resnet50.*.performance_sample_count_override = 1024
*.Offline.target_qps = 46000
*.Server.target_qps = 39800
*.Server.min_duration = 600000
```
### For Server runs:
```
[default]
number_cores = 251
resnet50.*.performance_sample_count_override = 1024
*.Offline.target_qps = 46000
*.Server.target_qps = 39800
*.Server.min_duration = 600000
```
