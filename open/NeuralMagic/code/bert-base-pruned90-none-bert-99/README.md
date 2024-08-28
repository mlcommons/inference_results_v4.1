The sparsified bert submissions are using the [MLCommons reference implementation](https://github.com/mlcommons/cm4mlops/tree/main/script/app-mlperf-inference-mlcommons-python) extended by NeuralMagic to add the [deepsparse backend](https://github.com/neuralmagic/inference/blob/deepsparse/language/bert/deepsparse_SUT.py).

Please follow [this script](https://github.com/mlcommons/cm4mlops/blob/main/script/run-all-mlperf-models/run-pruned-bert.sh) for generating an end to end submission for the sparsified bert models. 
