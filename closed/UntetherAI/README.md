# MLPerf Inference v4.1 - Untether AI speedAI240 submissions

Untether AI present Edge and Datacenter submissions under two categories:
- Available: [speedAI240 Slim](https://www.untether.ai/download/speedai240-slim-product-brief/);
- Preview: speedAI240 "Preview".

The submissions use the [KRAI](https://krai.ai) [KILT](http://github.com/krai/kilt-mlperf)
technology for fast, efficient and scalable inference, and the
[KRAI X](http://github.com/krai/axs) technology for workflow automation.

Detailed setup instructions per benchmark are provided in README files under the
[code](code) directory.  Individual benchmarking commands per system,
benchmark, scenario and mode are provided in README files under the respective
[measurements](measurements) directories.

The source code has been released under the permissive MIT license across
several public repositories (under the `mlperf_4.1` branches created by the
v4.1 submission deadline):

- https://github.com/krai/kilt-mlperf (KRAI Inference Library Technology for MLPerf submissions)
- https://github.com/krai/kilt4uai (KILT plugin for Untether AI products)
- https://github.com/krai/axs (KRAI X Workflow Automation Technology)
- https://github.com/krai/axs2mlperf
- https://github.com/krai/axs2system
- https://github.com/krai/axs2kilt
- https://github.com/krai/axs2uai (see a note below)

As some configuration files could disclose the submitted results (e.g. target QPS and latency values), they have been excluded from the public `axs2uai` repository prior to release and included in this submission package under [collections](collections), namely:
- `axs2uai/uai_sut_collection` is under `./collections/uai_sut_collection`;
- `axs2uai/configs_collection` is under `./collections/configs_collection`.

These files will be merged into the public repository following the official results announcement.

## Contact

### [Untether AI](https://www.untether.ai/about/contact/)

### [KRAI](info@krai.ai)
