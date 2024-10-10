INFERENCE_RESULTS_REPO_OWNER=${INFERENCE_RESULTS_REPO_OWNER:-mlcommons}
if [ ! -e docs ]; then
    git clone https://github.com/GATEOverflow/inference_results_visualization_template.git docs
    test $? -eq 0 || exit $?
fi
cp docs/docinit.sh .
INFERENCE_RESULTS_VERSION=v4.1 INFERENCE_RESULTS_REPO_OWNER=${INFERENCE_RESULTS_REPO_OWNER} INFERENCE_RESULTS_REPO_BRANCH=main INFERENCE_RESULTS_REPO_NAME=inference_results_v4.1 bash docinit.sh
