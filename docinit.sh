#!/bin/bash

python3 -m pip install -r docs/requirements.txt

if [ ! -e docs ]; then
    git clone https://github.com/GATEOverflow/inference_results_visualization_template.git docs
    test $? -eq 0 || exit $?
fi

if [ ! -e overrides ]; then
    cp -r docs/overrides overrides
    test $? -eq 0 || exit $?
fi

if [ ! -e docs/javascripts/config.js ]; then
   if [ -n "${INFERENCE_RESULTS_VERSION}" ]; then
   	echo "var result_version=\"${INFERENCE_RESULTS_VERSION}\";" >> docs/javascripts/config.js;
   else
	echo "Please export INFERENCE_RESULTS_VERSION=v4.1 or the corresponding version";
	exit 1
   fi
fi

if [ ! -e docs/thirdparty/tablesorter ]; then
    cd docs/thirdparty && git clone https://github.com/Mottie/tablesorter.git && cd -
    test $? -eq 0 || exit $?
fi

#python3 process.py
#python3 process_results_table.py
