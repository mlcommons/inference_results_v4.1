#!/bin/bash

export DOWNLOAD_DATA_DIR=/data/kits19/data

echo ${DOWNLOAD_DATA_DIR}/case_00185/imaging.nii.gz

if [ -s ${DOWNLOAD_DATA_DIR}/case_00185/imaging.nii.gz ]; then
    echo "Duplicating KITS19 case_00185 as case_00400..."
    cp -Rf ${DOWNLOAD_DATA_DIR}/case_00185 ${DOWNLOAD_DATA_DIR}/case_00400
else
    echo "KITS19 case_00185 not found! please download the dataset first..."
fi

BUILD_DIR=/model/build
mkdir ${BUILD_DIR}
cp mlperf.conf ${BUILD_DIR}/
cp calibration_result.json ${BUILD_DIR}/

make preprocess_data
make preprocess_calibration_data
make preprocess_gaussian_patches

export LD_PRELOAD=${CONDA_PREFIX}/lib/libiomp5.so:$LD_PRELOAD
python trace_model.py
