#!/bin/bash

export DATA_DIR=/data
cm run script --tags=get,preprocessed,dataset,criteo,_multihot,_mlc  -j --to=${DATA_DIR}

# cm doesn't recognize 'to' location.  Manually finding and moving from cache.
DLRM_PREPROCESSED=$(find /root/CM/repos/local/cache -name dlrm_preprocessed)
mv ${DLRM_PREPROCESSED}/* /data/
