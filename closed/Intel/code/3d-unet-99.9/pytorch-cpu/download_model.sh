#!/bin/bash

ZENODO_PYTORCH="https://zenodo.org/record/5597155/files/3dunet_kits19_pytorch_checkpoint.pth?download=1"
PYTORCH_MODEL="/model/3dunet_kits19_pytorch_checkpoint.pth"

wget -O ${PYTORCH_MODEL} ${ZENODO_PYTORCH}
