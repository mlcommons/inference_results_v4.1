set -e

python3 -m pip install --upgrade pip

python3 -m pip install -r requirements.${BUILD_CONTEXT}.1.txt
python3 -m pip install -r requirements.${BUILD_CONTEXT}.2.txt
