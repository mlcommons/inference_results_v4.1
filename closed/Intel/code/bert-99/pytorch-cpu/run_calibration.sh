set -x

if [ ! -d "${DATA_DIR}" ]; then
	echo "please export the DATA_DIR first!"
	exit 1
fi

if [ ! -d "${MODEL_DIR}" ]; then
	echo "please export the MODEL_DIR first!"
	exit 1
fi


#convert dataset and model
pushd models
python save_bert_inference.py -m ${MODEL_DIR} -o ${MODEL_DIR}/bert.pt
popd

pushd datasets
python save_squad_features.py -m ${MODEL_DIR} -d ${DATA_DIR} -o ${MODEL_DIR}/squad.pt
popd

