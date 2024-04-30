#!/bin/bash
set -eExuo
cd aws-neuron-sdk/src/examples/pytorch
cd torchserve
python trace_bert_neuronx.py
ls

pip install transformers==4.20.1 torchserve==0.7.0 torch-model-archiver==0.7.0 captum==0.6.0

sudo apt install openjdk-11-jdk -y

mkdir model_store
MAX_LENGTH=$(jq '.max_length' config.json)
BATCH_SIZE=$(jq '.batch_size' config.json)
MODEL_NAME=bert-max_length$MAX_LENGTH-batch_size$BATCH_SIZE
torch-model-archiver --model-name "$MODEL_NAME" --version 1.0 --serialized-file ./bert_neuron_b6.pt --handler "./handler_bert_neuronx.py" --extra-files "./config.json" --export-path model_store

ls model_store

torchserve --start --ncs --model-store model_store --ts-config torchserve.config 2>&1 >torchserve.log
sleep 10
curl http://127.0.0.1:8080/ping

MAX_BATCH_DELAY=5000 # ms timeout before a partial batch is processed
INITIAL_WORKERS=2 # Number from table above
curl -X POST "http://localhost:8081/models?url=$MODEL_NAME.mar&batch_size=$BATCH_SIZE&initial_workers=$INITIAL_WORKERS&max_batch_delay=$MAX_BATCH_DELAY"

python infer_bert.py

python benchmark_bert.py

torchserve --stop
