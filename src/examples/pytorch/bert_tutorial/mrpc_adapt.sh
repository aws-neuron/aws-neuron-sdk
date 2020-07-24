#! /bin/bash

BUCKET_NAME=""

while IFS= read -r line; do
    BUCKET_NAME=$line
    break
done < bert_bucket_name.txt

echo "Bucket name is $BUCKET_NAME"

if [[ "$BUCKET_NAME" == "" ]]; then
    echo "Failed to read from bert_bucket_name.txt - did you run setup.sh?"
    return 1
fi
  
export GLUE_DIR="$(pwd)/glue_data"
export TRANSFORMER_DIR="$(pwd)/transformers"
export TASK_NAME=MRPC

if [ -d $GLUE_DIR ]; then
    echo "Using GLUE data (including MRPC) from $GLUE_DIR"
else
    echo "Could not find GLUE data (including MRPC) at $GLUE_DIR, please make sure you ran setup.sh"
fi

if [ -d $TRANSFORMER_DIR ]; then
    echo "Using transformer directory $TRANSFORMER_DIR"   
else
    echo "Could not find transformer directory $TRANSFORMER_DIR, please make sure you ran setup.sh"
fi

echo "Adapting for MRPC ..."
pushd .
cd $TRANSFORMER_DIR

python -m torch.distributed.launch --nproc_per_node 8 ./examples/text-classification/run_glue.py   \
    --model_name_or_path bert-large-uncased-whole-word-masking \
    --task_name MRPC \
    --do_train   \
    --do_eval   \
    --data_dir $GLUE_DIR/MRPC/   \
    --max_seq_length 128   \
    --per_gpu_eval_batch_size=8   \
    --per_gpu_train_batch_size=8   \
    --learning_rate 2e-5   \
    --num_train_epochs 3.0  \
    --output_dir /tmp/mrpc_output/ \
    --overwrite_output_dir   \
    --overwrite_cache

popd
echo "... adaptation complete"

mv /tmp/mrpc_output/ bert-large-uncased-mrpc

echo "Archive adapted model ..."
tar cvfz bert-large-uncased-mrpc.tar.gz bert-large-uncased-mrpc
echo "... archive complete"

echo "Copy model to S3 ..."
if output="$(aws s3 ls $BUCKET_NAME)"; then
    echo "Bucket $1 already exists"
    #TIMESTAMP=$(date +%F-%H-%M-%S)
    FOLDER="bert_tutorial"
    echo "Command: aws s3 cp bert-large-uncased-mrpc.tar.gz ${BUCKET_NAME}/${FOLDER}/"
    aws s3 cp bert-large-uncased-mrpc.tar.gz ${BUCKET_NAME}/${FOLDER}/
else
    echo "Failure - bucket $BUCKET_NAME not accessible!"
    echo "$output"
fi

export S3_LOCATION=${BUCKET_NAME}/${FOLDER}/bert-large-uncased-mrpc.tar.gz
echo "Output to: $S3_LOCATION"

if output="$(aws s3 ls $S3_LOCATION)"; then
    echo "Content successfully uploaded to $S3_LOCATION"
else
    echo "FAILURE - $S3_LOCATION not copied!"
fi
