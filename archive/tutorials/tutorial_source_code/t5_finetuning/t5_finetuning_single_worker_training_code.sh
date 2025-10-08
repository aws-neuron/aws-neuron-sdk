#!/bin/bash
set -eExuo pipefail

cd ~/transformers/examples/pytorch/summarization

# Create run.sh file
tee run.sh > /dev/null <<EOF
#!/bin/bash
set -eExuo
if [ \$NEURON_PARALLEL_COMPILE == "1" ]
then
    XLA_USE_BF16=1 python3 ./run_summarization.py \
    --model_name_or_path t5-small \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --do_train \
    --do_eval \
    --source_prefix "summarize: " \
    --max_source_length 512 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --overwrite_output_dir \
    --pad_to_max_length \
    --max_steps 100 \
    --max_eval_samples 100 \
    --gradient_accumulation_steps=32 \
    --output_dir /tmp/tst-summarization |& tee log_run
else
    XLA_USE_BF16=1 python3 ./run_summarization.py \
    --model_name_or_path t5-small \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --do_train \
    --do_eval \
    --source_prefix "summarize: " \
    --max_source_length 512 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --overwrite_output_dir \
    --pad_to_max_length \
    --gradient_accumulation_steps=32 \
    --output_dir /tmp/tst-summarization |& tee log_run
fi
EOF

chmod +x run.sh

# Run precompilation and training
neuron_parallel_compile ./run.sh

./run.sh

# Insert code into run summarization in order to predict with generate
tee temp_run_summarization.py > /dev/null <<EOF
import libneuronxla
# Disable configuring xla env
def _configure_env():
    pass
libneuronxla.configure_environment = _configure_env
EOF

cat run_summarization.py >> temp_run_summarization.py
mv temp_run_summarization.py run_summarization.py
chmod +x run_summarization.py

# Run run summarization to predict without generate
NEURON_NUM_DEVICES=0 python3 ./run_summarization.py \
    --model_name_or_path <CHECKPOINT_DIR> \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --do_predict \
    --predict_with_generate \
    --source_prefix "summarize: " \
    --per_device_eval_batch_size 4 \
    --max_source_length 512 \
    --pad_to_max_length \
    --no_cuda \
    --output_dir /tmp/tst-summarization |& tee log_run