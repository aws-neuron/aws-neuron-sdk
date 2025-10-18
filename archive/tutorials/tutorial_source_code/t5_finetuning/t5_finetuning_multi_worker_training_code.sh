#!/bin/bash
set -eExuo pipefail

cd ~/transformers/examples/pytorch/summarization

# Create run 2 worker script
tee run_2w.sh > /dev/null <<EOF
#!/bin/bash
set -eExuo
if [ \$NEURON_PARALLEL_COMPILE == "1" ]
then
    XLA_USE_BF16=1 torchrun --nproc_per_node=2 ./run_summarization.py \
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
    XLA_USE_BF16=1 torchrun --nproc_per_node=2 ./run_summarization.py \
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

chmod +x run_2w.sh

# Precompile and run training
neuron_parallel_compile ./run_2w.sh

./run_2w.sh