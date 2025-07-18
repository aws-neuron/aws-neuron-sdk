#!/bin/bash
set -eExuo

cd ~/transformers/examples/pytorch/text-classification

# Create the run_2w.sh file
tee run_2w.sh > /dev/null <<EOF
#!/usr/bin/env bash
set -eExuo
export TASK_NAME=mrpc
export NEURON_CC_FLAGS="--model-type=transformer"
NEURON_RT_STOCHASTIC_ROUNDING_EN=1 torchrun --nproc_per_node=2 ./run_glue.py \\
--model_name_or_path bert-large-uncased \\
--task_name \$TASK_NAME \\
--do_train \\
--do_eval \\
--bf16 \\
--use_cpu True \\
--max_seq_length 128 \\
--per_device_train_batch_size 8 \\
--learning_rate 2e-5 \\
--num_train_epochs 5 \\
--save_total_limit 1 \\
--overwrite_output_dir \\
--output_dir /tmp/\$TASK_NAME/ |& tee log_run_2w
EOF

chmod +x run_2w.sh

# Pre-compile and train
neuron_parallel_compile ./run_2w.sh

./run_2w.sh
