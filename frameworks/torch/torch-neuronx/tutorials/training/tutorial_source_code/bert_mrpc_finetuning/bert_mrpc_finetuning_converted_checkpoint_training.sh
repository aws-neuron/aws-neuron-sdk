#!/bin/bash
set -eExuo

cd ~/transformers/examples/pytorch/text-classification
aws s3 cp --no-progress s3://neuron-s3/training_checkpoints/pytorch/dp_bert_large_hf_pretrain/ckpt_29688.pt ./ --no-sign-request

# Create convert file
tee convert.py > /dev/null <<EOF
import os
import sys
import argparse
import torch
import transformers
from transformers import (
    BertForPreTraining,
)
import torch_xla.core.xla_model as xm
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='bert-large-uncased',  help="Path to model identifier from huggingface.co/models")
    parser.add_argument('--output_saved_model_path', type=str, default='./hf_saved_model', help="Directory to save the HF pretrained model format.")
    parser.add_argument('--checkpoint_path', type=str, required=True, help="Path to pretrained checkpoint which needs to be converted to a HF pretrained model format")
    args = parser.parse_args(sys.argv[1:])

    model = BertForPreTraining.from_pretrained(args.model_name)
    check_point = torch.load(args.checkpoint_path, map_location='cpu')
    model.load_state_dict(check_point['model'], strict=False)
    model.save_pretrained(args.output_saved_model_path, save_config=True, save_function=xm.save)
    print("Done converting checkpoint {} to HuggingFace saved model in directory {}.".format(args.checkpoint_path, args.output_saved_model_path))
EOF

python convert.py --checkpoint_path ckpt_29688.pt

# Create run script
tee run_converted.sh > /dev/null <<EOF
#!/usr/bin/env bash
set -eExuo
export TASK_NAME=mrpc
export NEURON_CC_FLAGS="--model-type=transformer"
NEURON_RT_STOCHASTIC_ROUNDING_EN=1 torchrun --nproc_per_node=2 ./run_glue.py \\
--model_name_or_path hf_saved_model \\
--tokenizer_name bert-large-uncased \\
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
--output_dir /tmp/\$TASK_NAME/ |& tee log_run_converted
EOF

chmod +x run_converted.sh

# Pre-compile
neuron_parallel_compile ./run_converted.sh

#Run Training
./run_converted.sh
