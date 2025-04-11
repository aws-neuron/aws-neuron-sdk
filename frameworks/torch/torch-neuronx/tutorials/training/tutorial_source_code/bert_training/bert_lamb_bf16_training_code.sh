#!/bin/bash
set -eExuo

# Run the training script
cd ~/aws-neuron-samples/torch-neuronx/training/dp_bert_hf_pretrain
torchrun --nproc_per_node=32 \
dp_bert_large_hf_pretrain_hdf5.py \
--max_steps 7032 \
--batch_size 16 \
--optimizer LAMB \
--lr 6e-3 \
--grad_accum_usteps 128 | tee run_pretrain_log.txt
torchrun_exit_status=${PIPESTATUS[0]}
echo "Training return code: $torchrun_exit_status"
exit $torchrun_exit_status
