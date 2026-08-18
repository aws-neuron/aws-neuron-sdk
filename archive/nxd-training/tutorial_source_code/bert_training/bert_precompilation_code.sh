#!/bin/bash
# IMPORTANT: Neuron will stop supporting XLA-based training support in a future release. For now, this code sample is provided strictly for reference.
set -eExuo

# Navigate to the script directory and run the pre-compile script
cd ~/aws-neuron-samples/torch-neuronx/training/dp_bert_hf_pretrain
neuron_parallel_compile torchrun --nproc_per_node=32 \
dp_bert_large_hf_pretrain_hdf5.py \
--steps_this_run 10 \
--batch_size 16 \
--grad_accum_usteps 32 | tee compile_log.txt
torchrun_exit_status=${PIPESTATUS[0]}
echo "Training return code: $torchrun_exit_status"
exit $torchrun_exit_status
