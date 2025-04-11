#!/bin/bash
set -eExuo

aws s3 cp --no-progress s3://neuron-s3/training_checkpoints/pytorch/dp_bert_large_hf_pretrain/ckpt_28125.pt ~/aws-neuron-samples/torch-neuronx/training/dp_bert_hf_pretrain/output/ckpt_28125.pt --no-sign-request

cd ~/aws-neuron-samples/torch-neuronx/training/dp_bert_hf_pretrain
torchrun --nproc_per_node=32 dp_bert_large_hf_pretrain_hdf5.py \
    --data_dir ~/examples_datasets/bert_pretrain_wikicorpus_tokenized_hdf5_seqlen512/ \
    --lr 2.8e-4 \
    --phase2 \
    --resume_ckpt \
    --phase1_end_step 28125 \
    --batch_size 2 \
    --grad_accum_usteps 512 \
    --seq_len 512 \
    --max_pred_len 80 \
    --warmup_steps 781 \
    --max_steps 1563 \
    | tee run_pretrain_log_phase2.txt
torchrun_exit_status=${PIPESTATUS[0]}
echo "Training return code: $torchrun_exit_status"
exit $torchrun_exit_status
