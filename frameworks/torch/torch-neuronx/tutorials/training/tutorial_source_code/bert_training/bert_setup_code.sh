#!/bin/bash
set -eExuo

# Install the required Python packages
python3 -m pip install -r ~/aws-neuron-samples/torch-neuronx/training/dp_bert_hf_pretrain/requirements.txt

# Create a directory for the datasets and download the datasets
mkdir -p ~/examples_datasets/
pushd ~/examples_datasets/
aws s3 cp --no-progress s3://neuron-s3/training_datasets/bert_pretrain_wikicorpus_tokenized_hdf5/bert_pretrain_wikicorpus_tokenized_hdf5_seqlen128.tar .  --no-sign-request
tar -xf bert_pretrain_wikicorpus_tokenized_hdf5_seqlen128.tar
rm bert_pretrain_wikicorpus_tokenized_hdf5_seqlen128.tar
aws s3 cp --no-progress s3://neuron-s3/training_datasets/bert_pretrain_wikicorpus_tokenized_hdf5/bert_pretrain_wikicorpus_tokenized_hdf5_seqlen512.tar .  --no-sign-request
tar -xf bert_pretrain_wikicorpus_tokenized_hdf5_seqlen512.tar
rm bert_pretrain_wikicorpus_tokenized_hdf5_seqlen512.tar
popd