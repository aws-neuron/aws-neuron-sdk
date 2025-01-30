#!/bin/bash
set -eExuo

cd ~/neuronx-distributed/examples/training/llama/tp_zero1_llama_hf_pretrain
chmod +x tp_zero1_llama2_7B_hf_pretrain.sh
ln -sf 7B_config_llama2/config.json ./

sudo rm -rf /home/ubuntu/.cache/
pip install --upgrade filelock

python3 get_dataset.py --llama-version 2

PATH=$PATH:/opt/slurm/bin/

sbatch --exclusive \
--nodes 4 \
--cpus-per-task 128 \
--wrap="srun neuron_parallel_compile bash $(pwd)/tp_zero1_llama2_7B_hf_pretrain.sh"

sbatch --exclusive \
--nodes 4 \
--cpus-per-task 128 \
--wrap="srun bash $(pwd)/tp_zero1_llama2_7B_hf_pretrain.sh"
