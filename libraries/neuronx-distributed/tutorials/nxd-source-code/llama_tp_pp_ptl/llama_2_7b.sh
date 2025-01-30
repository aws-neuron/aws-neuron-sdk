#!/bin/bash
set -eExuo

cd ~/neuronx-distributed/examples/training/llama/lightning
chmod +x run_llama_7b_tp_ptl.sh
mkdir 7B_config_llama2
cp ~/neuronx-distributed/examples/training/llama/tp_zero1_llama_hf_pretrain/7B_config_llama2/config.json ./7B_config_llama2
ln -sf 7B_config_llama2/config.json ./

sudo rm -rf /home/ubuntu/.cache/
pip install --upgrade filelock

python3 get_dataset.py --llama-version 2

PATH=$PATH:/opt/slurm/bin/

sbatch --exclusive \
--nodes 4 \
--cpus-per-task 128 \
--wrap="srun neuron_parallel_compile bash $(pwd)/run_llama_7b_tp_ptl.sh"

sbatch --exclusive \
--nodes 4 \
--cpus-per-task 128 \
--wrap="srun bash $(pwd)/run_llama_7b_tp_ptl.sh"