#!/bin/bash
set -eExuo

cd ~/neuronx-distributed/examples/training/llama/lightning
chmod +x run_llama_13b_tp_pp_ptl.sh
mkdir 13B_config
cp ~/neuronx-distributed/examples/training/llama/tp_pp_llama_hf_pretrain/13B_config_llama2/config.json ./13B_config


sudo rm -rf /home/ubuntu/.cache/
pip install --upgrade filelock

python3 get_dataset.py --llama-version 2

PATH=$PATH:/opt/slurm/bin/

sbatch --exclusive \
--nodes 32 \
--cpus-per-task 128 \
--wrap="srun neuron_parallel_compile bash $(pwd)/run_llama_13b_tp_pp_ptl.sh"

sbatch --exclusive \
--nodes 32 \
--cpus-per-task 128 \
--wrap="srun bash $(pwd)/run_llama_13b_tp_pp_ptl.sh"
