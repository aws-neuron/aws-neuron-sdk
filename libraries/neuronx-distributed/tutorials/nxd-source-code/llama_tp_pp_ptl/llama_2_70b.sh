#!/bin/bash
set -eExuo

cd ~/neuronx-distributed/examples/training/llama/lightning
chmod +x run_llama_70b_tp_pp_ptl.sh
mkdir 70B_config
cp ~/neuronx-distributed/examples/training/llama/tp_pp_llama_hf_pretrain/70B_config_llama2/config.json ./70B_config
ln -sf 70B_config/config.json ./

sudo rm -rf /home/ubuntu/.cache/
pip install --upgrade filelock

python3 get_dataset.py --llama-version 2

PATH=$PATH:/opt/slurm/bin/

sbatch --exclusive \
--nodes 32 \
--cpus-per-task 128 \
--wrap="srun neuron_parallel_compile bash $(pwd)/run_llama_70b_tp_pp_ptl.sh"

sbatch --exclusive \
--nodes 32 \
--cpus-per-task 128 \
--wrap="srun bash $(pwd)/run_llama_70b_tp_pp_ptl.sh"