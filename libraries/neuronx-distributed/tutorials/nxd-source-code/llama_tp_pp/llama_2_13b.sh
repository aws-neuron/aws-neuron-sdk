#!/bin/bash
set -eExuo

cd ~/neuronx-distributed/examples/training/llama/tp_pp_llama_hf_pretrain

chmod +x run_llama2_13B_tp_pp.sh
ln -sf 13B_config_llama2/config.json ./

sudo rm -rf /home/ubuntu/.cache/
pip install --upgrade filelock

python3 get_dataset.py --llama-version 2

PATH=$PATH:/opt/slurm/bin/

sbatch --exclusive \
--nodes 32 \
--cpus-per-task 128 \
--wrap="srun neuron_parallel_compile bash $(pwd)/run_llama2_13B_tp_pp.sh"

sbatch --exclusive \
--nodes 32 \
--cpus-per-task 128 \
--wrap="srun bash $(pwd)/run_llama2_13B_tp_pp.sh"