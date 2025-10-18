#!/bin/bash
set -eExuo

cd ~/neuronx-distributed/examples/training/tp_dp_gpt_neox_hf_pretrain/tp_dp_gpt_neox_6.9b_hf_pretrain/
ln -sf ~/neuronx-distributed/examples/training/tp_dp_gpt_neox_hf_pretrain/common/adamw_fp32_optim_params.py ./
ln -sf ~/neuronx-distributed/examples/training/tp_dp_gpt_neox_hf_pretrain/common/get_dataset.py ./
ln -sf ~/neuronx-distributed/examples/training/tp_dp_gpt_neox_hf_pretrain/common/requirements.txt ./
ln -sf ~/neuronx-distributed/examples/training/tp_dp_gpt_neox_hf_pretrain/tp_dp_gpt_neox_20b_hf_pretrain/modeling_gpt_neox_nxd.py ./
ln -sf ~/neuronx-distributed/examples/training/tp_dp_gpt_neox_hf_pretrain/tp_dp_gpt_neox_20b_hf_pretrain/utils.py ./
python3 -m pip install -r requirements.txt

python3 get_dataset.py

PATH=$PATH:/opt/slurm/bin/

sbatch --exclusive \
--nodes 4 \
--wrap="srun neuron_parallel_compile bash $(pwd)/tp_dp_gpt_neox_6.9b_hf_pretrain.sh"

sbatch --exclusive \
--nodes 4 \
--wrap="srun bash $(pwd)/tp_dp_gpt_neox_6.9b_hf_pretrain.sh"
