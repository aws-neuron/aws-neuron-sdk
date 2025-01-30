.. _llama2_tp_pp_ptl_tutorial:

Training Llama-2-7B/13B/70B using Tensor Parallelism and Pipeline Parallelism with Neuron PyTorch-Lightning 
=========================================================================================

In this section, we showcase to pretrain a Llama2 7B/13B/70B with Tensor Parallelism and Pipeline Parallel using Neuron PyTorch-Lightning APIs, please refer to :ref:`Llama2 7B Tutorial <llama2_7b_tp_zero1_tutorial>`, :ref:`Llama2 13B/70B Tutorial <llama2_tp_pp_tutorial>`
and :ref:`Neuron PT-Lightning Developer Guide <_ptl_developer_guide>` for more context.


Setting up environment:
^^^^^^^^^^^^^^^^^^^^^^^
                       
For this experiment, we will use AWS ParallelCluster with at least four trn1.32xlarge compute nodes(at least 32 nodes are needed for 13B/70B model size).
`Train your model on ParallelCluster <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/devflows/training/parallelcluster/parallelcluster-training.html>`__
introduces how to setup and use a ParallelCluster.
To setup the packages on the headnode of the ParallelCluster, follow the instructions mentioned here:
:ref:`Install PyTorch Neuron on Trn1 <setup-torch-neuronx>`.

We also need to install the ``neuronx-distributed`` package inside the virtual env using the following command:

.. code:: ipython3

   python -m pip install neuronx_distributed --extra-index-url https://pip.repos.neuron.amazonaws.com
   git clone git@github.com:aws-neuron/neuronx-distributed.git

Let’s download the scripts for pretraining:


1. Navigate to a directory to hold our experiments

.. literalinclude:: nxd-source-code/llama_tp_pp_ptl/llama_tp_pp_ptl_setup.sh
   :language: shell
   :lines: 4

2. Link the training scripts for our experiments

.. literalinclude:: nxd-source-code/llama_tp_pp_ptl/llama_tp_pp_ptl_setup.sh
   :language: shell
   :lines: 5-10

If you want to pre-train Llama 7B, you would need to run the following steps -

.. literalinclude:: nxd-source-code/llama_tp_pp_ptl/llama_2_7b.sh
   :language: shell
   :lines: 5-8

If you want to pre-train Llama 13B, you would need to run the following steps -

.. literalinclude:: nxd-source-code/llama_tp_pp_ptl/llama_2_13b.sh
   :language: shell
   :lines: 5-8

If you want to pre-train Llama 70B, you would need to run the following steps -

.. literalinclude:: nxd-source-code/llama_tp_pp_ptl/llama_2_70b.sh
   :language: shell
   :lines: 5-8

3. Installing the additional requirements and giving the right permissions to our shell script

.. literalinclude:: nxd-source-code/llama_tp_pp_ptl/llama_tp_pp_ptl_setup.sh
   :language: shell
   :lines: 12-13


Next, we tokenize our dataset. 
``Note``: To tokenize the data, we must request the tokenizer from `HuggingFace` and `Meta` by following 
the instructions at the following link: `HuggingFace Llama 2 7B Model <https://huggingface.co/meta-llama/Llama-2-7b>`__ .
Use of the Llama 2 model is governed by the Meta license. In order to download the model weights and tokenizer, please 
visit the above website and accept their License before requesting access. After access has been granted, 
you may use the download scripts provided by Meta to download the model weights and tokenizer to your cluster.

Once you have downloaded the tokenizer and model weights, you can copy the ``tokenizer.model`` to the ``~/examples/llama2_lightning`` directory.

Next let’s download and pre-process the dataset:

.. literalinclude:: nxd-source-code/llama_tp_pp_ptl/llama_2_7b.sh
   :language: shell
   :lines: 13

``Note``: In case you see an error of the following form when downloading data: ``huggingface_hub.utils._validators.HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/home/ubuntu/examples/llama2_lightning'. Use `repo_type` argument if needed.`` 
This could be because of a stale cache. Try deleting the cache using: 

.. code:: ipython3

   sudo rm -rf /home/ubuntu/.cache/


At this point, you are all set to start training.

Training Llama2-7B with Tensor Parallelism
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By this step, the ParallelCluster is all setup for running experiments. 
Before we run training, we first pre-compile the graphs using the :ref:`neuron_parallel_compile <pytorch-neuronx-parallel-compile-cli>`.
Let’s run the command below:

.. literalinclude:: nxd-source-code/llama_tp_pp_ptl/llama_2_7b.sh
   :language: shell
   :lines: 17-20

This script uses a tensor-parallel size of 8.
This will automatically set the zero-1 sharding degree to 16 (4 * 32 workers / tensor_parallel_size). 

``Note``: You can use any number of nodes in this case, would just need to adjust the number of nodes in the above 
slurm command accordingly. Also, the number of nodes used in parallel_compile command should be same as the actual 
training run. This is because, as the number of nodes change, the data-parallel degree would change too. This would 
result in more workers participating in operations like `gradient all-reduce` which would result in new graphs getting 
created. 

Once the graphs are compiled we can now run training and observe our loss goes down.
To run the training, we just run the above command but without ``neuron_parallel_compile``.

.. literalinclude:: nxd-source-code/llama_tp_pp_ptl/llama_2_7b.sh
   :language: shell
   :lines: 22-25

Training Llama2-13B/70B with Tensor Parallelism and Pipeline Parallelism
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here we use ``Llama70B`` as an example. To run 13B, simply change the script from ``run_llama_70b_tp_pp.sh`` to ``run_llama_13B_tp_pp.sh``
Before we run training, we first pre-compile the graphs using the :ref:`neuron_parallel_compile <pytorch-neuronx-parallel-compile-cli>`.
Let’s run the command below:

Pre-compiling

.. literalinclude:: nxd-source-code/llama_tp_pp_ptl/llama_2_70b.sh
   :language: shell
   :lines: 17-20

This script uses a tensor-parallel size of 8, pipeline-parallel size of 8
To run the training, we just use the above command but without ``neuron_parallel_compile``.

.. literalinclude:: nxd-source-code/llama_tp_pp_ptl/llama_2_7b.sh
   :language: shell
   :lines: 22-25


Checkpointing:
^^^^^^^^^^^^^^

To enable checkpoint saving, add following flags to ``run_llama_7b_tp_ptl.sh``/ ``run_llama_13b_tp_pp.sh`` /  ``run_llama_70B_tp_pp.sh``:
* ``--save_checkpoint`` Add this flag to enable checkpoint saving
* ``--checkpoint_freq`` Number of steps to save a checkpoint
* ``--checkpoint_dir`` Direction to save the checkpoint 
* ``--num_kept_checkpoint`` Number of checkpoints to save, older checkpoint will be deleted manually, set to -1 to keep all saved checkpoints
* ``--save_load_xser`` load with torch xla serialization to reduce time saving, it's recommended to enable xser for significantly faster save/load. Note that if the chekpoint is saved with xser, it can only be loaded with xser, vice versa. 

To enable checkpoint loading, add following flags to ``run_llama_7b_tp_ptl.sh``/ ``run_llama_13b_tp_pp.sh`` /  ``run_llama_70B_tp_pp.sh``:
* ``--resume_ckpt`` 
* ``--load_step`` Step to retrieve checkpoint from
* ``--checkpoint_dir`` Direction to load the checkpoint from
* ``--save_load_xser`` load with torch xla serialization to reduce time saving, it's recommended to enable xser for significantly faster save/load. Note that if the chekpoint is saved with xser, it can only be loaded with xser, vice versa. 
