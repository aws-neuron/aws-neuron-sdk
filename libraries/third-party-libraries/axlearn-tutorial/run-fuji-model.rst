.. _fuji_v3_tiktoken:

Fuji 8B v3 titoken Pretraining with AXLearn
===========================================
In this tutorial, we will setup the virtual enviroment and pretrain a ``fuji-8B-v3-tiktoken``, a similar model with ``Llama 3.1 8B``, on trn2. Comparing to Llama 3.1 8B, fuji-8B-v3-tiktoken has a shorter context window of 8k tokens.

.. contents:: Table of contents
   :local:
   :depth: 2

How AXLearn intergrates with JAX and Neuron
-------------------------------------------
Consider `JAX <https://docs.jax.dev/en/latest/quickstart.html>`_ as a powerful engine, and `AXLearn <https://github.com/apple/axlearn>`_ is the car build around it.
AXLearn now has native support for `Neuron <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/index.html>`_, which means everything Neuron needs is already checked in to AXLearn.

Setting up the environment
--------------------------
Install runtime dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    # Needed for TC_MALLOC fix
    sudo apt-get -f install -y
    sudo apt-get install -y google-perftools libgoogle-perftools-dev

    # Install Runtime dependencies
    sudo dpkg -i aws-neuronx-runtime-lib-2.x.117.0-19b96f9e2.deb
    sudo dpkg -i aws-neuronx-collectives-2.x.24763.0-7eb7b60be.deb
    sudo dpkg -i aws-neuronx-dkms_2.x.4788.0_amd64.deb

Create virtual environment
^^^^^^^^^^^^^^^^^^^^^^^^^^
Create a new directory for this tutorial and move to that directory.

.. code-block:: bash

    # create the venv
    python3 -m venv axlearn_env

    # activate the venv
    source axlearn_env/bin/activate

Install python dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^
In the venv, we need to install jax packages TODO check released versions

.. code-block:: bash

    pip install jax_neuronx-0.1.3-py3-none-any.whl 
    pip install libneuronxla-2.2.721.0-py3-none-linux_x86_64.whl
    pip install neuronx_cc-2.0.122062.0a0+f7e4515f-cp310-cp310-linux_x86_64.whl

Git clone the AXLearn package from github

.. code-block:: bash

    git clone https://github.com/apple/axlearn.git

Install dependencies that AXLearn requires

.. code-block:: bash

    pip install -e axlearn/[core]
    pip install tokenizers # only needed for fuji v3

Update PJRT for compiler so that it does not change numpy and other conflicting dependencies TODO check released versions

.. code-block:: bash

    pip install --no-deps --force-reinstall jax_neuronx-0.1.3-py3-none-any.whl
    pip install --no-deps --force-reinstall libneuronxla-2.2.721.0-py3-none-linux_x86_64.whl
    pip install --no-deps --force-reinstall neuronx_cc-2.0.122062.0a0+f7e4515f-cp310-cp310-linux_x86_64.whl

Install necessary packages
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block::

    # Needed for downloading data from gcp
    sudo apt-get -f install -y
    sudo apt-get install -y google-perftools libgoogle-perftools-dev

Configure the fuji series model
-----------------------------------
Setup flags of training the model:

.. code-block::

    python -m axlearn.common.launch_trainer_main \
        --module=text.gpt.c4_trainer \
        --config=fuji-8B-v3-tiktoken \
        --trainer_dir="artifacts/${JOB_ID}/axlearn_out \
        --data_dir="gs://axlearn-public/tensorflow_datasets" \
        --jax_backend=neuron \
        --mesh_selector=neuron-trn2.48xlarge-64 \
        --distributed_coordinator=$MASTER_ADDR:$JAX_COORDINATOR_PORT \
        --num_processes=$num_nodes \
        --process_id=$NEURON_PJRT_PROCESS_INDEX

Download the llama 3.1 tokenizer from hugging face `here <https://huggingface.co/meta-llama/Llama-3.1-8B/blob/main/tokenizer.json>`_, and save it in your trn2 instance. If you find the file does not exist, you need to login and request for model access on huggingface.
Hardcode the tokenizer path: add the following line to axlearn/axlearn/experiments/text/gpt/vocabulary_fuji_v3.py line 106

.. code-block::

    filename = 'your/path/to/Llama3.1-8B-tokenizer.json'

Configure environment variables
-------------------------------
XLA flags
^^^^^^^^^
.. code-block::

    export XLA_FLAGS="--xla_dump_hlo_as_text"
    export XLA_FLAGS=${XLA_FLAGS} --xla_disable_hlo_passes=aws_neuron_flip_all_gather_dot,neuron-hierarchical-collectives
    export XLA_FLAGS=${XLA_FLAGS} --xla_dump_hlo_pass_re='.*'

PJRT flags
^^^^^^^^^^
.. code-block::

    export NEURON_FSDP_NUM_LAYER_EARLY_AG_SHIFT=1
    export NEURON_FSDP_NUM_LAYER_LATE_RS_SHIFT=2
    export NEURON_ENABLE_INT_MATMUL_DOWNCAST=1
    export NEURON_FSDP=1
    export NEURON_FSDP_NUM_LAYER_COALESCE=-1
    export NEURON_RUN_TRIVIAL_COMPUTATION_ON_CPU=1

Neuron runtime flags
^^^^^^^^^^^^^^^^^^^^
.. code-block::

    export NEURON_RT_DBG_CC_DMA_PACKET_SIZE=4096
    export NEURON_RT_DBG_DMA_PACKETIZATION_SIZE=104857
    export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=1
    export NEURON_RT_IO_RING_CACHE_SIZE=0
    export NEURON_RT_ENABLE_MEMORY_METRICS=0
    export NEURON_RT_VIRTUAL_CORE_SIZE=2
    export NEURON_RT_RESET_CORES=1
    export NEURON_RT_LOG_LEVEL="WARNING"
    export NEURON_RT_ENABLE_INTERNODE_EXECUTION_BARRIER=1


Neuron collective flags
^^^^^^^^^^^^^^^^^^^^^^^
.. code-block::

    export FI_LOG_LEVEL="warn"
    export OFI_NCCL_PROTOCOL=RDMA
    export LD_LIBRARY_PATH="/opt/amazon/efa/lib/"
    export FI_EFA_USE_DEVICE_RDMA="1"
    export FI_PROVIDER="efa"
    export FI_EFA_FORK_SAFE=1
    export OFI_NCCL_MR_CACHE_DISABLE=1


Neuron compiler flags
^^^^^^^^^^^^^^^^^^^^^
.. code-block::

    export NEURON_CC_FLAGS="--framework=XLA"
    export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --internal-max-instruction-limit=20000000"
    export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --target=trn2"
    export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --internal-num-neuroncores-per-sengine=2"
    export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --model-type transformer"
    export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --no-internal-hlo-remat"
    export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --enable-mixed-precision-accumulation"
    export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} -O1"
    export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --tensorizer-options='--enable-hoist-fsdp-collectives'"
    export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --internal-hlo2tensorizer-options='--remat-rope'"
    


The training script
-------------------

.. code-block::

    #!/usr/bin/env bash
    
    num_nodes=1 # entrer the number of nodes you launches
    devices_per_node=64
    MASTER_ADDR=$(hostname) # suppose you're running the script on the head node
    MASTER_PORT=41000
    JAX_COORDINATOR_PORT=41001
    export NEURON_RT_ROOT_COMM_ID="${MASTER_ADDR}:${MASTER_PORT}"
    export NEURON_PJRT_PROCESSES_NUM_DEVICES=$(printf '%s,' $(seq 1 $num_nodes | xargs -I {} echo $devices_per_node) | sed 's/,$//')
    export NEURON_PJRT_PROCESS_INDEX=0 # $SLUM_ID

    
    JOB_ID="your_job_id"
    ARTIFACTS_PATH="artifacts"
    TEST_ARTIFACTS_PATH="${ARTIFACTS_PATH}/${JOB_ID}"
    mkdir -p "$TEST_ARTIFACTS_PATH"

    # HLO dump
    HLO_DUMP_PATH=${TEST_ARTIFACTS_PATH}/hlo_dump
    export XLA_FLAGS=${XLA_FLAGS} --xla_dump_to=${HLO_DUMP_PATH}


    # Neuron dump
    NEURON_DUMP_PATH=${TEST_ARTIFACTS_PATH}/neuron_dump
    export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --dump=${NEURON_DUMP_PATH}"
    
    # JAX Cache
    export JAX_COMPILATION_CACHE_DIR="${ARTIFACTS_PATH}/${JOB_ID}/cache"
    mkdir -p ${JAX_COMPILATION_CACHE_DIR}

    source axlearn_env/bin/activate


    echo "Listing apt dependencies"
    apt list --installed | grep neuron
    echo "Listing pip dependencies"
    pip list | grep neuron
    echo "Done listing dependencies"
    printenv | grep NEURON
    printenv | grep XLA
    which python

    # TC MALLOC HACK
    LIBTCMALLOC=$(find /usr/lib/x86_64-linux-gnu -name "libtcmalloc.so.*" | sort -V | tail -n 1)
    
    if [ -n "$LIBTCMALLOC" ]; then
        # Create a symbolic link to the found libtcmalloc version
        sudo ln -sf "$LIBTCMALLOC" /usr/lib/libtcmalloc.so
        echo "Symbolic link created: /usr/lib/libtcmalloc.so -> $LIBTCMALLOC"
    
        # Export LD_PRELOAD
        export LD_PRELOAD=/usr/lib/libtcmalloc.so
        echo "LD_PRELOAD set to: $LD_PRELOAD"
    else
        echo "Error: libtcmalloc.so not found"
        exit 1
    fi

    OUTPUT_DIR="${TEST_ARTIFACTS_PATH}/axlearn_out"
    mkdir -p ${OUTPUT_DIR}
    DATA_DIR="gs://axlearn-public/tensorflow_datasets"

    python -m axlearn.common.launch_trainer_main \
        --module=text.gpt.c4_trainer \
        --config=fuji-8B-v3-tiktoken-flash-single-host \
        --trainer_dir=$OUTPUT_DIR \
        --data_dir=$DATA_DIR \
        --jax_backend=neuron \
        --mesh_selector=neuron-trn2.48xlarge-64 \
        --distributed_coordinator=$MASTER_ADDR:$JAX_COORDINATOR_PORT \
        --num_processes=$num_nodes \
        --process_id=$NEURON_PJRT_PROCESS_INDEX \

Visualizing HLOs
----------------

`Model explorer <https://github.com/google-ai-edge/model-explorer>`_ is a really useful tool to understand HLOs during development. To get started, install these packages. First is the model explorer itself, and second is an adapter for it to understand XLA HLOs.

.. code-block::

    pip install ai-edge-model-explorer
    pip install git+https://github.com/rahul003/hlo_adapter.git@main

Make sure the hlo files are dumped in proto format, check the XLA flags, if you'd like to view the HLOs in text, change --xla_dump_hlo_as_proto to --xla_dump_hlo_as_text.

.. code-block::

    export XLA_FLAGS=--xla_dump_hlo_as_proto --xla_dump_to=${HLO_DUMP_PATH} --xla_dump_hlo_pass_re='.*'

Rename \*.hlo.pb file to \*.hlo as the original pb adapter gets picked which doesn't work for these HLOs. (.pb is used by a different default plugin.) Or you can run this helper script:

.. code-block::

    ext=".hlo.pb"
    HLO_DIR=artifacts/${job_id}/hlo_dump
    packaged_hlos_dir=$HLO_DIR/packaged
    mkdir -p $packaged_hlos_dir

    before_prop=$(ls $HLO_DIR/*pjit__train_step*before_sharding-propagation${ext})
    after_prop=$(ls $HLO_DIR/*pjit__train_step*after_sharding-propagation.before_spmd-partitioning${ext})
    after_spmd_partitioning=$(ls $HLO_DIR/*pjit__train_step*after_spmd-partitioning*${ext})
    final=$(ls $HLO_DIR/*pjit__train_step*aggressive-optimizations.after_dce.before_pipeline-end${ext})
    cp $before_prop $packaged_hlos_dir/before_prop.hlo
    cp $after_prop $packaged_hlos_dir/after_prop.hlo
    cp $after_spmd_partitioning $packaged_hlos_dir/after_spmd_partitioning.hlo
    cp $final $packaged_hlos_dir/final.hlo

Start explorer and load the \*.hlo files.

.. code-block::

    model-explorer --extensions=hlo_adapter

Or start explorer with a given HLO directly

.. code-block::

    model-explorer --extensions=hlo_adapter path/to/your/*.hlo

Checkpoints
-----------

Profiling
---------
`Neuron Profile User Guide <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-sys-tools/neuron-profile-user-guide.html#overview>`_ provides a good introduction to Neuron Profiling.
For installation use `this <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-sys-tools/neuron-profile-user-guide.html#id3>`_.
Here's some basic command to get started with profiling:

.. code-block::

    # capture the profile
    neuron-profile capture -r <number_of_ranks> -n </path/to/file.neff>

    # view the profile
    neuron-profile view -n file.neff -s profile.ntff --output-format perfetto
