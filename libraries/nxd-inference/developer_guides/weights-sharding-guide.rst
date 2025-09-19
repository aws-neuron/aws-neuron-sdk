.. _nxdi-weights-sharding-guide:

NxD Inference Weights Sharding Guide
==========================================

NxD Inference provides two approaches to shard model weights and load them onto Neuron Devices, enabling parallel processing 
(e.g. Tensor Parallelism) on each device. This guide demonstrates the usage of both approaches using :ref:`nxdi-trn2-llama3.1-405b-speculative-tutorial`,
and provides insights into selecting the appropriate method based on the usage pattern and performance requirements.

.. note::

    Sharding speed on different storage volumes can vary. We recommend to use NVMe solid state drive (SSD) storage to achieve the best sharding performance.
    This guide shows sharding results on NVMe SSD. For more information about NVMe storage on EC2 instances, see the following:
    * `Instance store volumes <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/InstanceStorage.html>`__ in the Amazon EC2 User Guide. Instance store volumes are drives attached to EC2 instances that you can use for temporary storage. Neuron instances such as Trn1 and Trn2 include NVMe drives that you can use as instance store volumes.
    * `EBS volumes and NVMe <https://docs.aws.amazon.com/ebs/latest/userguide/nvme-ebs-volumes.html>`__ in the Amazon EBS User Guide. For persistent storage on NVMe, you can use EBS volumes built on AWS Nitro.


.. contents:: Table of contents
   :local:
   :depth: 1

Shard on compile (Pre-shard)
----------------------------

The shard on compile (pre-shard) approach loads the supported pretrained :ref:`checkpoints <nxdi-checkpoint-support>`, 
converts to Neuron compatible format, shards for each parallel rank and serializes sharded weights to disk as safetensors files. The entire sharding and serialization 
process can take a few minutes to hours depending on the model size and throughput of the storage volume. This approach is optimized to minimize the future model loading time.

The following example demonstrates how to run shard on compile with Llama3.1-405b.

First, complete the `prerequisites <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/tutorials/trn2-llama3.1-405b-speculative-tutorial.html#prerequisites>`__
for running Llama3.1-405b on a Trn2.48xlarge instance.

Next, enable shard on compile by adding ``--save-sharded-checkpoint`` to the command. The sharded checkpoints will be saved to the ``/weights`` folder under the specified ``COMPILED_MODEL_PATH``.

Full command to run shard on compile for Llama3.1-405b:
::

    # Replace this with the path where model files are downloaded.
    MODEL_PATH="/home/ubuntu/models/Llama-3.1-405B-Instruct/"
    # This is where the compiled model will be saved.
    COMPILED_MODEL_PATH="/home/ubuntu/traced_model/Llama-3.1-405B-Instruct/"

    NUM_CORES=128
    TP_DEGREE=64
    LNC=2

    export NEURON_RT_VIRTUAL_CORE_SIZE=$LNC
    export NEURON_RT_NUM_CORES=$((NUM_CORES/NEURON_RT_VIRTUAL_CORE_SIZE))
    export NEURON_RT_EXEC_TIMEOUT=600 

    inference_demo \
        --model-type llama \
        --task-type causal-lm \
            run \
            --model-path $MODEL_PATH \
            --compiled-model-path $COMPILED_MODEL_PATH \
            --torch-dtype bfloat16 \
            --start_rank_id 0 \
            --local_ranks_size $TP_DEGREE \
            --tp-degree $TP_DEGREE \
            --batch-size 1 \
            --max-context-length 2048 \
            --seq-len 2048 \
            --on-device-sampling \
            --top-k 1 \
            --fused-qkv \
            --sequence-parallel-enabled \
            --qkv-kernel-enabled \
            --attn-kernel-enabled \
            --mlp-kernel-enabled \
            --cc-pipeline-tiling-factor 1 \
            --pad-token-id 2 \
            --save-sharded-checkpoint \
            --prompt "What is annapurna labs?" 2>&1 | tee log

You should see the outputs below in your logs. The duration can slightly vary between runs. Note that model loading started only after sharding is completed. 

::

    INFO:Neuron:Sharding Weights for ranks: 0...63
    INFO:Neuron:Done sharding weights in 1856.5586961259833 seconds
    Loading model to Neuron...
    Total model loading time: 107.76132441597292 seconds

Now that sharded checkpoints have been serialized to disk, you may save sharding time in your next run by adding ``--skip-sharding`` to the command.
Sharded weights will be directly loaded from the disk for inference, which saves you 30+ minutes of sharding for each subsequent run in this example.

The total model loading time in each subsequent run is expected to be comparable with the first run.


Shard on load
------------------

.. warning::
    At high batch size (>=32), we have observed performance degradation with ``shard-on-load`` for some models such as Llama3.1-8B. If you observe worse inference performance with ``shard-on-load``, please disable this feature (by enabling the ``--save-sharded-checkpoint`` flag during compilation with ``inference_demo`` as above).
    Alternatively, if you are not using ``inference_demo``, you can also enable ``save_sharded_checkpoint`` directly in ``NeuronConfig`` which will be passed to model init when the model is traced and compiled.

The shard on load approach significantly reduces sharding overheads by parallelizing tensor movement in sharding/loading and skipping sharded checkpoints serialization. 
This approach is preferred when you are working with weights that are frequently retrained/fine-tuned so re-sharding becomes a bottleneck when serving with new weights.
Since Neuron 2.23 release, Shard on load is enabled by default in NxD Inference.

Full command to run shard on load for Llama3.1-405b is shown below. Note that ``--save-sharded-checkpoint`` is excluded from the command.
::

    # Replace this with the path where model files are downloaded.
    MODEL_PATH="/home/ubuntu/models/Llama-3.1-405B-Instruct/"
    # This is where the compiled model will be saved.
    COMPILED_MODEL_PATH="/home/ubuntu/traced_model/Llama-3.1-405B-Instruct/"

    NUM_CORES=128
    TP_DEGREE=64
    LNC=2

    export NEURON_RT_VIRTUAL_CORE_SIZE=$LNC
    export NEURON_RT_NUM_CORES=$((NUM_CORES/NEURON_RT_VIRTUAL_CORE_SIZE))
    export NEURON_RT_EXEC_TIMEOUT=600 

    inference_demo \
        --model-type llama \
        --task-type causal-lm \
            run \
            --model-path $MODEL_PATH \
            --compiled-model-path $COMPILED_MODEL_PATH \
            --torch-dtype bfloat16 \
            --start_rank_id 0 \
            --local_ranks_size $TP_DEGREE \
            --tp-degree $TP_DEGREE \
            --batch-size 1 \
            --max-context-length 2048 \
            --seq-len 2048 \
            --on-device-sampling \
            --top-k 1 \
            --fused-qkv \
            --sequence-parallel-enabled \
            --qkv-kernel-enabled \
            --attn-kernel-enabled \
            --mlp-kernel-enabled \
            --cc-pipeline-tiling-factor 1 \
            --pad-token-id 2 \
            --prompt "What is annapurna labs?" 2>&1 | tee log

You should see the outputs below in your logs. The duration can slightly vary between runs. Note that sharding happened while model was being loaded (i.e. shard on load).

::

    Loading model to Neuron...
    INFO:Neuron:Done Sharding weights in 49.31190276599955 seconds
    Total model loading time: 187.3972628650372 seconds

As you can see, weights sharding of shard on load is much faster than that of shard on compile.

When the current run finishes, no sharded checkpoints will be saved. Therefore, you cannot use ``--skip-sharding`` for your next run. 
In each subsequent run, NxD Inference will do the exact same amount of sharding work, so the total model loading time is expected to be 
comparable with the first run. It's also expected that the total model loading time is longer than that of shard on compile, due to the extra
sharding work it has to do during loading time.
