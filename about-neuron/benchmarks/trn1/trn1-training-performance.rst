.. _trn1-training-performance:

Trn1/Trn1n Training Performance
===============================

This section provides benchmark results for training various deep learning models on AWS Trn1 and Trn1n instances powered by AWS Trainium chips. The benchmarks cover a range of model architectures, including encoder models, decoder models, and vision transformers, demonstrating the performance capabilities of Trn1/Trn1n instances for different training workloads.

**Last update: February 19th, 2026**

.. contents:: Table of contents
   :local:

.. _NLP:

Encoder Models
--------------

.. csv-table::
   :file: training_data_encoder.csv
   :header-rows: 1


Decoder Models
--------------

.. csv-table::
   :file: training_data_decoder.csv
   :header-rows: 1

.. note::
   **TP (Tensor Parallel), PP (Pipeline Parallel) and DP (Data Parallel) Topology** configuration refers to the degrees of 3D Parallelism (How the model and data is sharded across NeuronCores).

   TP and PP are specified in the run script and DP is calculated by dividing **world size** (Number of nodes/instances * Number of neuron cores per instance) by TP * PP degrees.

   For example, ``TP = 4, PP = 4`` and`` Number of instances is 32 (trn1.32xlarge)``. The world size will be: ``32(num instances) * 32(Neuron Cores per instance) = 1024``. Now, ``DP degree = 1024 (World size)/ 4 (TP) * 4 (PP) = 64``

   For more information on batch sizes please refer to :ref:`neuron-batching`


Vision Transformer Models
--------------------------

.. csv-table::
   :file: training_data_vision_transformers.csv
   :header-rows: 1

.. note::
   Read more about strong vs weak scaling here :ref:`neuron-training-faq`
