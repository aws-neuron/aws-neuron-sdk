.. _neuron_hw_glossary:

Neuron Glossary
===============

.. contents:: Table of contents
   :local:
   :depth: 2


Terms
-----

Neuron Devices (Accelerated Machine Learning chips)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left
      

   * - Term
     - Description

   * - .. glossary::
          Inferentia
     - AWS first generation accelerated machine learning chip supporting inference only

   * - .. glossary::
          Trainium
     - AWS second generation accelerated machine learning chip supporting training and inference

   * - .. glossary::
          Neuron Device
     - Accelerated machine learning chip (e.g. Inferentia or Trainium)

Neuron powered Instances
^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left
      

   * - Term
     - Description


   * - .. glossary::
          Inf1
     - Inferentia powered accelerated compute EC2 instance

   * - .. glossary::
          Trn1
     - Trainium powered accelerated compute EC2 instance


NeuronCore terms
^^^^^^^^^^^^^^^^


.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left
      

   * - Term
     - Description


   * - .. glossary::
          NeuronCore
     - The machine learning compute cores within Inferentia/Trainium

   * - .. glossary::
          NeuronCore-v1
     - Neuron Core withing Inferentia

   * - .. glossary::
          NeuronCore-v2
     - Neuron Core withing Trainium

   * - .. glossary::
          Tensor Engine
     - 2D systolic array (within the NeuronCore), used for matrix computations

   * - .. glossary::
          Scalar Engine
     - A scalar-engine within each NeuronCore, which can accelerate element-wise operations (e.g. GELU, ReLU, reciprocal, etc)

   * - .. glossary::
          Vector Engine
     - A vector-engine with each NeuronCore, which can accelerate spatial operations (e.g. layerNorm, TopK, pooling, etc)

   * - .. glossary::
          GPSIMD Engine
     - Embedded General Purpose SIMD cores, within each NeuronCore, to accelerate custom-operators

   * - .. glossary::
          Sync Engine
     - The SP engine, which is integrated inside NeuronCore. Used for synchronization and DMA triggering.

   * - .. glossary::
          Collective Communication Engine
     - Dedicated engine for collective communication, allows for overlapping computation and communication

   * - .. glossary::
          NeuronLink
     - Interconnect between NeuronCores

   * - .. glossary::
          NeuronLink-v1
     - Interconnect between NeuronCores in Inferentia device

   * - .. glossary::
          NeuronLink-v2
     - Interconnect between NeuronCores in Trainium device


Abbreviations
-------------

.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left
      

   * - Abbreviation
     - Description


   * - .. glossary::
          NC
     - Neuron Core

   * - .. glossary::
          NeuronCore
     - Neuron Core
     
   * - .. glossary::
          ND
     - Neuron Device

   * - .. glossary::
          NeuronDevice
     - Neuron Device

   * - .. glossary::
          TensEng
     - Tensor Engine

   * - .. glossary::
          ScalEng
     - Scalar Engine

   * - .. glossary::
          VecEng
     - Vector Engine

   * - .. glossary::
          SyncEng
     - Sync Engine

   * - .. glossary::
          CCE
     - Collective Communication Engine

   * - .. glossary::
          FP32
     - Float32

   * - .. glossary::
          TF32
     - TensorFloat32

   * - .. glossary::
          FP16
     - Float16

   * - .. glossary::
          BF16
     - Bfloat16

   * - .. glossary::
          cFP8
     - Configurable Float8

   * - .. glossary::
          RNE
     - Round Nearest Even

   * - .. glossary::
          SR
     - Stochastic Rounding

   * - .. glossary::
          CustomOps
     - Custom Operators

   * - .. glossary::
          RT
     - Neuron Runtime

   * - .. glossary::
          DP
     - Data Parallel

   * - .. glossary::
          DPr
     - Data Parallel degree

   * - .. glossary::
          TP
     - Tensor Parallel

   * - .. glossary::
          TPr
     - Tensor Parallel degree

   * - .. glossary::
          PP
     - Pipeline Parallel

   * - .. glossary::
          PPr
     - Pipeline Parallel degree

