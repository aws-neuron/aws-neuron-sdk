.. meta::
   :description: TorchTitan training tutorials on AWS Neuron. End-to-end walk-throughs for basic training, SimpleFSDP with Llama and DeepSeek, and GPT-OSS.
   :keywords: TorchTitan, TorchNeuron, AWS Neuron, Trainium, tutorial, Llama, DeepSeek, GPT-OSS, FSDP, SimpleFSDP
   :date-modified: 2026-10-08

.. _torchtitan-neuron-tutorials:

================================
TorchTitan training tutorials
================================

End-to-end walk-throughs for common TorchTitan pre-training workloads on
AWS Trainium.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card::
      :link: /training/torchtitan/tutorials/tutorial-basic-training
      :link-type: doc
      :class-card: sd-border-2

      **Basic training**
      :bdg-primary:`tutorial`
      ^^^
      A first, complete TorchTitan training run on Trainium.

   .. grid-item-card::
      :link: /training/torchtitan/tutorials/tutorial-simple-fsdp-training
      :link-type: doc
      :class-card: sd-border-2

      **SimpleFSDP training with Llama and DeepSeek**
      :bdg-primary:`tutorial`
      ^^^
      Compiler-based sharding for two real model families.

   .. grid-item-card::
      :link: /training/torchtitan/tutorials/tutorial-gpt-oss-training
      :link-type: doc
      :class-card: sd-border-2

      **GPT-OSS training**
      :bdg-primary:`tutorial`
      ^^^
      End-to-end GPT-OSS training with TorchTitan on Trainium.

.. toctree::
   :maxdepth: 1
   :hidden:

   Basic training <tutorial-basic-training>
   SimpleFSDP training with Llama and DeepSeek <tutorial-simple-fsdp-training>
   GPT-OSS training <tutorial-gpt-oss-training>
