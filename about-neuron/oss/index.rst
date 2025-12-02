.. meta::
    :description: GitHub repositories for AWS Neuron open source components, libraries, and tools.
    :date-modified: 12/02/2025

Neuron Open Source Repositories and Contribution
===================================================

AWS Neuron provides open source code and samples for some of its components, libraries, and tools under the Apache 2.0 license. The current public repositories open to contribution at this time are listed below.

Neuron Open Source GitHub Repositories
---------------------------------------

.. grid:: 1
   :gutter: 3

   .. grid-item-card:: 
      :class-body: sphinx-design-class-title-small
 
      **TorchNeuron PyTorch Extension Open Source**
      ^^^
      Source code for the Neuron Native PyTorch extension and the TorchNeuron library that implements it for AWS Trainium.

      * Neuron GitHub source repository: https://github.com/aws-neuron/torch-neuronx

   .. grid-item-card:: 
      :class-body: sphinx-design-class-title-small
 
      **NKI Compiler and Language Open Source**
      ^^^
      Source code for the NKI Compiler and the NKI languages and APIs.

      * Neuron GitHub source repository (NKI Compiler): https://github.com/aws-neuron/nki-compiler 
      * Neuron GitHub source repository (NKI APIs): https://github.com/aws-neuron/nki

   .. grid-item-card:: 
      :class-body: sphinx-design-class-title-small
 
      **Neuron Kernel Library Open Source**
      ^^^
      Source code and specifications for the pre-built kernels that ship with the NKI Library .

      * Neuron GitHub source repository: https://github.com/aws-neuron/nki-library
  
   .. grid-item-card:: 
      :class-body: sphinx-design-class-title-small
 
      **vLLM for Neuron Open Source**
      ^^^
      Source code for the vLLM integrations with Neuron, supporting AWS Trainium and Inferentia.

      * Neuron GitHub source repository: https://github.com/vllm-project/vllm-neuron
      * **Note**: Released under vLLM project license (`LICENSE <https://github.com/vllm-project/vllm-neuron/blob/main/LICENSE>`__). 
  
   .. grid-item-card:: 
      :class-body: sphinx-design-class-title-small
 
      **NKI Samples**
      ^^^
      Full code examples that support NKI kernel development.

      * Neuron GitHub source repository: https://github.com/aws-neuron/nki-samples

How to Contribute to Neuron Open Source
----------------------------------------

Contributions via pull requests are appreciated! Before sending us a pull request, please ensure that:

1. You are working against the latest source on the `main`` branch.
2. You check existing open and recently merged pull requests and GitHub Issues to make sure someone else hasn't addressed the problem already.
3. You open a GitHub Issue for the repo to discuss any significant work.

To send us a pull request:

1. Fork the repository.
2. Modify the source; please focus on the specific change you are contributing. If you also reformat all the code, it will be hard for us to focus on your change.
3. Ensure local tests pass.
4. Commit to your fork using clear commit messages.
5. Send us a pull request, answering any default questions in the pull request interface.
6. Pay attention to any automated CI failures reported in the pull request, and stay involved in the conversation.

GitHub provides documentation on `forking a repository <https://help.github.com/articles/fork-a-repo/>`_ and `creating a pull request <https://help.github.com/articles/creating-a-pull-request/>`_.

For the specific details on licenses and contributing to each OSS repo, review the ``CONTRIBUTING.md`` pages linked below:

* Contribute to TorchNeuron: https://github.com/aws-neuron/torch-neuronx/blob/main/CONTRIBUTING.md
* Contribute to the NKI Library: https://github.com/aws-neuron/nki-library/blob/main/CONTRIBUTING.md
* Contribute to the NKI Compiler and NKI APIs: https://github.com/aws-neuron/nki/blob/main/CONTRIBUTING.md
* Contribute the the NKI samples: https://github.com/aws-neuron/nki-samples/blob/main/CONTRIBUTING.md
  
.. Re-add this when available: * Contribute to vLLM Neuron: https://github.com/vllm-project/vllm-neuron/blob/main/CONTRIBUTING.md
