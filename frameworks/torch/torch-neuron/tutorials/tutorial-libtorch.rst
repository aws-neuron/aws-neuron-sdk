.. _pytorch-tutorials-libtorch:

LibTorch C++ Tutorial
=========================

.. contents:: Table of Contents
   :local:
   :depth: 2


Overview
--------

This tutorial demonstrates the use of `LibTorch <https://pytorch.org/cppdocs/installing.html>`_ with Neuron, the SDK for Amazon Inf1, Inf2 and Trn1 instances. By the end of this tutorial, you will understand how to write a native C++ application that performs inference on EC2 Inf1, Inf2 and Trn1 instances. We will use an inf1.6xlarge and a pretrained BERT-Base model to determine if one sentence is a paraphrase of another.

Verify that this tutorial is running in a virtual environement that was set up according to the `Torch-Neuronx Installation Guide <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/setup/torch-neuronx.html#setup-torch-neuronx>` or `Torch-Neuron Installation Guide <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/setup/torch-neuron.html#setup-torch-neuron>`

Notes
-----

The tutorial has been tested on Inf1, Inf2 and Trn1 instances on ubuntu instances.


Run the tutorial
----------------

This tutorial is self contained.  It produces similar output to :ref:`[html] </src/examples/pytorch/bert_tutorial/tutorial_pretrained_bert.ipynb>` :pytorch-neuron-src:`[notebook] <bert_tutorial/tutorial_pretrained_bert.ipynb>`.

Note:  The tutorial will use about 8.5 GB of disk space.  Please ensure you have sufficient space before beginning.

Right-click and copy :download:`this link address to the tutorial archive</src/examples/pytorch/libtorch_demo.tar.gz>`.

.. code:: bash

  wget <paste archive URL>
  tar xvf libtorch_demo.tar.gz

Your directory tree should now look like this:

::

  libtorch_demo
  ├── bert_neuronx
  │   ├── compile.py
  │   └── detect_instance.py
  ├── clean.sh
  ├── core_count
  │   ├── build.sh
  │   └── main.cpp
  ├── example_app
  │   ├── build.sh
  │   ├── core_count.hpp
  │   ├── example_app.cpp
  │   ├── README.txt
  │   ├── utils.cpp
  │   └── utils.hpp
  ├── neuron.patch
  ├── run_tests.sh
  ├── setup.sh
  ├── tokenizer.json
  └── tokenizers_binding
      ├── build_python.sh
      ├── build.sh
      ├── remote_rust_tokenizer.h
      ├── run_python.sh
      ├── run.sh
      ├── tokenizer.json
      ├── tokenizer_test
      ├── tokenizer_test.cpp
      └── tokenizer_test.py

This tutorial uses the `HuggingFace Tokenizers <https://github.com/huggingface/tokenizers>`_ library implemented in Rust.
Install Cargo, the package manager for the Rust programming language.


 +----------------------------------+----------------------------------+
 | Ubuntu                           | AL2                              |
 +----------------------------------+----------------------------------+
 | .. code-block:: bash             | .. code-block:: bash             |
 |                                  |                                  |
 |    sudo apt install -y cargo   |    sudo yum install -y cargo   |
 +----------------------------------+----------------------------------+


Run the setup script to download additional depdendencies and build the app. (This may take a few minutes to complete.)

.. literalinclude:: tutorial_source_instructions/run_libtorch.sh
   :language: bash
   :lines: 6-7

::

  ...
  + g++ utils.cpp example_app.cpp -o ../example-app -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -I../libtorch/include -L../tokenizers_binding/lib -L/opt/aws/neuron/lib/ -L../libtorch/lib -Wl,-rpath,libtorch/lib -Wl,-rpath,tokenizers_binding/lib -Wl,-rpath,/opt/aws/neuron/lib/ -ltokenizers -ltorchneuron -ltorch_cpu -lc10 -lpthread -lnrt
  ~/libtorch_demo
  Successfully completed setup

.. _libtorch-benchmark:

Benchmark
---------

The setup script should have compiled and saved a PyTorch model compiled for neuron (bert_neuron_b6.pt).  Run the provided sanity tests to ensure everything is working properly.

.. literalinclude:: tutorial_source_instructions/run_libtorch.sh
   :language: bash
   :lines: 10

::

  Running tokenization sanity checks.

  None of PyTorch, TensorFlow >= 2.0, or Flax have been found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
  Tokenizing: 100%|██████████████████████████████████████████████████████████████████████████████████| 10000/10000 [00:00<00:00, 15021.69it/s]
  Python took 0.67 seconds.
  Sanity check passed.
  Begin 10000 timed tests.
  ..........
  End timed tests.
  C++ took 0.226 seconds.

  Tokenization sanity checks passed.
  Running end-to-end sanity check.

  The company HuggingFace is based in New York City
  HuggingFace's headquarters are situated in Manhattan
  not paraphrase: 10%
  paraphrase: 90%

  The company HuggingFace is based in New York City
  Apples are especially bad for your health
  not paraphrase: 94%
  paraphrase: 6%

  Sanity check passed.

Finally, run the example app directly to benchmark the BERT model.

.. note::

  You can safely ignore the warning about ``None of PyTorch, Tensorflow >= 2.0, ...``. This occurs because the test runs in a small virtual environment that doesn't require the full frameworks.

.. literalinclude:: tutorial_source_instructions/run_libtorch.sh
   :language: bash
   :lines: 13

::

  Getting ready................
  Benchmarking................
  Completed 32000 operations in 43 seconds => 4465.12 pairs / second
  
  ====================
  Summary information:
  ====================
  Batch size = 6
  Num neuron cores = 16
  Num runs per neuron core = 2000

**Congratulations!** By now you should have successfully built and used a native C++ application with LibTorch.

Troubleshooting
---------------

* In the event of SIGBUS errors you may have insufficient disk space for the creation of temporary model files at runtime.  Consider clearing space or mounting additional disk storage.
* In the event of a neuron runtime failure, confirm that the Neuron kernel module is loaded using ``sudo modprobe neuron``.

.. _libtorch-cleanup:


