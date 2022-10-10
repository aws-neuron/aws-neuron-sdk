.. _pytorch-tutorials-libtorch:

LibTorch C++ Tutorial
=========================

.. contents:: Table of Contents
   :local:
   :depth: 2


Overview
--------

This tutorial demonstrates the use of `LibTorch <https://pytorch.org/cppdocs/installing.html>`_ with Neuron, the SDK for Amazon Inf1 instances. By the end of this tutorial, you will understand how to write a native C++ application that performs inference on EC2 Inf1 instances. We will use an inf1.6xlarge and a pretrained BERT-Base model to determine if one sentence is a paraphrase of another.


Run the tutorial
----------------

First run the HuggingFace Pretrained BERT tutorial :ref:`[html] </src/examples/pytorch/bert_tutorial/tutorial_pretrained_bert.ipynb>` :pytorch-neuron-src:`[notebook] <bert_tutorial/tutorial_pretrained_bert.ipynb>`.


You should now have a compiled ``bert_neuron_b6.pt`` file, which is required going forward.
Right-click and copy :download:`this link address to the tutorial archive</src/examples/pytorch/libtorch_demo.tar.gz>`.

.. code:: bash

  $ wget <paste archive URL>
  $ tar xvf libtorch_demo.tar.gz

Your directory tree should now look like this:

::

  .
  ├── bert_neuron_b6.pt
  ├── libtorch_demo
  │   ├── example_app
  │   │   ├── README.txt
  │   │   ├── build.sh
  │   │   ├── example_app.cpp
  │   │   ├── utils.cpp
  │   │   └── utils.hpp
  │   ├── neuron.patch
  │   ├── run_tests.sh
  │   ├── setup.sh
  │   └── tokenizers_binding
  │       ├── build.sh
  │       ├── build_python.sh
  │       ├── remote_rust_tokenizer.h
  │       ├── run.sh
  │       ├── run_python.sh
  │       ├── tokenizer_test
  │       ├── tokenizer_test.cpp
  │       └── tokenizer_test.py
  └── libtorch_demo.tar.gz

Copy the compiled model from Step 2 into the new ``libtorch_demo`` directory.

.. code:: bash

  $ cp bert_neuron_b6.pt libtorch_demo/

This tutorial uses the `HuggingFace Tokenizers <https://github.com/huggingface/tokenizers>`_ library implemented in Rust.
Install Cargo, the package manager for the Rust programming language.


 +----------------------------------+----------------------------------+
 | Ubuntu                           | AL2                              |
 +----------------------------------+----------------------------------+
 | .. code-block:: bash             | .. code-block:: bash             |
 |                                  |                                  |
 |    $ sudo apt install -y cargo   |    $ sudo yum install -y cargo   |
 +----------------------------------+----------------------------------+


Run the setup script to download additional depdendencies and build the app. (This may take a few minutes to complete.)

.. code:: bash

  $ cd libtorch_demo
  $ chmod +x setup.sh && ./setup.sh

::

  ...
  [100%] Built target example_app
  make[1]: Leaving directory '/home/ubuntu/libtorch_demo/example_app/build'
  /usr/local/lib/python3.6/dist-packages/cmake/data/bin/cmake -E cmake_progress_start /home/ubuntu/libtorch_demo/example_app/build/CMakeFiles 0
  ~/libtorch_demo/example_app
  ~/libtorch_demo
  Successfully completed setup


.. _libtorch-benchmark:

Benchmark
---------

Run the provided sanity tests to ensure everything is working properly.

.. code:: bash

  $ ./run_tests.sh bert_neuron_b6.pt

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

.. code:: bash

  $ LD_LIBRARY_PATH="libtorch/lib:tokenizers_binding/lib" ./example-app bert_neuron_b6.pt

::

  Getting ready....
  Benchmarking....
  Completed 4000 operations in 22 seconds => 1090.91 pairs / second

  ====================
  Summary information:
  ====================
  Batch size = 6
  Num neuron cores = 4
  Num runs per neruon core = 1000

**Congratulations!** By now you should have successfully built and used a native C++ application with LibTorch.

.. _libtorch-cleanup:


