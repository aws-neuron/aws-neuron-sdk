.. _pytorch-tutorials-libtorch:

LibTorch C++ Tutorial
=========================

.. contents:: Table of Contents
   :local:
   :depth: 2


Overview
--------

This tutorial demonstrates the use of `LibTorch <https://pytorch.org/cppdocs/installing.html>`_ with Amazon Inferentia hardware and the Neuron SDK. By the end of this tutorial, you will understand how to write a native C++ application that performs inference on EC2 Inf1 instances. We will use a pretrained BERT-Base model to determine if one sentence is a paraphrase of another.

.. note::

  Model compilation can be executed on a non-inf1 instance for later deployment. Follow the same EC2 Developer Flow Setup using other instance families and leverage Amazon Simple Storage Service (S3) to share the compiled models between different instances.

This tutorial is divided into the following parts:

* :ref:`libtorch-setup` - Steps needed to setup the compilation and deployment environments that will enable you to run this tutorial. In this tutorial, a single inf1 instance will provide both the compilation and deployment enviroments.
* :ref:`libtorch-run` - Follow the steps to compile and run the example app.
* :ref:`libtorch-benchmark` - Benchmark the compiled model.
* :ref:`libtorch-cleanup` - After running the tutorial, make sure to cleanup instance/s used for this tutorial.


.. _libtorch-setup:

Setup the Enviornment
-----------------------------------------

Please launch Inf1 instance by following the below steps, and make sure to choose an inf1.6xlarge instance.

.. include:: /neuron-intro/install-templates/launch-inf1-dlami.rst


.. _libtorch-run:

Run the Tutorial
----------------

Complete the `HuggingFace Pretrained BERT Tutorial <https://github.com/aws/aws-neuron-sdk/blob/master/src/examples/pytorch/bert_tutorial/tutorial_pretrained_bert.ipynb>`_. You should now have a compiled ``bert_neuron_b6.pt`` file, which is required going forward.


Open a shell on the instance you prepared earlier. Download and extract the tutorial archive.

.. code:: bash

  $ wget https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/neuron-frameworks/pytorch-neuron/tutorials/libtorch/libtorch_demo.tar.gz
  $ tar xvf libtorch_demo.tar.gz

Your directoy tree should now look like this:

::

  .
  ├── bert_neuron_b6.pt
  ├── libtorch_demo
  │   ├── example_app
  │   │   ├── CMakeLists.txt
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

.. code:: bash

  $ sudo apt install -y cargo

Verify you are using a compatible version of Python.

.. code:: bash

  $ python3 --version

::

  Python 3.7.6

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

  $ ./example-app bert_neuron_b6.pt

::

  Getting ready....
  Benchmarking....
  Completed 4000 operations in 25 seconds => 960 pairs / second

  ====================
  Summary information:
  ====================
  Batch size = 6
  Num neuron cores = 4
  Num runs per neruon core = 1000

**Congratulations!** By now you should have successfully built and used a native C++ application with LibTorch.

.. _libtorch-cleanup:

Clean up your instance/s
------------------------

After you've finished with the instance/s that you created for this tutorial, you should clean up by terminating the instance/s, please follow instructions at `Clean up your instance <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html#ec2-clean-up-your-instance>`_.
