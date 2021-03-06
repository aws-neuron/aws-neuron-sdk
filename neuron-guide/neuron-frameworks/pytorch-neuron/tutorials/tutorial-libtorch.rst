.. _pytorch-tutorials-libtorch:

LibTorch C++ Tutorial
=========================

.. contents:: Table of Contents
   :local:
   :depth: 2


Overview
--------

This tutorial demonstrates the use of `LibTorch <https://pytorch.org/cppdocs/installing.html>`_ with Neuron, the SDK for Amazon Inf1 instances. By the end of this tutorial, you will understand how to write a native C++ application that performs inference on EC2 Inf1 instances. We will use an inf1.6xlarge and a pretrained BERT-Base model to determine if one sentence is a paraphrase of another.

.. note::

  Model compilation can be executed on a non-inf1 instance for later deployment. Follow the same :ref:`EC2 Developer Flow Setup <ec2-then-ec2-devflow>` using other instance families and leverage Amazon Simple Storage Service (S3) to share the compiled models between different instances.

This tutorial is divided into the following parts:

* :ref:`libtorch-setup` - Steps needed to setup the compilation and deployment environments that will enable you to run this tutorial. In this tutorial, a single inf1 instance will provide both the compilation and deployment enviroments.
* :ref:`libtorch-run` - Steps needed to compile and run the example app.
* :ref:`libtorch-benchmark` - Steps needed to benchmark the model.
* :ref:`libtorch-cleanup` - Steps needed to cleanup instance used for this tutorial.


.. _libtorch-setup:

Setup the Enviornment
-----------------------------------------

Launch Inf1 instance by following the below steps, and make sure to choose an inf1.6xlarge instance running the DLAMI.

.. include:: /neuron-intro/install-templates/launch-inf1-dlami.rst


.. _libtorch-run:

Run the Tutorial
----------------

After connecting to the instance from the terminal, clone the Neuron Github repository to the EC2 instance and then change the working directory to the tutorial directory:

.. code:: bash

  git clone https://github.com/aws/aws-neuron-sdk.git
  cd aws-neuron-sdk/src/examples/pytorch


The Jupyter notebook is available as a file with the name :pytorch-neuron-src:`tutorial_pretrained_bert.ipynb <bert_tutorial/tutorial_pretrained_bert.ipynb>`, you can either run the Jupyter notebook from a browser or run it as a script from terminal:


* **Running tutorial from browser**

  * First setup and launch the Jupyter notebook on your local browser by following instructions at :ref:`Running Jupyter Notebook Browser`
  * Open the Jupyter notebook from the menu and follow the instructions


You can also view the Jupyter notebook at:

.. toctree::
   :maxdepth: 1

   /src/examples/pytorch/bert_tutorial/tutorial_pretrained_bert.ipynb


You should now have a compiled ``bert_neuron_b6.pt`` file, which is required going forward.
Right-click and copy :download:`this link address to the tutorial archive</src/libtorch_demo.tar.gz>`.

.. code:: bash

  $ wget <paste archive URL>
  $ tar xvf libtorch_demo.tar.gz

Your directory tree should now look like this:

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

After you've finished with the instance/s that you created for this tutorial, you should clean up by terminating the instance/s, follow instructions at `Clean up your instance <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html#ec2-clean-up-your-instance>`_.
