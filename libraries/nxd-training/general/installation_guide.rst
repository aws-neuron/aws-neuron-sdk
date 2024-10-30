.. _nxdt_installation_guide:

Setup
=====

Neuronx Distributed Training framework is built on top of
`NeuronxDistributed (NxD) <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/index.html>`_ ,
`NeMo <https://github.com/NVIDIA/NeMo/tree/v1.14.0>`_ libraries and
`PyTorch-Lightning <https://github.com/Lightning-AI/pytorch-lightning/tree/1.8.6>`_. The guide below will provide
a step-by-step instructions on how to setup the environment to run training using NeuronX Distributed Training
framework.

.. contents:: Table of contents
   :local:
   :depth: 2


.. _nxdt_python_venv:

Setup a python Virtual Environment
----------------------------------

Let's first setup a virtual env for our development. This can be done using the command below:

.. code-block :: shell

    python3 -m venv env
    source env/bin/activate

.. _nxdt_neuron_deps:

Installing Neuron Dependencies
------------------------------

Install the neuron packages using the command:

.. code-block :: shell

    pip install --upgrade neuronx-cc==2.* torch-neuronx torchvision neuronx_distributed --extra-index-url https://pip.repos.neuron.amazonaws.com

.. _nxdt_nemo_deps:

Building Apex
-------------

Since NxD Training is built on top of NeMo, we have to install its dependencies too. One of which is the
`Apex <https://github.com/NVIDIA/apex/tree/master>`_ library. NeMo uses it for few of the fused module implementations.

.. note::
    NeMo used to use Apex for all distributed training APIs. Since we are using NxD for the same purpose, the use of
    Apex for this framework is very minimal. It's been added as a dependency since some of the minor imports inside NeMo
    will break without it. Hence, when building Apex, we build a slim CPU version using the instructions below:

1. Clone Apex repo

.. code-block :: shell

    git clone https://github.com/ericharper/apex.git ~/
    cd apex
    git checkout nm_v1.14.0


2. Replace the contents of the ``setup.py`` with the following contents:

.. code-block :: python

    import sys
    import warnings
    import os
    from packaging.version import parse, Version

    from setuptools import setup, find_packages
    import subprocess

    import torch
    from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME, load

    setup(
        name="apex",
        version="0.1",
        packages=find_packages(
            exclude=("build", "csrc", "include", "tests", "dist", "docs", "tests", "examples", "apex.egg-info",)
        ),
        install_requires=["packaging>20.6",],
        description="PyTorch Extensions written by NVIDIA",
    )

3. Install python dependencies:

.. code-block :: shell

    pip install packaging wheel


4. Build the wheel using the command:

.. code-block :: shell

    python setup.py bdist_wheel


5. After this, you should see the wheel at ``dist/``. You can use this for installation in the next section.
6. Come out of the ``apex`` directory using ``cd ..``.


.. _nxdt_nxdt_reqs:

Installing the requirements
---------------------------

Download the ``requirements.txt`` using the command:

.. code-block :: shell

    wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed-training/master/requirements.txt

We can now install the dependencies of the library using the following command:

.. code-block :: shell

    pip install -r requirements.txt ~/apex/dist/apex-0.1-py3-none-any.whl


.. _nxdt_nxdt_nxdt_install:

Installing Neuronx Distributed Training framework
-------------------------------------------------

To install the library, one can run the following command:

.. code-block :: shell

    pip install neuronx_distributed_training --extra-index-url https://pip.repos.neuron.amazonaws.com


.. _nxdt_installation_common_failures:

Common failures during installation
-----------------------------------

This section goes over the common failures one can see during setup and how to resolve them.

1. **``ModuleNotFoundError: No module named 'Cython'``**

   You may have to install Cython explicitly using ``pip install Cython``

2. **Error while building ``youtokentome``**

   If you get an error that says ``Python.h file not found``, you may have to install python-dev and recreate the
   virtual env. To install python-dev, you can use the command: ``sudo apt-get install python-dev``

3. **Mismatched torch and torch-xla version**

   When you see an error that looks like:

::

    ImportError: env/lib/python3.10/site-packages/_XLAC.cpython-310-x86_64-linux-gnu.so: undefined symbol: _ZN3c109TupleTypeC1ESt6vectorINS_4Type24SingletonOrSharedTypePtrIS2_EESaIS4_EENS_8optionalINS_13QualifiedNameEEESt10shared_ptrINS_14FunctionSchemaEE

   It indicates that the major versions of torch and torch-xla don't match.

.. note::
    If you install torch again, make sure to install the corresponding torchvision version else that would have
    a conflict.

4. **Torch vision version error**

   The below error indicates incorrect torchvision version. If installing ``torch=2.1``, install ``torchvision=0.16``
   (This `link <https://pypi.org/project/torchvision/>`_ shows which version of torchvision is compatible with
   which version of torch).

::

    ValueError: Could not find the operator torchvision::nms. Please make sure you have already registered the operator
    and (if registered from C++) loaded it via torch.ops.load_library.`

5. **Matplotlib lock error**

   If you see the below error:

::

    TimeoutError: Lock error: Matplotlib failed to acquire the following lock file

   This error means there is some contention in compute/worker nodes to access the matlotlib cache, and hence the timeout
   error. To resolve this error, add or run ``python -c 'import matplotlib.pyplot as plt'`` command as part of your setup.
   This will create a matplotlib cache and avoid the race condition.



