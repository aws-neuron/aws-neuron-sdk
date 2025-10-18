.. _jax-neuron-setup:

JAX NeuronX plugin Setup
------------------------------

The JAX NeuronX plugin is a set of modularized JAX plugin packages integrating
AWS Trainium and Inferentia machine learning accelerators into JAX as pluggable
devices. It includes the following Python packages, all hosted on the AWS Neuron
pip repository.

* ``libneuronxla``: A package containing Neuron's integration into JAX's runtime `PJRT <https://openxla.org/xla/pjrt_integration>`__, built using the `PJRT C-API plugin <https://github.com/openxla/xla/blob/5564a9220af230c6c194e37b37938fb40692cfc7/xla/pjrt/c/docs/pjrt_integration_guide.md>`__ mechanism. Installing this package enables using Trainium and Inferentia natively as JAX devices.
* ``jax-neuronx``: A package containing Neuron-specific JAX features, such as the `Neuron NKI <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/nki_rn.html>`__ JAX interface. It also serves as a meta-package for providing a tested combination of the ``jax-neuronx``, ``jax``, ``jaxlib``, ``libneuronxla``, and ``neuronx-cc`` packages. Making proper use of the features provided in ``jax-neuronx`` will unleash the full potential of Trainium and Inferentia.

.. include:: /setup/install-templates/trn1-ga-warning.txt

.. note:: 
    JAX requires ``Python 3.10`` or newer. Ensure a supported python version is installed on your system prior to installing JAX.

.. dropdown::  Launch the Instance
    :class-title: sphinx-design-class-title-small
    :class-body: sphinx-design-class-body-small
    :animate: fade-in

    * To launch an instance, follow the instructions at `launch an Amazon EC2 Instance <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html#ec2-launch-instance>`_. Make sure to select the correct instance type on the EC2 console.
    * For more information about instance sizes and pricing, see `Amazon EC2 Trn1 Instances <https://aws.amazon.com/ec2/instance-types/trn1/>`_ and `Amazon EC2 Inf2 Instances <https://aws.amazon.com/ec2/instance-types/inf2/>`_
    * Select Ubuntu Server 22 AMI.
    * When launching a Trn1, adjust your primary EBS volume size to a minimum of 512GB.
    * After launching the instance, follow the instructions in `Connect to your instance <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstancesLinux.html>`_ to connect to the instance.

.. dropdown::  Install Drivers and Tools
    :class-title: sphinx-design-class-title-small
    :class-body: sphinx-design-class-body-small
    :animate: fade-in

    Ubuntu

    .. include:: /src/helperscripts/installationScripts/python_instructions.txt
        :start-line: 242
        :end-line: 243

    Amazon Linux 2023

    .. include:: /src/helperscripts/installationScripts/python_instructions.txt
        :start-line: 239
        :end-line: 240

.. dropdown::  Install the JAX NeuronX Plugin
    :class-title: sphinx-design-class-title-small
    :class-body: sphinx-design-class-body-small
    :animate: fade-in

    We provide two methods for installing the JAX NeuronX plugin. The first is to install
    the ``jax-neuronx`` meta-package from the AWS Neuron pip repository. This method provides
    a production-ready JAX environment where ``jax-neuronx``'s major dependencies, namely
    ``jax``, ``jaxlib``, ``libneuronxla``, and ``neuronx-cc``, have undergone thorough testing
    by the AWS Neuron team and will have their versions pinned during installation.

    .. code:: bash

        python3 -m pip install jax-neuronx[stable] --extra-index-url=https://pip.repos.neuron.amazonaws.com

    The second is to install packages ``jax``, ``jaxlib``, ``libneuronxla``,
    and ``neuronx-cc`` separately, with ``jax-neuronx`` being an optional addition.
    Because ``libneuronxla`` supports a broad range of ``jaxlib`` versions through
    the PJRT C-API mechanism, this method provides flexibility when choosing
    ``jax`` and ``jaxlib`` versions, enabling JAX users to bring the JAX NeuronX plugin
    into their own JAX environments.

    .. code:: bash

        python3 -m pip install jax==0.4.38 jaxlib==0.4.38
        python3 -m pip install jax-neuronx libneuronxla neuronx-cc==2.* --extra-index-url=https://pip.repos.neuron.amazonaws.com

We can now run some simple JAX programs on the Trainium or Inferentia
accelerators.

.. code:: bash

   ~$ python3 -c 'import jax; print(jax.numpy.multiply(1, 1))'
   Platform 'neuron' is experimental and not all JAX functionality may be correctly supported!
   .
   Compiler status PASS
   1

Compatibility between packages ``jaxlib`` and ``libneuronxla`` can be
determined from `PJRT C-API
version <https://github.com/openxla/xla/blob/0d1b60216ea13b0d261d59552a0f7ef20c4f76c5/xla/pjrt/c/pjrt_c_api.h>`__.
For more information, see `PJRT integration
guide <https://github.com/openxla/xla/blob/0d1b60216ea13b0d261d59552a0f7ef20c4f76c5/docs/pjrt/pjrt_integration.md>`__.

To determine compatible JAX versions, you can use the
``libneuronxla.supported_clients`` API for querying known supported
client packages and their versions.

.. code::

   Help on function supported_clients in module libneuronxla.version:

   supported_clients()
       Return a description of supported client (jaxlib, torch-xla, etc.) versions,
       as a list of strings formatted as `"<package> <version> (PJRT C-API <c-api version>)"`.
       For example,
       >>> import libneuronxla
       >>> libneuronxla.supported_clients()
       ['jaxlib 0.4.38 (PJRT C-API 0.58)', torch_xla 2.6.0 (PJRT C-API 0.55)', 'torch_xla 2.6.1 (PJRT C-API 0.55)', 'torch_xla 2.7.0 (PJRT C-API 0.61)']

Note that the list of supported client packages and versions covers
known versions only and may be incomplete. More versions could be
supported, including Google's future ``jaxlib`` releases, assuming the
PJRT C-API stays compatible with the current release of
``libneuronxla``. As a result, we avoid specifying any dependency
relationship between ``libneuronxla`` and ``jaxlib``. This provides more
freedom when coordinating ``jax`` and ``libneuronxla`` installations.
