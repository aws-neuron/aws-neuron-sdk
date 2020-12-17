.. _install-neuron-mxnet:

Install Neuron MXNet
====================

Please select your choice of environment, please note that `AWS Deep Learning AMI (DLAMI) <https://docs.aws.amazon.com/dlami/index.html>`_ is 
the recommended AMI to use with Neuron SDK.

- :ref:`mxnet-ubuntu-dlami`
- :ref:`mxnet-al2-dlami`
- :ref:`mxnet-pip-ubuntu`
- :ref:`mxnet-pip-al2`
- :ref:`mxnet-conda`

Refer to the `The AWS Inferentia Chip With DLAMI <https://docs.aws.amazon.com/dlami/latest/devguide/tutorial-inferentia.html>`__
guide to learn how to use the DLAMI with Neuron.

.. _mxnet-ubuntu-dlami:

Ubuntu DLAMI
-------------

Update Packages:
~~~~~~~~~~~~~~~~

.. include:: /neuron-intro/install-templates/ubuntu-dlami-update.rst

Activate Neuron MXNet
~~~~~~~~~~~~~~~~~~~

.. include:: /neuron-intro/install-templates/dlami-enable-neuron-mxnet.rst


.. _mxnet-al2-dlami:

Amazon Linux DLAMI
---------------------


Update Packages:
~~~~~~~~~~~~~~~~~~

.. include:: /neuron-intro/install-templates/al2-dlami-update.rst

Activate Neuron MXNet
~~~~~~~~~~~~~~~~~~~~~

.. include:: /neuron-intro/install-templates/dlami-enable-neuron-mxnet.rst



.. _mxnet-pip-ubuntu:


Ubuntu 16,18 AMI (via Pip)
-------------------------

Configuring Linux for repository updates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: /neuron-intro/install-templates/ubuntu-pip-install.rst

Install Neuron Pip Packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: /neuron-intro/install-templates/neuron-pip-install.rst

Install MXNet
^^^^^^^^^^^^^

.. include:: /neuron-intro/install-templates/mxnet-pip-install.rst

.. _mxnet-pip-al2:

Amazon Linux2 AMI (via Pip)
---------------------------

Configuring Linux for repository updates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: /neuron-intro/install-templates/al2-pip-install.rst

Install Neuron Pip Packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: /neuron-intro/install-templates/neuron-pip-install.rst

Install MXNet
^^^^^^^^^^^^^

.. include:: /neuron-intro/install-templates/mxnet-pip-install.rst

.. _mxnet-conda:

Ubuntu or Amazon Linux AMI (via Conda)
-----------------------------------

.. include:: /neuron-intro/install-templates/conda-install.rst
.. include:: /neuron-intro/install-templates/mxnet-conda-install.rst
.. include:: /neuron-intro/install-templates/conda-neuron-cc.rst



