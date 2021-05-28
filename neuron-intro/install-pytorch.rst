.. _install-neuron-pytorch:

Install Neuron PyTorch
======================

.. _pytorch-pip-ubuntu:

Ubuntu 18 AMI (via Pip)
---------------------------

.. warning::

   :ref:`Starting with Neuron 1.14.0, Ubuntu 16 is no longer supported <eol-ubuntu16>`

Configuring Linux for repository updates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: /neuron-intro/install-templates/ubuntu-pip-install.rst

Install Neuron Pip Packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: /neuron-intro/install-templates/neuron-pip-install.rst

Install PyTorch
^^^^^^^^^^^^^^^

.. include:: /neuron-intro/install-templates/pytorch-pip-install.rst

.. _pytorch-pip-al2:

Amazon Linux2 AMI (via Pip)
-----------------------------

Configuring Linux for repository updates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: /neuron-intro/install-templates/al2-pip-install.rst

Install Neuron Pip Packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: /neuron-intro/install-templates/neuron-pip-install.rst

Install PyTorch
^^^^^^^^^^^^^^^

.. include:: /neuron-intro/install-templates/pytorch-pip-install.rst

.. _pytorch-conda:

Ubuntu or Amazon Linux2 AMI (via Conda)
---------------------------------------

.. warning::

   :ref:`Starting with Neuron 1.14.0, Neuron Conda packages in Deep Learning AMI are no longer supported<eol-conda-packages>`, for more information see `blog announcing the end of support for Neuron conda packages <https://aws.amazon.com/blogs/developer/neuron-conda-packages-eol/>`_ 

.. include:: /neuron-intro/install-templates/conda-install.rst
.. include:: /neuron-intro/install-templates/pytorch-conda-install.rst
.. include:: /neuron-intro/install-templates/conda-neuron-cc.rst
