.. _install-neuron-tensorflow:

Install Neuron Tensorflow
=========================

Please select your choice of environment, please note that `AWS Deep Learning AMI (DLAMI) <https://docs.aws.amazon.com/dlami/index.html>`_ is 
the recommended AMI to use with Neuron SDK.

- :ref:`tensorflow-ubuntu-dlami`
- :ref:`tensorflow-al2-dlami`
- :ref:`tensorflow-pip-ubuntu`
- :ref:`tensorflow-pip-al2`
- :ref:`tensorflow-conda`

Refer to the `The AWS Inferentia Chip With DLAMI <https://docs.aws.amazon.com/dlami/latest/devguide/tutorial-inferentia.html>`__
guide to learn how to use the DLAMI with Neuron.

.. _tensorflow-ubuntu-dlami:

Ubuntu DLAMI
------------------

Update Packages:
~~~~~~~~~~~~~~~~

.. include:: /neuron-intro/install-templates/ubuntu-dlami-update.rst

Activate Neuron Tensorflow
~~~~~~~~~~~~~~~~~~~~~~~~

.. include:: /neuron-intro/install-templates/dlami-enable-neuron-tensorflow.rst


.. _tensorflow-al2-dlami:

Amazon Linux DLAMI
--------------------------

Update Packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. include:: /neuron-intro/install-templates/al2-dlami-update.rst

Activate Neuron Tensorflow
~~~~~~~~~~~~~~~~~~~~~~~~

.. include:: /neuron-intro/install-templates/dlami-enable-neuron-tensorflow.rst


.. _tensorflow-pip-ubuntu:

Ubuntu 16,18 AMI (via Pip)
------------------------------

Configuring Linux for repository updates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: /neuron-intro/install-templates/ubuntu-pip-install.rst

Install Neuron Pip Packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: /neuron-intro/install-templates/neuron-pip-install.rst

Install TensorFlow
^^^^^^^^^^^^^^^^^^

.. include:: /neuron-intro/install-templates/tensorflow-pip-install.rst

.. _tensorflow-pip-al2:

Amazon Linux2 AMI (via Pip)
--------------------------------

Configuring Linux for repository updates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: /neuron-intro/install-templates/al2-pip-install.rst

Install Neuron Pip Packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: /neuron-intro/install-templates/neuron-pip-install.rst

Install TensorFlow
^^^^^^^^^^^^^^^^^^

.. include:: /neuron-intro/install-templates/tensorflow-pip-install.rst

.. _tensorflow-conda:

Ubuntu or Amazon Linux AMI (via Conda)
--------------------------------------

.. include:: /neuron-intro/install-templates/conda-install.rst
.. include:: /neuron-intro/install-templates/tensorflow-conda-install.rst
.. include:: /neuron-intro/install-templates/conda-neuron-cc.rst
.. include:: /neuron-intro/install-templates/conda-tensorflow-info.rst
