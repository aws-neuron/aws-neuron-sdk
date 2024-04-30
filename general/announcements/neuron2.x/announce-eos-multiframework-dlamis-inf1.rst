.. post:: April 24, 2024
    :language: en
    :tags: announce-eos-dlamis-inf1, dlami-inf1

.. _announce-update-multiframework-dlami:

Announcing end of support for Neuron virtual environments in AWS Deep Learning AMI (Amazon Linux 2)
----------------------------------------------------------------------------------------------------

:ref:`Neuron release 2.18.2 <neuron-2.18.2-whatsnew>` will be the last release that will include support for the following virtual environments in AWS Deep Learning AMI (Amazon Linux 2):

- ``aws_neuron_pytorch_p38: PyTorch 1.13, Python 3.8``
- ``aws_neuron_tensorflow2_p38: TensorFlow 2.10, Python 3.8``

Future releases will not include Neuron support for these virtual environments.

Current users of Neuron virtual environments in `AWS Deep Learning AMI (Amazon Linux 2) <https://aws.amazon.com/releasenotes/aws-deep-learning-ami-amazon-linux-2/>`_ are required to migrate to the `Neuron multi framework DLAMI <https://aws.amazon.com/releasenotes/aws-deep-learning-ami-neuron-ubuntu-22-04/>`_.

To see a list of Neuron supported virtual environments, please refer to :ref:`Neuron Multi Framework DLAMI User Guide <neuron-dlami-overview>`.
