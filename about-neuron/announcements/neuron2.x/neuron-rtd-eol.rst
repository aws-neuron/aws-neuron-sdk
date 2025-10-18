.. post:: Oct 10, 2022 01:00
    :language: en
    :tags: eol, neuron2.x

.. _announce-neuron-rtd-eol:

Announcing Neuron Runtime 1.x (``neuron-rtd``) end-of-support
-------------------------------------------------------------

Starting with :ref:`Neuron release 2.3 <neuron-2.3.0-whatsnew>`, Neuron components like Neuron System Tools
and Neuron Driver will no longer support Neuron Runtime 1.x.

In addition, starting with :ref:`Neuron release 2.3 <neuron-2.3.0-whatsnew>`, the `AWS Neuron Runtime Proto GitHub <https://github.com/aws-neuron/aws-neuron-runtime-proto>`_  and `AWS Neuron Driver GitHub <https://github.com/aws-neuron/aws-neuron-driver>`_ repositories will no longer be supported.

Why are we removing support for Neuron Runtime 1.x?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Neuron Runtime 1.x (``neuron-rtd``) entered :ref:`maintenance mode <maintenance_rtd>` when Neuron 1.16.0 
was released. While Neuron components like Neuron Driver and Neuron System Tools continued to support 
Neuron Runtime 1.x in addition to supporting Neuron Runtime 2.x, Neuron supported frameworks (e.g. PyTorch Neuron,
TensorFlow Neuron, and MXNet Neuron) stopped supporting Neuron Runtime 1.x starting with Neuron 1.16.0. 
For detailed information see :ref:`introduce-libnrt`.
