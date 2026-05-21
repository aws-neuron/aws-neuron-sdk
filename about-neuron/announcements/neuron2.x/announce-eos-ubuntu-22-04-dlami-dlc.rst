.. post:: May 21, 2026
    :language: en
    :tags: announce-eos-ubuntu-22-04-dlami-dlc

.. _announce-eos-ubuntu-22-04-dlami-dlc:

Neuron no longer includes Ubuntu 22.04 DLAMIs and DLCs starting this release
-----------------------------------------------------------------------------

Starting with :ref:`Neuron Release 2.30.0 <neuron-2.30.0-whatsnew>`, Neuron no longer publishes Deep Learning AMIs (DLAMIs) and Deep Learning Containers (DLCs) built on Ubuntu 22.04. This follows the December 2025 announcement that single-framework DLAMIs for PyTorch and JAX on Ubuntu 22.04 would reach end of support in a future release (:ref:`announce-eos-dlami-ubuntu-22-04`).

Customers using Ubuntu 22.04 Neuron DLAMIs and DLCs should migrate to offerings based on Ubuntu 24.04. For customers who need to continue using Ubuntu 22.04, you can create custom AMIs based on the Ubuntu 22.04 base image and install Neuron components manually. See :ref:`container-faq` and :ref:`neuron-dlami-overview` for guidance.

This change does not affect support for the base Ubuntu 22.04 operating system. Ubuntu 22.04 LTS (Jammy Jellyfish) remains in standard Canonical support through April 2027, with Ubuntu Pro extending coverage through April 2032. AWS continues to support Ubuntu 22.04 through the standard DLAMI and EC2 lifecycles. The change above is specific to Neuron-published DLAMIs and DLCs.

For more information, see:

- :ref:`neuron-dlami-overview` for the current list of supported Neuron DLAMIs.
- :ref:`sdk-maintenance-policy` for the Neuron SDK maintenance policy.
- `AWS DLAMI Support Policy <https://docs.aws.amazon.com/dlami/latest/devguide/support-policy.html>`_ for the AWS-wide DLAMI support model.
