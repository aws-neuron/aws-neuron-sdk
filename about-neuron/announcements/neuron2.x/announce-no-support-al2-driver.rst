.. post:: October 16, 2026
    :language: en
    :tags: announce-no-support-al2-driver

.. _announce-no-support-al2-driver:

Neuron no longer supports Amazon Linux 2 (AL2) starting with Neuron 2.33.0
==========================================================================

Starting with :ref:`Neuron Release 2.33.0 <neuron-2-33-0-whatsnew>`, Neuron no longer
supports Amazon Linux 2 (AL2). The Neuron Driver (``aws-neuronx-dkms``), previously the
last Neuron package to support AL2, no longer supports AL2 as of this release. Amazon
Linux 2 reached end of life on June 30, 2026.

Customers using Amazon Linux 2 must migrate to Amazon Linux 2023 (AL2023). For supported
operating systems and migration options, see :ref:`neuron-dlami-overview`. For Amazon
Linux 2 end-of-life details, see the `Amazon Linux 2 FAQs
<https://aws.amazon.com/amazon-linux-2/faqs/>`_.
