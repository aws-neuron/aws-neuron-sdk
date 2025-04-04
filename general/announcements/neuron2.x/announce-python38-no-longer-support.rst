.. post:: April 3, 2025
    :language: en
    :tags: announce-python-version-no-longer-support

.. _announce-python-no-longer-support:

Neuron no longer includes Python 3.8 support starting this release
-------------------------------------------------------------------

Starting with :ref:`Neuron Release 2.22 <neuron-2.22.0-whatsnew>`, Neuron no longer includes support for Python 3.8 as it has its reached end-of-life status.

=========================
How does this impact me?
=========================

I currently use Python 3.8.
============================

To avoid security issues and bugs, current users of Python 3.8 are advised to migrate to a Neuron supported Python version (3.9, 3.10, or 3.11) as Neuron no longer supports Python 3.8. For a list of supported Python versions according to Neuron package, please see :ref:`latest-neuron-release-artifacts`.

I currently use Ubuntu 20, which has Python 3.8 as the default version. Am I affected?
=======================================================================================

Although Python 3.8 is the default version of Ubuntu 20.04, Neuron will continue to support Ubuntu 20.04 until April 2025, due to extended standard support of Python 3.8 in Ubuntu 20. Please see the :ref:`sdk-maintenance-policy` for more information.
