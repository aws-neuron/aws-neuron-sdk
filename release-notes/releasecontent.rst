.. _neuron-release-content:

Release Details
===============

.. contents::
   :local:
   :depth: 2


Current Release - Neuron 1.16.2 (12/15/2021)
--------------------------------------------



Release included packages
^^^^^^^^^^^^^^^^^^^^^^^^^

.. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --list packages --neuron-version=1.16.2

See :ref:`neuron-maintenance-policy` for more information.

Release supported frameworks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. program-output:: python3 src/helperscripts/neuronsetuphelper.py --file src/helperscripts/neuron-releases-manifest.json --list frameworks --neuron-version=1.16.2



Dependency Software Supported Versions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left

   * - Software
     - Supported
   * - Python
     - * Python 3.6
       * Python 3.7





Deep Learning AMI (DLAMI) Versions Matrix
-----------------------------------------

The DLAMI version can be determined by examining the AMI name of the EC2 instance from EC2 console or examining the file README in the default home directory.

.. include:: /neuron-intro/dlami/dlami-neuron-matrix-table.rst

See :ref:`dlami-section` for more information.


Previous Releases Content
-------------------------

:ref:`pre-release-content`

