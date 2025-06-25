.. _neuron-release-content:
.. _latest-neuron-release-content:
.. _latest-neuron-release-artifacts:

Release Content
===============

.. contents:: Table of contents
   :local:
   :depth: 2

Neuron 2.24.0 (06/24/2025)
---------------------------

Trn1 packages
^^^^^^^^^^^^^

.. program-output:: python3 src/helperscripts/n2-helper.py --list=packages --instance=trn1 --file=src/helperscripts/n2-manifest.json --neuron-version=2.24.0

Trn2 packages
^^^^^^^^^^^^^

.. program-output:: python3 src/helperscripts/n2-helper.py --list=packages --instance=trn2 --file=src/helperscripts/n2-manifest.json --neuron-version=2.24.0


Inf2 packages
^^^^^^^^^^^^^

.. program-output:: python3 src/helperscripts/n2-helper.py --list=packages --instance=inf2 --file=src/helperscripts/n2-manifest.json --neuron-version=2.24.0

Inf1 packages
^^^^^^^^^^^^^

.. program-output:: python3 src/helperscripts/n2-helper.py --list=packages --instance=inf1 --file=src/helperscripts/n2-manifest.json --neuron-version=2.24.0

Supported Python Versions for Inf1 packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. program-output:: python3 src/helperscripts/n2-helper.py --list=pyversions --instance=inf1 --file=src/helperscripts/n2-manifest.json --neuron-version=2.24.0

Supported Python Versions for Inf2/Trn1/Trn2 packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. program-output:: python3 src/helperscripts/n2-helper.py --list=pyversions --instance=inf2 --file=src/helperscripts/n2-manifest.json --neuron-version=2.24.0

Supported NumPy Versions
^^^^^^^^^^^^^^^^^^^^^^^^
Neuron supports versions >= 1.21.6 and <= 1.22.2

Supported Hugging Face Transformers Versions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
+----------------------------------+----------------------------------+
| Package                          | Supported Hugging Face           |
|                                  | Transformers Versions            |
+==================================+==================================+
| torch-neuronx                    | < 4.35 and >=4.37.2              |
+----------------------------------+----------------------------------+
| transformers-neuronx             | >= 4.36.0                        |
+----------------------------------+----------------------------------+
| neuronx-distributed - Llama      | 4.31                             |
| model class                      |                                  |
+----------------------------------+----------------------------------+
| neuronx-distributed - GPT NeoX   | 4.26                             |
| model class                      |                                  |
+----------------------------------+----------------------------------+
| neuronx-distributed - Bert model | 4.26                             |
| class                            |                                  |
+----------------------------------+----------------------------------+
| nemo-megatron                    | 4.31.0                           |
+----------------------------------+----------------------------------+

Supported Protobuf Versions
^^^^^^^^^^^^^^^^^^^^^^^^^^^
+----------------------------------+----------------------------------+
| Package                          | Supported Probuf versions        |
+==================================+==================================+
| neuronx-cc                       | > 3                              |
+----------------------------------+----------------------------------+
| torch-neuronx                    | >= 3.20                          |
+----------------------------------+----------------------------------+
| torch-neuron                     | < 3.20                           |
+----------------------------------+----------------------------------+
| transformers-neuronx             | >= 3.20                          |
+----------------------------------+----------------------------------+
| neuronx-distributed              | >= 3.20                          |
+----------------------------------+----------------------------------+
| tensorflow-neuronx               | < 3.20                           |
+----------------------------------+----------------------------------+
| tensorflow-neuron                | < 3.20                           |
+----------------------------------+----------------------------------+
  
Previous Neuron Releases Content
--------------------------------

* :ref:`pre-release-content`
* :ref:`pre-n1-release-content`
