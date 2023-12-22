.. _sdk-classification:

Neuron Software Classification
==============================

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
---------

This document explains the Neuron software classification for APIs,
libraries, packages, features, and Neuron supported model classes
mentioned in the Neuron documentation.

.. note::

   For APIs, libraries, packages, features and model classes, only
   Alpha and Beta software classifications will be mentioned. Otherwise,
   they should be considered as “Stable.”

.. note::
   
   APIs, libraries, packages, features, and model classes at
   Alpha or Beta classification should not be used in production
   environments, and are meant for early access and feedback purposes only.
   Alpha and Beta releases are Developer Preview releases under the `AWS SDK
   policy <https://docs.aws.amazon.com/sdkref/latest/guide/maint-policy.html>`__.

.. _api-software-classification:

APIs Software Classification
-------------------------------

This section details the classification of APIs supported in any Neuron
Components, in addition to environment variables and flags (e.g.
compiler flags). Examples of APIs supported by Neuron are Neuron APIs like
:func:`torch_neuron.trace`, Neuron Environment variables like
:ref:`pytorch-neuronx-envvars`, and Neuron flags like :ref:`Neuron compiler flags <neuron-compiler-cli-reference-guide>`.

.. note::

   Alpha and Beta classified APIs are APIs in a Developer Preview
   release (see `AWS SDK
   policy) <https://docs.aws.amazon.com/sdkref/latest/guide/maint-policy.html>`__
   that should not be used in production environments and are meant for
   early access and feedback purposes only.

+--------+-----------------------------+-----------------------------+
|        | API Contract                | API Backward                |
|        |                             | Compatibility               |
+========+=============================+=============================+
| Alpha  | Major changes may happen    | No                          |
+--------+-----------------------------+-----------------------------+
| Beta   | Minor changes may happen    | No                          |
+--------+-----------------------------+-----------------------------+
| Stable | Incremental changes in new  | Yes\*                       |
|        | releases (without breaking  |                             |
|        | the API contract)\*         |                             |
+--------+-----------------------------+-----------------------------+

*In case when a new Neuron version of a Stable release will break backwards compatibility, AWS will notify customers of the breaking change at least one month before the change.

.. _packages--libraries-software-classification:

Packages / Libraries Software Classification
---------------------------------------------

This section details the classification of Neuron packages or libraries
such as :ref:`Neuron
Runtime <neuron_runtime>`,
:ref:`PyTorch
Neuron <pytorch-neuronx-main>`
or :ref:`Neuron
Distributed <neuronx-distributed-index>`.

.. note::

   Alpha and Beta classified packages/libraries are packages/libraries in a Developer Preview release (see `AWS SDK
   policy) <https://docs.aws.amazon.com/sdkref/latest/guide/maint-policy.html>`__ that should not be used in production environments and are meant for
   early access and feedback purposes only.

+--------+---------------------+---------------------+-------------+
|        | Testing             | Features            | Performance |
+========+=====================+=====================+=============+
| Alpha  | Basic               | Basic               | Not tested  |
+--------+---------------------+---------------------+-------------+
| Beta   | Basic               | Minimal Viable      | Not tested  |
|        |                     | Product (MVP)\*     |             |
+--------+---------------------+---------------------+-------------+
| Stable | Standard Product    | Incremental         | Tested      |
|        | Testing             | additions/changes   |             |
|        |                     | in new releases     |             |
+--------+---------------------+---------------------+-------------+

*A minimum viable product (MVP) for a package/library contains just enough features to be usable by early customers who can then provide feedback for future development. MVP can be different per use case and depends on the specific package/library of interest.
Please note that in many cases, an MVP can also represent an advanced level of features.

.. _features-software-classification:

Features Software Classification
----------------------------------

This section details the classification for Neuron features. An example
of a Neuron feature is :ref:`Neuron Persistent Cache <neuron-caching>` in the :ref:`Transformers
Neuron <transformers_neuronx_readme>` library.

.. note::

   Alpha and Beta classified features are features in a Developer
   Preview release (see `AWS SDK
   policy) <https://docs.aws.amazon.com/sdkref/latest/guide/maint-policy.html>`__
   that should not be used in production environments and are meant for
   early access and feedback purposes only.

+--------+---------------------+---------------------+-------------+
|        | Testing             | Functionality       | Performance |
+========+=====================+=====================+=============+
| Alpha  | Basic               | Basic               | Not tested  |
+--------+---------------------+---------------------+-------------+
| Beta   | Basic               | Minimal Viable      | Not tested  |
|        |                     | Product (MVP)\*     |             |
+--------+---------------------+---------------------+-------------+
| Stable | Standard Product    | Incremental         | Tested      |
|        | Testing             | additions/changes   |             |
|        |                     | in new releases     |             |
+--------+---------------------+---------------------+-------------+

*A minimum viable product (MVP) for a feature contains just enough functionality to be usable by early customers who can then provide feedback for future development. MVP can be different per use case and depends on the specific feature of interest. Please note that in many cases, an MVP can also represent an advanced level of functionality.

.. _models-software-classification:

Neuron Model Classes Software Classification
----------------------------------------------

This section details the classification for Neuron model classes which
mainly refers throughput/latency and accuracy for both training and
inference.

.. note::

   A Neuron supported model class is tightly coupled with a
   specific supported ML Framework (e.g. PyTorch Neuron), specific ML
   library (e.g. NeuronX Distributed) and the workload type (e.g. Training
   or Inference). For example a model can be supported at Beta level in
   PyTorch Neuron for training and Stable level in PyTorch Neuron for
   inference.

.. note::
   Alpha and Beta classified model classes are model classes in a
   Developer Preview release (see `AWS SDK
   policy) <https://docs.aws.amazon.com/sdkref/latest/guide/maint-policy.html>`__
   that should not be used in production environments and are meant for
   early access and feedback purposes only.

====== ====================== ====================
\      Accuracy / Convergence Throughput / Latency
====== ====================== ====================
Beta   Validated              Not tested
Stable Validated              Tested
====== ====================== ====================

