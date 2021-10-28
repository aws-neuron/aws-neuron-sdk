.. _software-maintenance:

Software maintenance mode
==========================

.. contents::
	:local:
	:depth: 1
	

.. _maintenance_rtd:

Neuron Runtime 1.x (``neuron-rtd``) enters maintenance mode
-----------------------------------------------------------

10/27/2021 - Starting with *Neuron 1.16.0* release, *Neuron Runtime 1.x* (``neuron-rtd``) is entering maintenance mode and replaced 
with *Neuron Runtime 2.x*, a shared library named ``libnrt.so``. 
Future releases of *Neuron Runtime 1.x* (``neuron-rtd``) will address critical bug fixes and security issues only. Previous releases of 
*Neuron Runtime 1.x* (``neuron-rtd``) will continue to be available via ``rpm`` and ``deb`` packages.

For more information please see:

	* :ref:`introduce-libnrt`
	* :ref:`neuron-install-guide`
	* :ref:`neuron-maintenance-policy`


.. _maintenance_mxnet_1_5:

Neuron support for *Apache MXNet 1.5* enters maintenance mode
--------------------------------------------------------------

10/27/2021 - Starting *Neuron release 1.16.0*,  Neuron support for *MXNet 1.5* is entering maintenance mode.
Future releases of Neuron supporting *MXNet 1.5*  will address critical bug fixes and security issues only.
Previous releases of *Apache MXNet 1.5* will continue to be available via ``pip`` packages.

Current users of *Neuron MXNet 1.5* can migrate their applications to *Neuron MXNet 1.8*, for more information 
about Neuron MXNet support and how to upgrade to latest *Neuron MXNet 1.8*, please see visit :ref:`neuron-mxnet`.


.. _maintenance_neuron-cli:

``neuron-cli`` enters maintenance mode
--------------------------------------

10/27/2021 - Starting *Neuron release 1.16.0*, with the introduction of *Neuron Runtime 2.x*, ``neuron-cli`` is entering maintenance mode. ``neuron-cli`` 
functionality will be available only if *Neuron Runtime 1.x* (``neuron-rtd``) is being used by the application. If the application is using 
*Neuron Runtime 2.x* shared library(``libnrt.so``), ``neuron-cli`` functionality will not be available.


If you have used ``neuron-cli`` in previous releases, and you are migrating to
newer Neuron releases where applications require *Neuron Runtime 2.x* shared library, please see the below :ref:`neuron-cli-mntnce-faq`.
Future releases of ``neuron-cli`` will address 
critical bug fixes and security issues only. Previous releases of ``neuron-cli`` will continue to be available via ``rpm`` and ``deb`` packages.

.. _neuron-cli-mntnce-faq:

Frequently Asked questions (FAQ)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Is there another tool that provide the same functionality as ``neuron-cli list-model``?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes, please see :ref:`neuron-ls-ug` or :ref:`neuron-monitor-ug`.

Is there another tool that provide the same functionality as ``neuron-cli create-ncg``, ``neuron-cli destroy-ncg``, and ``neuron-cli list-ncg``?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No, these functionalities are no longer needed with *Neuron Runtime 2.x*,NeuronCore Groups (NCG) :ref:`is deprecated <eol-ncg>` and ``NEURONCORE_GROUP_SIZES`` environment variable :ref:`is in the process of being deprecated <eol-ncgs-env>`, Please start using ``NEURON_RT_VISIBLE_CORES`` instead. See :ref:`nrt-configuration` and :ref:`neuron-migrating-apps-neuron-to-libnrt` 

for more information.

Is there another tool that provide the same functionality as ``neuron-cli reset``?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No, this functionality is no longer needed with *Neuron Runtime 2.x*. Before introducing ``libnrt.so``, in certain cases after an application 
crashed  models had to be unloaded manually by calling neuron-cli reset.

With ``libnrt.so``, applications runs in the context of the ``libnrt.so`` shared library and when an application exits the Neuron driver will free all resources associated with the application.


For more information please see:

	* :ref:`introduce-libnrt`
	* :ref:`neuron-tools`
	* :ref:`neuron-install-guide`
	* :ref:`neuron-maintenance-policy`




