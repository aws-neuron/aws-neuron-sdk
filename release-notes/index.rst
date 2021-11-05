.. _neuron-whatsnew:

What's New
==========

.. contents::
   :local:
   :depth: 2

.. _latest-neuron-release:

Neuron 1.16.1 (11/05/2021)
--------------------------

**Neuron 1.16.1** is a patch release. This release fixes a bug in Neuron Runtime that would have prevented users from launching a container that doesn’t use all of the Neuron Devices in the instance. If you are using Neuron within a container, please update to this new release by updating to latest Neuron ML framework package, Neuron Tools, and/or TensorFlow Neuron Model Server.


* To update to latest PyTorch 1.9.1:
  ``pip install --upgrade torch-neuron neuron-cc[tensorflow] torchvision``

* To update to latest TensorFlow 2.5.1:
  ``pip install --upgrade tensorflow-neuron[cc]``

* To update to latest TensorFlow 1.15.5:
  ``pip install --upgrade tensorflow-neuron==1.15.5.* neuron-cc``

* To update to latest MXNet 1.8.0:
  ``pip install --upgrade mx_neuron neuron-cc``


For more details on how to update the framework packages, please check out our :ref:`QuickStart guides <neuron-gettingstarted>`.


Neuron 1.16.0 (10/27/2021)
--------------------------

**Neuron 1.16.0 is a release that requires your attention**. **You must update to the latest Neuron Driver (** ``aws-neuron-dkms`` **version 2.1 or newer) 
for successful installation or upgrade**. 

This release introduces 
:ref:`Neuron Runtime 2.x <introduce-libnrt>`, upgrades :ref:`PyTorch Neuron <neuron-pytorch>` to 
PyTorch 1.9.1, adds support for new APIs (:func:`torch.neuron.DataParallel` and ``torch_neuron.is_available()``), 
adds new features and capabilities (compiler ``--fast-math`` :ref:`option for better fine-tuning of accuracy/performance <mixed-precision>` and :ref:`MXNet FlexEG feature <flexeg>`),
improves :ref:`tools <neuron-tools>`, adds support for additional :ref:`operators <neuron-supported-operators>`, 
improves :ref:`performance <appnote-performance-benchmark>`
(Up to 20% additional throughput and up to 25% lower latency),
and reduces model loading times. It also simplifies :ref:`Neuron installation steps <neuron-install-guide>`, 
and improves the user experience of :ref:`container creation and deployment <neuron-containers>`. 
In addition it includes bug fixes, new :ref:`application notes <neuron-appnotes>`, updated :ref:`tutorials <neuron-tutorials>`, 
and announcements of software :ref:`deprecation <software-deprecation>` and :ref:`maintenance <software-maintenance>`.


-  **Neuron Runtime 2.x**

   - :ref:`introduce-libnrt` - In this release we are introducing Neuron Runtime 2.x. 
     The new runtime is a shared library (``libnrt.so``), replacing Neuron Runtime 1.x
     which was a server daemon (``neruon-rtd``).

     Upgrading to ``libnrt.so`` is expected to improves throughput and
     latency, simplifies Neuron installation and upgrade process,
     introduces new capabilities for allocating NeuronCores to
     applications, streamlines container creation, and deprecates tools
     that are no longer needed. The new library-based runtime
     (``libnrt.so``) is directly integrated into Neuron’s ML Frameworks (with the exception of MXNet 1.5) and Neuron
     Tools packages. As a result, users no longer need to install/deploy the
     ``aws-neuron-runtime``\ package. 
    
     .. important::

        -  You must update to the latest Neuron Driver (``aws-neuron-dkms`` version 2.1 or newer) 
           for proper functionality of the new runtime library.
        -  Read :ref:`introduce-libnrt`
           application note that describes :ref:`why we are making this
           change <introduce-libnrt-why>` and
           how :ref:`this change will affect the Neuron
           SDK <introduce-libnrt-how-sdk>` in detail.
        -  Read :ref:`neuron-migrating-apps-neuron-to-libnrt` for detailed information of how to
           migrate your application.


-  **Performance**

   -  Updated :ref:`performance numbers <appnote-performance-benchmark>` - Improved performance: Up to 20% additional throughput 
      and up to 25% lower latency.

-  **Documentation resources**

   -  Improved :ref:`Neuron Setup Guide <neuron-install-guide>`.
   -  New :ref:`introduce-libnrt` application note.
   -  New :ref:`bucketing_app_note` application note.
   -  New :ref:`mixed-precision` application note.
   -  New :ref:`torch-neuron-dataparallel-app-note` application note.
   -  New :ref:`flexeg` application note.
   -  New :ref:`parallel-exec-ncgs` application note.
   -  New :ref:`Using NEURON_RT_VISIBLE_CORES with TensorFlow Serving <tensorflow-serving-neuronrt-visible-cores>` tutorial.
   -  Updated :ref:`ResNet50 model for Inferentia </src/examples/pytorch/resnet50.ipynb>` tutorial to use :func:`torch.neuron.DataParallel`.

-  **PyTorch**

   -  PyTorch now supports Neuron Runtime 2.x only. Please visit :ref:`introduce-libnrt` for
      more information.
   -  Introducing PyTorch 1.9.1 support. 
   -  Introducing new APIs: :func:`torch.neuron.DataParallel` (see :ref:`torch-neuron-dataparallel-app-note` application note for more details) and
      ``torch_neuron.is_available()``.
   -  Introducing :ref:`new operators support <neuron-cc-ops-pytorch>`.
   -  For more information visit :ref:`neuron-pytorch`

-  **TensorFlow 2.x**

   -  TensorFlow 2.x now supports Neuron Runtime 2.x only. Please visit
      :ref:`introduce-libnrt` for more information.
   -  Updated Tensorflow 2.3.x from Tensorflow 2.3.3 to Tensorflow
      2.3.4.
   -  Updated Tensorflow 2.4.x from Tensorflow 2.4.2 to Tensorflow
      2.4.3.
   -  Updated Tensorflow 2.5.x from Tensorflow 2.5.0 to Tensorflow
      2.5.1.
   -  Introducing :ref:`new operators support <tensorflow-ref-neuron-accelerated-ops>`
   -  For more information visit :ref:`tensorflow-neuron`

-  **TensorFlow 1.x**

   -  TensorFlow 1.x now supports Neuron Runtime 2.x only. Please visit
      :ref:`introduce-libnrt` for more information.
   -  Introducing :ref:`new operators support <neuron-cc-ops-tensorflow>`.
   -  For more information visit :ref:`tensorflow-neuron`

-  **MXNet 1.8**

   -  MXNet 1.8 now supports Neuron Runtime 2.x only. Please visit
      :ref:`introduce-libnrt` for more information.
   -  Introducing Flexible Execution Groups (FlexEG) feature.
   -  MXNet 1.5 enters maintenance mode. Please visit :ref:`maintenance_mxnet_1_5` for more
      information.
   -  For more information visit :ref:`neuron-mxnet`

-  **Neuron Compiler**

   -  Introducing the ``–-fast-math`` option for better fine-tuning of accuracy/performance. See :ref:`mixed-precision`
   -  Support added for new ArgMax and ArgMin operators. See :ref:`neuron-cc-rn`.
   -  For more information visit :ref:`neuron-cc`

-  **Neuron Tools**

   -  Updates have been made to ``neuron-ls`` and ``neuron-top`` to
      improve the interface and utility of information
      provided.
   -  `neuron-monitor`` has been enhanced to include additional information when
      used to monitor the latest Frameworks released with Neuron 1.16.0. See :ref:`neuron-tools-rn`.
   -  ``neuron-cli`` is entering maintenance mode as its use is no longer
      relevant when using ML Frameworks with an integrated Neuron
      Runtime (libnrt.so).
   -  For more information visit :ref:`neuron-tools`

-  **Neuron Containers**

   -  Starting with Neuron 1.16.0, installation of Neuron ML Frameworks now includes
      an integrated Neuron Runtime library. As a result, it is
      no longer required to deploy ``neuron-rtd``. Please visit :ref:`introduce-libnrt` for
      information.
   -  When using containers built with components from Neuron 1.16.0, or
      newer, please use ``aws-neuron-dkms`` version 2.1 or newer and the
      latest version of ``aws-neuron-runtime-base``. Passing additional
      system capabilities is no longer required.
   -  For more information visit :ref:`neuron-containers`

-  **Neuron Driver**

   -  Support is added for Neuron Runtime 2.x (libnrt.so).
   -  Memory improvements have been made to ensure all allocations are made with
      4K alignments.


-  **Software Deprecation**

   - :ref:`eol-ncgs-env`
   - :ref:`eol-ncg`


-  **Software maintenance mode**

   - :ref:`maintenance_rtd`
   - :ref:`maintenance_mxnet_1_5`
   - :ref:`maintenance_neuron-cli`   

Detailed release notes
----------------------

.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left
   
   * - Software 
     - Details
      
   * - General     
      
     - * :ref:`neuron-release-content`
      
       * :ref:`software-deprecation`

       * :ref:`software-maintenance`
      
   * - PyTorch   
     - * :ref:`pytorch-neuron-rn`
      
       * :ref:`neuron-cc-ops-pytorch` 
      

   * - TensorFlow 2.x
     - * :ref:`tensorflow-neuron-rn-v2`

       * :ref:`tensorflow-ref-neuron-accelerated-ops`
      
       * :ref:`tensorflow-modelserver-rn-v2`      


      
   * - TensorFlow 1.x
     - * :ref:`tensorflow-neuron-rn`

       * :ref:`neuron-cc-ops-tensorflow`
      
       * :ref:`tensorflow-modelserver-rn`

      
   * - Apache MXNet (Incubating)
      
     - * :ref:`mxnet-neuron-rn`
      
       * :ref:`neuron-cc-ops-mxnet`
      
      

   * - Compiler              
     - * :ref:`neuron-cc-rn`

       * :ref:`neuron-supported-operators`
      
   * - Runtime
     - * :ref:`neuron-runtime-release-notes`

       * :ref:`neuron-driver-release-notes`

   * - Containers
     - * :ref:`neuron-k8-rn`

       * :ref:`neuron-containers-release-notes`


   * - Tools
      
     - * :ref:`neuron-tools-rn`
      
       * :ref:`neuron-tensorboard-rn`
      
   * - Software Deprecation
   
     - * :ref:`software-deprecation`

   * - Software Maintenance
   
     - * :ref:`software-maintenance`


Previous Releases
-----------------

.. toctree::
   :maxdepth: 1

   README
   
