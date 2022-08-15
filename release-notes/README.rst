.. _main-rn:

Previous Neuron Release Notes
=============================

.. contents::
   :local:
   :depth: 1


Neuron 1.19.1 (05/27/2022)
^^^^^^^^^^^^^^^^^^^^^^^^^^
**Neuron 1.19.1** is a patch release. This release fixes a bug in Neuron Driver (``aws-neuron-dkms``). Neuron driver version 2.3.11 included in this release fixes a bug that causes kernel panic when a large memory allocation on Neuron device fails.  Neuron Driver 2.3.11 also introduces a new functionality required by the upcoming Neuron 1.20.0 release.  Because the new functionality is mandatory for Neuron 1.20.0 support, Neuron Driver 2.3.11 adds a compatibility check that will prevents Neuron 1.20.0 from running with older versions of the driver.   An attempt to run Neuron 1.20.0 with an older version of the driver will result in the application terminating with an error message.

In addition, this release updates ``tensorflow-neuron`` installation instructions to pin ``protobuf`` version to avoid `compatibility issues <https://github.com/protocolbuffers/protobuf/issues/10051>`__ with older versions of TensorFlow.

.. important ::

   For successful installation or update to next releases (Neuron 1.20.0 and newer):
      * Uninstall ``aws-neuron-dkms`` by running: ``sudo apt remove aws-neuron-dkms`` or ``sudo yum remove aws-neuron-dkms``
      * Install or upgrade to latest Neuron driver (``aws-neuron-dkms``) by following the ":ref:`neuron-install-guide`" instructions.

Neuron 1.19.0 (04/29/2022)
^^^^^^^^^^^^^^^^^^^^^^^^^^
**Neuron 1.19.0** release adds support for PyTorch version 1.11, updates torch-neuron 1.10 to 1.10.2, and adds support for TensorFlow version 2.8, as well as minor enhancements and bug fixes.

Please note that starting with this release (*Neuron 1.19.0*), installing ``aws-neuron-runtime-base`` and ``oci-add-hooks`` are no longer required for Neuron Kubernetes device driver plugin. In addition starting with this release, *torch-neuron 1.5* :ref:`will no longer be supported <eol-pt-15>`.

Neuron 1.18.0 (03/25/2022)
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Neuron 1.18.0** release introduces the beta release of :ref:`NeuronPerf <neuronperf>`, NeuronPerf is a Python library with a simple API that enables fast measurements of performance when running models with Neuron. This release adds new 5 models to the :ref:`appnote-performance-benchmark` together with  NeuronPerf scripts used to compile these models and run the benchmarks.


This release also introduces additional ``torch-neuron`` packages that support C++11 ABI, updates TensorFlow-Neuron 2.5 to 2.5.3, adds support for TensorFlow-Neuron 2.6 and 2.7, and introduces Runtime NEURON_RT_NUM_CORES :ref:`environment variable <nrt-configuration>`. In addition this release include minor enhancements and bug fixes in Compiler, Neuron Framework Extensions, Runtime 2.x library and tools. See below detailed release notes.

Starting with this release, *TensorFlow Neuron versions 2.1, 2.2, 2.3 and 2.4* will :ref:`no longer be supported <eol-tf-21-24>` . We will also :ref:`stop supporting PyTorch Neuron version 1.5 <announce-eol-pt-1-5>` starting with Neuron 1.19.0 release, and :ref:`will stop supporting <eol-ncgs-env_2>`  ``NEURONCORE_GROUP_SIZES`` environment variable starting with Neuron 1.20.0 release.

Neuron 1.17.2 (02/18/2022)
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Neuron 1.17.2** is a patch release. This release fixes a bug in TensorFlow Neuron versions 2.1, 2.2. 2.3 and 2.4. The fixed bug was causing a memory leak of 128B for each inference. Starting this release, TensorFlow Neuron versions 2.1, 2.2, 2.3 and 2.4 are :ref:`entering maintenance mode <maintenance_tf21_tf24>`. Future releases of TensorFlow Neuron versions 2.1, 2.2, 2.3 and 2.4 will address security issues only.

Neuron 1.17.1 (02/16/2022)
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Neuron 1.17.1** is a patch release. This release fixes a bug in TensorFlow Neuron that caused a memory leak. The memory leak was approximately 128b for each inference and 
exists in all versions of TensorFlow Neuron versions part of Neuron 1.16.0 to Neuron 1.17.0 releases. see :ref:`pre-release-content` for exact versions included in each release.  This release only fixes the memory leak for TensorFlow versions 1.15 and 2.5 from Neuron.  The other versions of TensorFlow Neuron will be fixed in a shortly upcoming release.


Neuron 1.17.0 (01/20/2022)
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Neuron 1.17.0** release introduces the support of PyTorch 1.10,  Tensorflow 2.5 update to version 2.5.2, new operators support in PyTorch
and TensorFlow 1.15, in addition to enhancements and bug fixes in PyTorch, TensorFlow, MxNet, Compiler, Runtime and Tools.

- **PyTorch**
   * First PyTorch 1.10 support.
   * Added new operators support.
   * See :ref:`pytorch-neuron-rn` and :ref:`neuron-cc-ops-pytorch` for more details.
- **TensorFlow 2.x**
   * Updated Tensorflow 2.5 to version 2.5.2.
   * Updated tensorflow-model-server 2.5 to version 2.5.3.
   * See :ref:`tensorflow-neuron-rn-v2` and :ref:`tensorflow-modelserver-rn-v2` for more details.
- **TensorFlow 1.15**
   * Added new operators support.
   * See :ref:`tensorflow-neuron-rn` and :ref:`neuron-cc-ops-tensorflow` for more details.
- **MXNet**
   * Added support for ``mx_neuron.__version__`` to get the build version of MXNet Neuron plugin.
   * See :ref:`mxnet-neuron-rn` for more details.
- **Tools 2.x**
   * ``neuron-top`` - Added “all” tab that aggregates all running Neuron processes into a single view.
   * ``neuron-top`` - Improved startup time by approximately 1.5 seconds in most cases.
   * See :ref:`neuron-tools-rn` for more details.
- **Compiler**
   * Enhancements and minor bug fixes.
   * See :ref:`neuron-cc-rn` for more details.
- **Runtime 2.x**
   * Enhancements and minor bug fixes.
   * See :ref:`neuron-runtime-release-notes` for more details.

Neuron 1.16.3 (01/05/2022)
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Neuron 1.16.3** is a minor release. This release includes performance enhancements and operator support in :ref:`PyTorch Neuron <pytorch-neuron-rn>`
and minor bug fixes in :ref:`Neuron Compiler <neuron-cc-rn>`.


Neuron 1.16.2 (12/15/2021)
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Neuron 1.16.2** is a patch release. This release includes performance enhancements and minor bug fixes in :ref:`Neuron Compiler <neuron-cc-rn>`
and :ref:`PyTorch Neuron <pytorch-neuron-rn>`.

Neuron 1.16.1 (11/05/2021)
^^^^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^

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

Neuron 1.15.2 (09/22/2021)
^^^^^^^^^^^^^^^^^^^^^^^^^^

Neuron 1.15.2 includes bug fixes for the tensorflow-model-server-neuron 2.5.1.1.6.8.0 package and several other bug fixes for tensorflow-neuron/tensorflow-model-server-neuron packages.

Neuron 1.15.1 (08/30/2021)
^^^^^^^^^^^^^^^^^^^^^^^^^^

Neuron 1.15.1 includes bug fixes for the aws-neuron-dkms package and several other bug fixes for related packages.

Neuron 1.15.0 (08/12/2021)
^^^^^^^^^^^^^^^^^^^^^^^^^^

Neuron 1.15.0 is the first release to support TensorFlow 2. In this release TensorFlow 2 supports language transformer base models like BERT. The TensorFlow 2 support will be enhanced in future releases to support additional models.

* **TensorFlow 2.x** - To get started with TensorFlow 2.x:

  *  Run the TensorFlow 2  :ref:`HuggingFace distilBERT Tutorial </src/examples/tensorflow/huggingface_bert/huggingface_bert.ipynb>`.
  *  Read :ref:`tf2_faq`
  *  See newly introduced :ref:`TensorFlow-Neuron 2.x Tracing API <tensorflow-ref-neuron-tracing-api>`.
  *  See :ref:`tensorflow-ref-neuron-accelerated-ops`.


* **Documentation**

  *  **New** :ref:`models-inferentia` application note added in this release. This application note describes what types of deep learning model architectures perform well out of the box and provides guidance on techniques you can use to optimize your deep learning models for Inferentia.
  *  **New** :ref:`Neuron inference performance page <appnote-performance-benchmark>` provides performance information for popular models and links to test these models in your own environment. The data includes throughout and latency numbers, cost per inference, for both realtime and offline applications.
  *  **New** :ref:`TensorFlow 2 HuggingFace distilBERT Tutorial </src/examples/tensorflow/huggingface_bert/huggingface_bert.ipynb>`.
  *  **New** :ref:`Bring your own HuggingFace pretrained BERT container to Sagemaker Tutorial </src/examples/pytorch/byoc_sm_bert_tutorial/sagemaker_container_neuron.ipynb>`.



* **More information**

  *  :ref:`tensorflow-neuron-rn`
  *  :ref:`neuron-cc-rn`
  *  :ref:`tensorflow-modelserver-rn`
  

.. _07-02-2021-rn:

Neuron 1.14.2 (07/26/2021)
^^^^^^^^^^^^^^^^^^^^^^^^^^

This release (Neuron 1.14.2) , include bug fixes and minor enhancements to Neuron Runtime:

    * Neuron Runtime - see :ref:`neuron-runtime-release-notes`

Neuron 1.14.1 (07/02/2021)
^^^^^^^^^^^^^^^^^^^^^^^^^^

This release (Neuron 1.14.1) , include bug fixes and minor enhancements:

* Neuron PyTorch - This release adds “Dynamic Batching” feature support, see PyTorch-Neuron trace python API for more information, the release also add support for new operators and include additional bug fixes and minor enhancements, for more information see :ref:`pytorch-neuron-rn`.
* Neuron TensorFlow - see :ref:`tensorflow-neuron-rn`.
* Neuron MXNet - see :ref:`mxnet-neuron-rn`.
* Neuron Compiler - see :ref:`neuron-cc-rn`.
* Neuron Runtime - see :ref:`neuron-runtime-release-notes`.
* Neuron Tools - see :ref:`neuron-tools-rn`.


.. _05-28-2021-rn:

Neuron 1.14.0 (05/28/2021)
^^^^^^^^^^^^^^^^^^^^^^^^^^

This release (Neuron 1.14.0) introduces first release of Neuron PyTorch 1.8.1, tutorials update, performance enhancements and memory optimizations for Neuron PyTorch, Neuron TensorFlow and Neuron MXNet.


* Neuron PyTorch - First release of Neuron PyTorch 1.8.1.
* Neuron PyTorch - Convolution operator support has been extended to include ConvTranspose2d variants.
* Neuron PyTorch - Updated  tutorials to use Hugging Face Transformers 4.6.0.
* Neuron PyTorch - Additional performance enhancements, memory optimizations, and bug fixes. see :ref:`pytorch-neuron-rn`.
* Neuron Compiler - New feature  -  Uncompressed NEFF format for faster loading models prior inference. Enable it by –enable-fast-loading-neuron-binaries. Some cases of large models may be detrimentally  impacted as it will not be compressed but many cases will benefit.
* Neuron Compiler - Additional performance enhancements, memory optimizations, and bug fixes, see :ref:`neuron-cc-rn`.
* Neuron TensorFlow - Performance enhancements, memory optimizations, and bug fixes. see :ref:`tensorflow-neuron-rn`. 
* Neuron MXNet - Enhancements and minor bug fixes (MXNet 1.8), see :ref:`mxnet-neuron-rn`.
* Neuron Runtime - Performance enhancements, memory optimizations, and bug fixes. :ref:`neuron-runtime-release-notes`.
* Neuron Tools - Minor bug fixes and enhancements.
* Software Deprecation

    * End of support for Neuron Conda packages in Deep Learning AMI, users should use pip upgrade commands to upgrade to latest Neuron version in DLAMI, see `blog <https://aws.amazon.com/blogs/developer/neuron-conda-packages-eol/>`_.
    * End of support for Ubuntu 16, see  `documentation <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/deprecation.html>`_.


Neuron 1.14.0 (05/28/2021)
^^^^^^^^^^^^^^^^^^^^^^^^^^

This release (Neuron 1.14.0) introduces first release of Neuron PyTorch 1.8.1, tutorials update, performance enhancements and memory optimizations for Neuron PyTorch, Neuron TensorFlow and Neuron MXNet.


* Neuron PyTorch - First release of Neuron PyTorch 1.8.1.
* Neuron PyTorch - Convolution operator support has been extended to include ConvTranspose2d variants.
* Neuron PyTorch - Updated  tutorials to use Hugging Face Transformers 4.6.0.
* Neuron PyTorch - Additional performance enhancements, memory optimizations, and bug fixes. see :ref:`pytorch-neuron-rn`.
* Neuron Compiler - New feature  -  Uncompressed NEFF format for faster loading models prior inference. Enable it by –enable-fast-loading-neuron-binaries. Some cases of large models may be detrimentally  impacted as it will not be compressed but many cases will benefit.
* Neuron Compiler - Additional performance enhancements, memory optimizations, and bug fixes, see :ref:`neuron-cc-rn`.
* Neuron TensorFlow - Performance enhancements, memory optimizations, and bug fixes. see :ref:`tensorflow-neuron-rn`. 
* Neuron MXNet - Enhancements and minor bug fixes (MXNet 1.8), see :ref:`mxnet-neuron-rn`.
* Neuron Runtime - Performance enhancements, memory optimizations, and bug fixes. :ref:`neuron-runtime-release-notes`.
* Neuron Tools - Minor bug fixes and enhancements.
* Software Deprecation

    * End of support for Neuron Conda packages in Deep Learning AMI, users should use pip upgrade commands to upgrade to latest Neuron version in DLAMI, see `blog <https://aws.amazon.com/blogs/developer/neuron-conda-packages-eol/>`_.
    * End of support for Ubuntu 16, see  `documentation <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/deprecation.html>`_.


Neuron 1.13.0 (05/01/2021)
^^^^^^^^^^^^^^^^^^^^^^^^^^

This release introduces higher performance, updated framework support, new tutorials, and adding models and tools:

* Additional compiler improvements boost performance up to 20% higher throughput compared to previous release across model types.
* Improving usability for NLP models, with out-of-the-box 12x higher-throughput at 70% lower cost for Hugging Face Transformers pre-trained BERT Base models, see :ref:`pytorch-tutorials-neuroncore-pipeline-pytorch`.
* Upgrade Apache MXNet (Incubating) to 1.8, where Neuron is now a plugin, see :ref:`mxnet-neuron-rn`.
* PyTorch ResNext models now functional with new operator support, see :ref:`pytorch-neuron-rn`.
* PyTorch Yolov5 support, see :ref:`pytorch-neuron-rn`.
* MXNet (Incubating): Gluon API and Neuron support for NLP BERT models, see :ref:`mxnet-neuron-rn`.
* PyTorch Convolution operator support has been extended to include most Conv1d and Conv3d variants, please see :ref:`neuron-cc-ops-pytorch`  for the complete list of operators.
* First release of Neuron plugin for TensorBoard, see :ref:`neuron-tensorboard-rn`.

**Software Deprecation**

* :ref:`eol-conda-packages`
* :ref:`eol-ubuntu16`
* :ref:`eol-classic-tensorboard`


.. _03-04-2021-rn:

March 4, 2021 Release (Patch)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This release include bug fixes and minor enhancements to the Neuron Runtime and Tools. 


February 24, 2021 Release (Patch)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This release updates all Neuron packages and libraries in response to the Python Secutity issue CVE-2021-3177 as described here: https://nvd.nist.gov/vuln/detail/CVE-2021-3177. This vulnerability potentially exists in multiple versions of Python including 3.5, 3.6, 3.7. Python is used by various components of Neuron, including the Neuron compiler as well as Machine Learning frameworks including TensorFlow, PyTorch and Apache MXNet (Incubating). It is recommended that the Python interpreters used in any AMIs and containers used with Neuron are also updated. 

Python 3.5 reached `end-of-life <https://devguide.python.org/devcycle/?highlight=python%203.5%20end%20of%20life#end-of-life-branches>`_, from this release Neuron packages will not support Python 3.5.
Users should upgrade to latest DLAMI or upgrade to a newer Python versions if they are using other AMI.


January 30, 2021 Release
^^^^^^^^^^^^^^^^^^^^^^^^

This release continues to improves the NeuronCore Pipeline performance for BERT models. For example, running BERT Base with the the neuroncore-pipeline-cores compile option, at batch=3, seqlen=32 using 16 Neuron Cores, results in throughput of up to  5340 sequences per second and P99 latency of 9ms using Tensorflow Serving. 

This release also adds operator support and performance improvements for the PyTorch based DistilBert model for sequence classification.


December 23, 2020 Release
^^^^^^^^^^^^^^^^^^^^^^^^^

This release introduces a PyTorch 1.7 based torch-neuron package as a part of the Neuron SDK. Support for PyTorch model serving with TorchServe 0.2 is added and will be demonstrated with a tutorial. This release also provides an example tutorial for PyTorch based Yolo v4 model for Inferentia. 

To aid visibility into compiler activity, the Neuron-extended Frameworks TensorFlow and PyTorch will display a new compilation status indicator that prints a dot (.) every 20 seconds to the console as compilation is executing. 

Important to know:
------------------

1. This update continues to support the torch-neuron version of PyTorch 1.5.1 for backwards compatibility.
2. As Python 3.5 reached end-of-life in October 2020, and many packages including TorchVision and Transformers have
stopped support for Python 3.5, we will begin to stop supporting Python 3.5 for frameworks, starting with
PyTorch-Neuron version :ref:`neuron-torch-11170` in this release. You can continue to use older versions with Python 3.5.

November 17, 2020 Release
^^^^^^^^^^^^^^^^^^^^^^^^^

This release improves NeuronCore Pipeline performance. For example,
running BERT Small, batch=4, seqlen=32 using 4 Neuron Cores, results in
throughput of up to 7000 sequences per second and P99 latency of 3ms
using Tensorflow Serving.

Neuron tools updated the NeuronCore utilization metric to include all
inf1 compute engines and DMAs. Added a new neuron-monitor example that
connects to Grafana via Prometheus. We've added a new sample script
which exports most of neuron-monitor's metrics to a Prometheus
monitoring server. Additionally, we also provided a sample Grafana
dashboard. More details at :ref:`neuron-tools`.

ONNX support is limited and from this version onwards we are not
planning to add any additional capabilities to ONNX. We recommend
running models in TensorFlow, PyTorch or MXNet for best performance and
support.

October 22, 2020 Release
^^^^^^^^^^^^^^^^^^^^^^^^

This release adds a Neuron kernel mode driver (KMD). The Neuron KMD
simplifies Neuron Runtime deployments by removing the need for elevated
privileges, improves memory management by removing the need for huge
pages configuration, and eliminates the need for running neuron-rtd as a
sidecar container. Documentation throughout the repo has been updated to
reflect the new support. The new Neuron KMD is backwards compatible with
prior versions of Neuron ML Frameworks and Compilers - no changes are
required to existing application code.

More details in the Neuron Runtime release notes at :ref:`neuron-runtime`.

September 22, 2020 Release
^^^^^^^^^^^^^^^^^^^^^^^^^^

This release improves performance of YOLO v3 and v4, VGG16, SSD300, and
BERT. As part of these improvements, Neuron Compiler doesn’t require any
special compilation flags for most models. Details on how to use the
prior optimizations are outlined in the neuron-cc :ref:`neuron-cc-rn`.

The release also improves operational deployments of large scale
inference applications, with a session management agent incorporated
into all supported ML Frameworks and a new neuron tool called
neuron-monitor allows to easily scale monitoring of large fleets of
Inference applications. A sample script for connecting neuron-monitor to
Amazon CloudWatch metrics is provided as well. Read more about using
neuron-monitor :ref:`neuron-monitor-ug`.

August 19, 2020 Release
^^^^^^^^^^^^^^^^^^^^^^^

Bug fix for an error reporting issue with the Neuron Runtime. Previous
versions of the runtime were only reporting uncorrectable errors on half
of the dram per Inferentia. Other Neuron packages are not changed.

August 8, 2020 Release
^^^^^^^^^^^^^^^^^^^^^^

This release of the Neuron SDK delivers performance enhancements for the
BERT Base model. Sequence lengths including 128, 256 and 512 were found
to have best performance at batch size 6, 3 and 1 respectively using
publically available versions of both Pytorch (1.5.x) and
Tensorflow-based (1.15.x) models. The compiler option "-O2" was used in
all cases.

A new Kubernetes scheduler extension is included in this release to
improve pod scheduling on inf1.6xlarge and inf1.24xlarge instance sizes.
Details on how the scheduler works and how to apply the scheduler can be
found :ref:`neuron-k8-scheduler-ext`.
Check the :ref:`neuron-k8-rn` for details
changes to k8 components going forward.

August 4, 2020 Release
^^^^^^^^^^^^^^^^^^^^^^

Bug fix for a latent issue caused by a race condition in Neuron Runtime
leading to possible crashes. The crash was observed under stress load
conditons. All customers are encouraged to update the latest Neuron
Runtime package (aws-neuron-runtime), version 1.0.8813.0 or newer. Other
Neuron packages are being updated as well, but are to be considered
non-critical updates.

July 16, 2020 Release
^^^^^^^^^^^^^^^^^^^^^

This release of Neuron SDK adds support for the OpenPose (posenet)
Neural Network. An example of using Openpose for end to end inference is
available :ref:`/src/examples/tensorflow/openpose_demo/openpose.ipynb`.

A new PyTorch auto-partitioner feature now automatically builds a Neuron
specific graph representation of PyTorch models. The key benefit of this
feature is automatic partitioning the model graph to run the supported
operators on the NeuronCores and the rest on the host. PyTorch
auto-partitioner is enabled by default with ability to disable if a
manual partition is needed. More details :ref:`neuron-pytorch`. The
release also includes various bug fixes and increased operator support.

Important to know:
------------------

1. This update moves the supported version for PyTorch to the current
   release (PyTorch 1.5.1)
2. This release supports Python 3.7 Conda packages in addition to Python
   3.6 Conda packages

June 18, 2020 Release
^^^^^^^^^^^^^^^^^^^^^

Point fix an error related to yum downgrade/update of Neuron Runtime
packages. The prior release fails to successfully downgrade/update
Neuron Runtime Base package and Neuron Runtime package when using Yum on
Amazon Linux 2.

Please remove and then install both packages on AL2 using these
commands:

::

   # Amazon Linux 2
   sudo yum remove aws-neuron-runtime-base
   sudo yum remove aws-neuron-runtime
   sudo yum install aws-neuron-runtime-base
   sudo yum install aws-neuron-runtime

Jun 11, 2020 Release
^^^^^^^^^^^^^^^^^^^^

This Neuron release provides support for the recent launch of EKS for
Inf1 instance types and numerous other improvements. More details about
how to use EKS with the Neuron SDK can be found in AWS documentation
`here <https://docs.aws.amazon.com/eks/latest/userguide/inferentia-support.html>`__.

This release adds initial support for OpenPose PoseNet for images with
resolutions upto 400x400.

This release also adds a '-O2' option to the Neuron Compiler. '-O2' can
help with handling of large tensor inputs.

In addition the Neuron Compiler increments the version of the compiled
artifacts, called "NEFF", to version 1.0. Neuron Runtime versions
earlier than the 1.0.6905.0 release in May 2020 will not be able to
execute NEFFs compiled from this release forward. Please see :ref:`neff-support-table` for
compatibility.

Stay up to date on future improvements and new features by following the
`Neuron SDK Roadmap <https://github.com/aws/aws-neuron-sdk/projects/2>`__.

Refer to the detailed release notes for more information on each Neuron
component.

.. _important-to-know-1:

Important to know:
------------------

1. Size of neural network. The current Neuron compiler release has a
   limitation in terms of the size of neural network it could
   effectively optimize for. The size of neural network is influenced by
   a number of factors including: a) type of neural network (CNN, LSTM,
   MLP) , b) number of layers, c) sizes of input (dimension of the
   tensors, batch size, ...). Using the Neuron Compiler '-O2' option can
   help with handling of large tensor inputs for some models. If not
   used, Neuron limits the size of CNN models like ResNet to an input
   size of 480x480 fp16/32, batch size=4; LSTM models like GNMT to have
   a time step limit of 900; MLP models like BERT to have input size
   limit of sequence length=128, batch=8.

2. INT8 data type is not currently supported by the Neuron compiler.

3. Neuron does not support TensorFlow 2 or PyTorch 1.4.0.

May 15, 2020 Release
^^^^^^^^^^^^^^^^^^^^

Point fix an error related to installation of the Neuron Runtime Base
package. The prior release fails to successfully start Neuron Discovery
when the Neuron Runtime package is not also installed. This scenario of
running Neuron Discovery alone is critical to users of Neuron in
container environments.

Please update the aws-neuron-runtime-base package:

::

   # Ubuntu 18 or 16:
   sudo apt-get update
   sudo apt-get install aws-neuron-runtime-base

   # Amazon Linux, Centos, RHEL
   sudo yum update
   sudo yum install aws-neuron-runtime-base

May 11, 2020 Release
^^^^^^^^^^^^^^^^^^^^

This release provides additional throughput improvements to running
inference on a variety of models; for example BERTlarge throughput has
improved by an additional 35% compared to the previous release and with
peak thoughput of 360 seq/second on inf1.xlarge (more details :ref:`tensorflow-bert-demo` ).

In addition to the performance boost, this release adds PyTorch, and
MXNet framework support for BERT models, as well as expands container
support in preparation to an upcoming EKS launch.

We continue to work on new features and improving performance further,
to stay up to date follow this repository and our `Neuron roadmap <https://github.com/aws/aws-neuron-sdk/projects/2>`__.

Refer to the detailed release notes for more information for each Neuron
component.

.. _important-to-know-2:

Important to know:
------------------

1. Size of neural network. The current Neuron compiler release has a
   limitation in terms of the size of neural network it could
   effectively optimize for. The size of neural network is influenced by
   a number of factors including: a) type of neural network (CNN, LSTM,
   MLP) , b) number of layers, c) sizes of input (dimension of the
   tensors, batch size, ...). As a result, we limit the sizes of CNN
   models like ResNet to have an input size limit of 480x480 fp16/32,
   batch size=4; LSTM models like GNMT to have a time step limit of 900;
   MLP models like BERT to have input size limit of sequence length=128,
   batch=8.

2. INT8 data type is not currently supported by the Neuron compiler.

3. Neuron does not support TensorFlow 2 or PyTorch 1.4.0.

Mar 26, 2020 Release
^^^^^^^^^^^^^^^^^^^^

This release supports a variant of the SSD object detection network, a
SSD inference demo is available :ref:`tensorflow-ssd300`

This release also enhances our Tensorboard support to enable CPU-node
visibility.

Refer to the detailed release notes for more information for each neuron
component.

.. _important-to-know-3:

Important to know:
------------------

1. Size of neural network. The current Neuron compiler release has a
   limitation in terms of the size of neural network it could
   effectively optimize for. The size of neural network is influenced by
   a number of factors including: a) type of neural network (CNN, LSTM,
   MLP) , b) number of layers, c) sizes of input (dimension of the
   tensors, batch size, ...). As a result, we limit the sizes of CNN
   models like ResNet to have an input size limit of 480x480 fp16/32,
   batch size=4; LSTM models like GNMT to have a time step limit of 900;
   MLP models like BERT to have input size limit of sequence length=128,
   batch=8.

2. INT8 data type is not currently supported by the Neuron compiler.

3. Neuron does not support TensorFlow 2 or PyTorch 1.4.0.

Feb 27, 2020 Release
^^^^^^^^^^^^^^^^^^^^

This release improves performance throughput by up to 10%, for example
ResNet-50 on inf1.xlarge has increased from 1800 img/sec to 2040
img/sec, Neuron logs include more detailed messages and various bug
fixes. Refer to the detailed release notes for more details.

We continue to work on new features and improving performance further,
to stay up to date follow this repository, and watch the `AWS Neuron
developer
forum <https://forums.aws.amazon.com/forum.jspa?forumID=355>`__.

.. _important-to-know-4:

Important to know:
------------------

1. Size of neural network. The current Neuron compiler release has a
   limitation in terms of the size of neural network it could
   effectively optimize for. The size of neural network is influenced by
   a number of factors including: a) type of neural network (CNN, LSTM,
   MLP) , b) number of layers, c) sizes of input (dimension of the
   tensors, batch size, ...). As a result, we limit the sizes of CNN
   models like ResNet to have an input size limit of 480x480 fp16/32,
   batch size=4; LSTM models like GNMT to have a time step limit of 900;
   MLP models like BERT to have input size limit of sequence length=128,
   batch=8.

2. Computer-vision object detection and segmentation models are not yet
   supported.

3. INT8 data type is not currently supported by the Neuron compiler.

4. Neuron does not support TensorFlow 2 or PyTorch 1.4.0.

Jan 28, 2020 Release
^^^^^^^^^^^^^^^^^^^^

This release brings significant throughput improvements to running
inference on a variety of models; for example Resnet50 throughput is
increased by 63% (measured 1800 img/sec on inf1.xlarge up from 1100/sec,
and measured 2300/sec on inf1.2xlarge). BERTbase throughput has improved
by 36% compared to the re:Invent launch (up to 26100seq/sec from
19200seq/sec on inf1.24xlarge), and BERTlarge improved by 15% (230
seq/sec, compared to 200 running on inf1.2xlarge). In addition to the
performance boost, this release includes various bug fixes as well as
additions to the GitHub with  :ref:`neuron-fundamentals`
diving deep on how Neuron performance features work and overall improved
documentation following customer input.

We continue to work on new features and improving performance further,
to stay up to date follow this repository, and watch the `AWS Neuron
developer
forum <https://forums.aws.amazon.com/forum.jspa?forumID=355>`__.

.. _important-to-know-5:

Important to know:
------------------

1. Size of neural network. The current Neuron compiler release has a
   limitation in terms of the size of neural network it could
   effectively optimize for. The size of neural network is influenced by
   a number of factors including: a) type of neural network (CNN, LSTM,
   MLP) , b) number of layers, c) sizes of input (dimension of the
   tensors, batch size, ...). As a result, we limit the sizes of CNN
   models like ResNet to have an input size limit of 480x480 fp16/32,
   batch size=4; LSTM models like GNMT to have a time step limit of 900;
   MLP models like BERT to have input size limit of sequence length=128,
   batch=8.

2. Computer-vision object detection and segmentation models are not yet
   supported.

3. INT8 data type is not currently supported by the Neuron compiler.

4. Neuron does not support TensorFlow 2 or PyTorch 1.4.0.

Neuron SDK Release Notes Structure
----------------------------------

The Neuron SDK is delivered through commonly used package mananagers
(e.g. PIP, APT and YUM). These packages are then themselves packaged
into Conda packages that are integrated into the AWS DLAMI for minimal
developer overhead.

The Neuron SDK release notes follow a similar structure, with the core
improvements and known-issues reported in the release notes of the
primary packages (e.g. Neuron-Runtime or Neuron-Compiler release notes),
and additional release notes specific to the package-integration are
reported through their dedicated release notes (e.g. Conda or DLAMI
release notes).
