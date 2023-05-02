.. _mxnet-neuron-rn:


Apache MXNet Neuron (Incubating) Release Notes
==============================================

.. contents:: Table of contents
   :local:
   :depth: 1

This document lists the release notes for MXNet-Neuron framework.

Apache MXNet Neuron release [1.8.0.2.4.1.0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date: 5/1/2023

New in this release
-------------------

* Updated Neuron Runtime library to version 2.12
* Added missing LICENSE.txt

Known Issues and Limitations
----------------------------

* Bert-base in 16 NeuronCores pipeline mode has 50% lower performance when running 16 inferences in parallel with Runtime version 2.12.

[1.5.1.1.10.39.0]
^^^^^^^^^^^^^^^^^

Date: 5/1/2023

Summary
-------

Minor bug fixes and enhancements for MXNet 1.5 Neuron.

Apache MXNet Neuron release [1.8.0.2.2.43.0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date: 11/23/2022

Summary
-------

Minor bug fixes and enhancements for MXNet 1.8 Neuron.

[1.5.1.1.10.11.0]
^^^^^^^^^^^^^^^^^

Date: 11/23/2022

Summary
-------

Minor bug fixes and enhancements for MXNet 1.5 Neuron.

[1.5.1.1.10.0.0]
^^^^^^^^^^^^^^^^

Date: 04/28/2022

Summary
-------

Minor bug fixes and enhancements for MXNet 1.5 Neuron.

Apache MXNet Neuron release [1.8.0.2.2.2.0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date: 03/25/2022

New in this release
-------------------

* Added support for unloading models from a NeuronDevice by deleting the model instance in user application. Users can now call ``del`` in Python on an executor and to unload the model from a NeuronDevice (provided the deleted executor is the last executor pointing to the given model). This requires the latest ``aws-mx-1.8`` package from ``https://aws-mx-pypi.s3.us-west-2.amazonaws.com/1.8.0/aws_mx-1.8.0.2-py2.py3-none-manylinux2014_x86_64.whl``. 

Bug fixes
---------

* Fixed a memory leak caused by stale unloaded models in NeuronDevice memory. For this fix to take effect please install aws-mx package from https://aws-mx-pypi.s3.us-west-2.amazonaws.com/1.8.0/aws_mx-1.8.0.2-py2.py3-none-manylinux2014_x86_64.whl along with the latest mx-neuron package.

[1.5.1.1.9.0.0]
^^^^^^^^^^^^^^^

Date: 03/25/2022

Summary
-------

Minor bug fixes and enhancements for MXNet 1.5 Neuron.


Apache MXNet Neuron release [1.8.0.2.1.5.0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date: 01/20/2022

New in this release
-------------------

* Added support of ``mx_neuron.__version__`` to get the build version of MXNet Neuron plugin

Bug fixes
---------

* Fixed assertion errors when inference was completed with NaNs. The expected behavior is to complete inference successfully and warn the 
  user that ``NaN``s were seen during the current inference. 
* Fixed compile issue when individual output nodes have multiple output nodes. Because the output index was being dropped, fewer number 
  of output feature maps were being considered and that caused failures during inference. 


Apache MXNet Neuron release [1.8.0.2.0.276.0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date: 11/05/2021

* Updated Neuron Runtime (which is integrated within this package) to ``libnrt 2.2.18.0`` to fix a container issue that was preventing 
  the use of containers when /dev/neuron0 was not present. See details here :ref:`neuron-runtime-release-notes`.

Apache MXNet Neuron release [1.8.0.2.0.271.0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date 10/27/2021

New in this release
-------------------

-  MXNet Neuron 1.8 now support Neuron Runtime 2.x (``libnrt.so`` shared library) only.

     .. important::

        -  You must update to the latest Neuron Driver (``aws-neuron-dkms`` version 2.1 or newer) 
           for proper functionality of the new runtime library.
        -  Read :ref:`introduce-libnrt`
           application note that describes :ref:`why are we making this
           change <introduce-libnrt-why>` and
           how :ref:`this change will affect the Neuron
           SDK <introduce-libnrt-how-sdk>` in detail.
        -  Read :ref:`neuron-migrating-apps-neuron-to-libnrt` for detailed information of how to
           migrate your application.

-  Introducing Flexible Execution Groups (FlexEG) feature. See :ref:`flexeg` application note.


Resolved Issues
---------------

-  Fixed a bug that prevented compilation of gluon models with multiple
   cpu and neuron nodes.
-  Added more debug logic to help with profiling of model load timing.


[1.5.1.1.7.0.0]
^^^^^^^^^^^^^^^

Date 10/27/2021

New in this release
-------------------

-  MXNet 1.5 enters maintenance mode. Please visit :ref:`maintenance_mxnet_1_5` for more
   information.

Resolved Issues
---------------

 -  Minor bug fixes.


[1.5.1.1.6.5.0]
^^^^^^^^^^^^^^^

Date 08/12/2021

Summary
-------

Minor bug fixes and enhancements for MXNet 1.5 Neuron.

[1.8.0.1.3.4.0]
^^^^^^^^^^^^^^^

Date 08/12/2021

Summary
-------

Minor bug fixes and enhancements for MXNet 1.8 Neuron.


[1.5.1.1.6.1.0]
^^^^^^^^^^^^^^^

Date 07/02/2021

Summary
-------

Minor bug fixes and enhancements for MXNet 1.5 Neuron.

[1.8.0.1.3.0.0]
^^^^^^^^^^^^^^^

Date 07/02/2021

Summary
-------

Support for Autoloop, Cpredict API and minor bug fixes and enhancements for MXNet 1.8 Neuron.

Major New Features
------------------

- Added support for Autoloop feature for MXNet 1.8 Neuron.

Resolved Issues
---------------

- Added support for CPredict API.


[1.8.0.1.2.1.0]
^^^^^^^^^^^^^^^

Date 5/28/2021

Summary
-------

Minor bug fixes and enhancements for MXNet 1.8 Neuron

Resolved Issues
---------------
- Added support for Neuron profiler 


[1.8.0.1.1.2.0]
^^^^^^^^^^^^^^^

Date 4/30/2021

Summary
-------

Initial release of Apache MXNet (Incubating) 1.8 for Neuron

Major New Features
------------------

- Gluon API and Neuron support for NLP BERT models

- Neuron is now a plugin

- Please note new API changes to support plugin mode: :ref:`ref-mxnet-neuron-compilation-python-api`

[1.5.1.1.4.x.x]
^^^^^^^^^^^^^^^

Date 5/28/2021

Summary
-------

- Minor enhancements.

[1.5.1.1.4.4.0]
^^^^^^^^^^^^^^^

Date 4/30/2021

Summary
-------

- Resolve an issue with Neuron profiling.

Resolved Issues
---------------

- Issue: when Neuron profiling is enabled in MXNet-Neuron 1.5.1 (using NEURON_PROFILE=<dir>), and TensorBoard is used to read in the profiled data, user would see an error messsage "panic: runtime error: index out of range". This issue is resolved in this release.

[1.5.1.1.3.8.0]
^^^^^^^^^^^^^^^

Date 3/4/2021

Summary
-------

Minor enhancements.

[1.5.1.1.3.7.0]
^^^^^^^^^^^^^^^

Date 2/24/2021

Summary
-------

Fix for CVE-2021-3177.

[1.5.1.1.3.2.0]
^^^^^^^^^^^^^^^

Date 1/30/2021

Summary
-------

Various minor improvements

[1.5.1.1.2.1.0]
^^^^^^^^^^^^^^^

Date 12/23/2020

Summary
-------

Various minor improvements

[1.5.1.1.1.88.0]
^^^^^^^^^^^^^^^^

Date 11/17/2020

Summary
-------

This release includes the bug fix for MXNet Model Server not being able to clean up
Neuron RTD states after model is unloaded (deleted) from model server.

Resolved Issues
---------------

-  Issue: MXNet Model Server is not able to clean up Neuron RTD states
   after model is unloaded (deleted) from model server.

    -  Workaround for earlier versions: run “\ ``/opt/aws/neuron/bin/neuron-cli reset``\ “ to
   clear Neuron RTD states after all models are unloaded and server is
   shut down.

[1.5.1.1.1.52.0]
^^^^^^^^^^^^^^^^

Date 09/22/2020

Summary
-------

Various minor improvements.

Major New Features
------------------

Resolved Issues
---------------

-  Issue: When first importing MXNet into python process and subprocess
   call is invoked, user may get an OSError exception "OSError: [Errno
   14] Bad address" during subprocess call (see
   https://github.com/apache/incubator-mxnet/issues/13875 for more
   details). This issue is fixed with a mitigation patch from MXNet for
   Open-MP fork race conditions.

   -  Workaround for earlier versions: Export KMP_INIT_AT_FORK=false
      before running python process.

.. _1511110:

[1.5.1.1.1.1.0]
^^^^^^^^^^^^^^^

Date 08/08/2020

.. _summary-1:

Summary
-------

Various minor improvements.

.. _major-new-features-1:

Major New Features
------------------

.. _resolved-issues-1:

Resolved Issues
---------------

.. _1511021010:

[1.5.1.1.0.2101.0]
^^^^^^^^^^^^^^^^^^

Date 08/05/2020

.. _summary-2:

Summary
-------

Various minor improvements.

.. _major-new-features-2:

Major New Features
------------------

.. _resolved-issues-2:

Resolved Issues
---------------

.. _1511020930:

[1.5.1.1.0.2093.0]
^^^^^^^^^^^^^^^^^^

Date 07/16/2020

.. _summary-3:

Summary
-------

This release contains a few bug fixes and user experience improvements.

.. _major-new-features-3:

Major New Features
------------------

.. _resolved-issues-3:

Resolved Issues
---------------

-  User can specify NEURONCORE_GROUP_SIZES without brackets (for
   example, "1,1,1,1"), as can be done in TensorFlow-Neuron and
   PyTorch-Neuron.
-  Fixed a memory leak when inferring neuron subgraph properties
-  Fixed a bug dealing with multi-input subgraphs

.. _1511020330:

[1.5.1.1.0.2033.0]
^^^^^^^^^^^^^^^^^^

Date 6/11/2020

.. _summary-4:

Summary
-------

-  Added support for profiling during inference

.. _major-new-features-4:

Major New Features
------------------

-  Profiling can now be enabled by specifying the profiling work
   directory using NEURON_PROFILE environment variable during inference.
   For an example of using profiling, see :ref:`tensorboard-neuron`.
   (Note that graph view of MXNet graph is not available via
   TensorBoard).

.. _resolved-issues-4:

Resolved Issues
---------------

Known Issues and Limitations
----------------------------

Other Notes
-----------

.. _1511019000:

[1.5.1.1.0.1900.0]
^^^^^^^^^^^^^^^^^^

Date 5/11/2020

.. _summary-5:

Summary
-------

Improved support for shared-memory communication with Neuron-Runtime.

.. _major-new-features-5:

Major New Features
------------------

-  Added support for the BERT-Base model (base: L-12 H-768 A-12), max
   sequence length 64 and batch size of 8.
-  Improved security for usage of shared-memory for data transfer
   between framework and Neuron-Runtime
-  Improved allocation and cleanup of shared-memory resource
-  Improved container support by automatic falling back to GRPC data
   transfer if shared-memory cannot be allocated by Neuron-Runtime

.. _resolved-issues-5:

Resolved Issues
---------------

-  User is unable to allocate Neuron-Runtime shared-memory resource when
   using MXNet-Neuron in a container to communicate with Neuron-Runtime
   in another container. This is resolved by automatic falling back to
   GRPC data transfer if shared-memory cannot be allocated by
   Neuron-Runtime.
-  Fixed issue where some large models could not be loaded on
   inferentia.

.. _known-issues-and-limitations-1:

Known Issues and Limitations
----------------------------

.. _other-notes-1:

Other Notes
-----------

.. _1511015960:

[1.5.1.1.0.1596.0]
^^^^^^^^^^^^^^^^^^

Date 3/26/2020

.. _summary-6:

Summary
-------

No major changes or fixes

.. _major-new-features-6:

Major New Features
------------------

.. _resolved-issues-6:

Resolved Issues
---------------

.. _known-issues-and-limitations-2:

Known Issues and Limitations
----------------------------

.. _other-notes-2:

Other Notes
-----------

.. _1511014980:

[1.5.1.1.0.1498.0]
^^^^^^^^^^^^^^^^^^

Date 2/27/2020

.. _summary-7:

Summary
-------

No major changes or fixes.

.. _major-new-features-7:

Major New Features
------------------

.. _resolved-issues-7:

Resolved Issues
---------------

The issue(s) below are resolved:

-  Latest pip version 20.0.1 breaks installation of MXNet-Neuron pip
   wheel which has py2.py3 in the wheel name.

.. _known-issues-and-limitations-3:

Known Issues and Limitations
----------------------------

-  User is unable to allocate Neuron-Runtime shared-memory resource when
   using MXNet-Neuron in a container to communicate with Neuron-Runtime
   in another container. To work-around, please set environment variable
   NEURON_RTD_USE_SHM to 0.

.. _other-notes-3:

Other Notes
-----------

.. _1511014010:

[1.5.1.1.0.1401.0]
^^^^^^^^^^^^^^^^^^

Date 1/27/2020

.. _summary-8:

Summary
-------

No major changes or fixes.

.. _major-new-features-8:

Major New Features
------------------

.. _resolved-issues-8:

Resolved Issues
---------------

-  The following issue is resolved when the latest multi-model-server
   with version >= 1.1.0 is used with MXNet-Neuron. You would still need
   to use "``/opt/aws/neuron/bin/neuron-cli reset``" to clear all Neuron
   RTD states after multi-model-server is exited:

   -  Issue: MXNet Model Server is not able to clean up Neuron RTD
      states after model is unloaded (deleted) from model server and
      previous workaround "``/opt/aws/neuron/bin/neuron-cli reset``" is
      unable to clear all Neuron RTD states.

.. _known-issues-and-limitations-4:

Known Issues and Limitations
----------------------------

-  Latest pip version 20.0.1 breaks installation of MXNet-Neuron pip
   wheel which has py2.py3 in the wheel name. This breaks all existing
   released versions. The error looks like:

::

   Looking in indexes: https://pypi.org/simple, https://pip.repos.neuron.amazonaws.com
   ERROR: Could not find a version that satisfies the requirement mxnet-neuron (from versions: none)
   ERROR: No matching distribution found for mxnet-neuron

-  Work around: install the older version of pip using "pip install
   pip==19.3.1".

.. _other-notes-4:

Other Notes
-----------

.. _1511013250:

[1.5.1.1.0.1325.0]
^^^^^^^^^^^^^^^^^^

Date 12/1/2019

.. _summary-9:

Summary
-------

.. _major-new-features-9:

Major New Features
------------------

.. _resolved-issues-9:

Resolved Issues
---------------

-  Issue: Compiler flags cannot be passed to compiler during compile
   call. The fix: compiler flags can be passed to compiler during
   compile call using “flags” option followed by a list of flags.

-  Issue: Advanced CPU fallback option is a way to attempt to improve
   the number of operators on Inferentia. The default is currently set
   to on, which may cause failures. The fix: This option is now off by
   default.

.. _known-issues-and-limitations-5:

Known Issues and Limitations
----------------------------

-  Issue: MXNet Model Server is not able to clean up Neuron RTD states
   after model is unloaded (deleted) from model server and previous
   workaround "``/opt/aws/neuron/bin/neuron-cli reset``" is unable to
   clear all Neuron RTD states.

   -  Workaround: run “\ ``sudo systemctl restart neuron-rtd``\ “ to
      clear Neuron RTD states after all models are unloaded and server
      is shut down.

.. _other-notes-5:

Other Notes
-----------

.. _1511013490:

[1.5.1.1.0.1349.0]
^^^^^^^^^^^^^^^^^^

Date 12/20/2019

.. _summary-10:

Summary
-------

No major changes or fixes. Released with other Neuron packages.

.. _1511013250-1:

[1.5.1.1.0.1325.0]
^^^^^^^^^^^^^^^^^^

Date 12/1/2019

.. _summary-11:

Summary
-------

.. _major-new-features-10:

Major New Features
------------------

.. _resolved-issues-10:

Resolved Issues
---------------

-  Issue: Compiler flags cannot be passed to compiler during compile
   call. The fix: compiler flags can be passed to compiler during
   compile call using “flags” option followed by a list of flags.

-  Issue: Advanced CPU fallback option is a way to attempt to improve
   the number of operators on Inferentia. The default is currently set
   to on, which may cause failures. The fix: This option is now off by
   default.

.. _known-issues-and-limitations-6:

Known Issues and Limitations
----------------------------

-  Issue: MXNet Model Server is not able to clean up Neuron RTD states
   after model is unloaded (deleted) from model server and previous
   workaround "``/opt/aws/neuron/bin/neuron-cli reset``" is unable to
   clear all Neuron RTD states.

   -  Workaround: run “\ ``sudo systemctl restart neuron-rtd``\ “ to
      clear Neuron RTD states after all models are unloaded and server
      is shut down.

.. _other-notes-6:

Other Notes
-----------

.. _1511012600:

[1.5.1.1.0.1260.0]
^^^^^^^^^^^^^^^^^^

Date: 11/25/2019

.. _summary-12:

Summary
-------

This version is available only in released DLAMI v26.0 and is based on
MXNet version 1.5.1. Please :ref:`dlami-rn-known-issues` to latest version.

.. _major-new-features-11:

Major new features
------------------

.. _resolved-issues-11:

Resolved issues
---------------

.. _known-issues-and-limitations-7:

Known issues and limitations
----------------------------

-  Issue: Compiler flags cannot be passed to compiler during compile
   call.

-  Issue: Advanced CPU fallback option is a way to attempt to improve
   the number of operators on Inferentia. The default is currently set
   to on, which may cause failures.

   -  Workaround: explicitly turn it off by setting compile option
      op_by_op_compiler_retry to 0.

-  Issue: Temporary files are put in current directory when debug is
   enabled.

   -  Workaround: create a separate work directory and run the process
      from within the work directory

-  Issue: MXNet Model Server is not able to clean up Neuron RTD states
   after model is unloaded (deleted) from model server.

   -  Workaround: run “\ ``/opt/aws/neuron/bin/neuron-cli reset``\ “ to
      clear Neuron RTD states after all models are unloaded and server
      is shut down.

-  Issue: MXNet 1.5.1 may return inconsistent node names for some
   operators when they are the primary outputs of a Neuron subgraph.
   This causes failures during inference.

   -  Workaround : Use the ``excl_node_names`` compilation option to
      change the partitioning of the graph during compile so that these
      nodes are not the primary output of a neuron subgraph. See
      :ref:`ref-mxnet-neuron-compilation-python-api`

   .. code:: python

      compile_args = { 'excl_node_names': ["node_name_to_exclude"] }

Models Supported
----------------

The following models have successfully run on neuron-inferentia systems

1. Resnet50 V1/V2
2. Inception-V2/V3/V4
3. Parallel-WaveNet
4. Tacotron 2
5. WaveRNN

.. _other-notes-7:

Other Notes
-----------

-  Python versions supported:

   -  3.5, 3.6, 3.7

-  Linux distribution supported:

   -  Ubuntu 18, Amazon Linux 2
