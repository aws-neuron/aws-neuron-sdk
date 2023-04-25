.. _neuron-tensorboard-rn:


Neuron Plugin for TensorBoard Release Notes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. contents:: Table of Contents
   :local:
   :depth: 1


Known Issues and Limitations - Updated 11/29/2022
================================================

The following are not limitations in the Neuron plugin, but may affect your ability to
use TensorBoard.

- The Neuron plugin for Trn1 (``tensorboard-plugin-neuronx``) is not compatible with the Neuron plugin
  for Inf1 (``tensorboard-plugin-neuron``).  Please ensure you only have only the correct package installed.


Neuron Plugin for TensorBoard release [2.5.26.0]
================================================

Date: 04/28/2023

Summary
-------

* Neuron operator timeline view now includes Neuron Runtime setup/teardown time and a collapsed execution of NC engines and DMA - see Tensorboard tutorial for updated views. 

* Improved execution categorization to include "control" instructions



Neuron Plugin for TensorBoard release [2.5.25.0]
================================================

Date: 03/28/2023

Summary
-------

- Supports INF2 and TRN1.


Neuron Plugin for TensorBoard release [2.5.0.0]
===============================================

Date: 12/09/2022

Summary
-------

- Added support for PyTorch Neuron on Trn1 (``torch-neuronx``) with new views!  Includes a trace view,
  an operator view, and an operator timeline view.  For more info, check out the documentation
  :ref:`neuronx-plugin-tensorboard`.

  .. important::

    - You must update to the latest Neuron Tools (``aws-neuronx-tools`` version 2.6 or newer) and install
      ``tensorboard-plugin-neuronx`` for proper functionality of the Neuron plugin on Trn1.
    - For Inf1, please continue to use ``tensorboard-plugin-neuron``.  Refer to the getting started guide
      on Inf1 :ref:`neuron-plugin-tensorboard`.


Neuron Plugin for TensorBoard release [2.4.0.0]
===============================================

Date: 04/29/2022

Summary
-------

- Minor updates.


Neuron Plugin for TensorBoard release [2.3.0.0]
===============================================

Date: 03/25/2022

Summary
-------

- Minor updates.


Neuron Plugin for TensorBoard release [2.2.0.0]
===============================================

Date: 10/27/2021

New in this release
-------------------

   -  Neuron Plugin for TensorBoard now support applications built with Neuron Runtime 2.x (``libnrt.so``).

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


[2.1.2.0]
=========

Date: 8/12/2021

Summary
-------

- Adds support for Neuron Tensorflow 2.5+


.. _2.1.0.0:

[2.1.0.0]
=========

Date: 5/28/2021

Summary
-------

- No major changes or fixes. Released with other Neuron packages.

.. _2.0.29.0:

[2.0.29.0]
=========

Date: 4/30/2021

Summary
-------

- First release Neuron plugin for TensorBoard.  Check out it out here:
  :ref:`neuron-plugin-tensorboard`.

   - The Neuron plugin is now compatible with TensorBoard 2.0 and higher,
     in addition to TensorBoard 1.15

   - Provides a centralized place to better understand execution using
     Neuron SDK.

   - Continues support visualization for TensorFlow graphs, with support
     for PyTorch and MXNet coming in future releases.

- Neuron plugin for TensorBoard is supported for Neuron tools >= 1.5, which is first
  introduced in Neuron v1.13.0 release
- TensorBoard-Neuron is deprecated, and only supported for Neuron tools <= 1.4.12.0.
  The final version, 1.4.12.0 is part of Neuron v1.12.2 release.


.. _11501260:

[1.15.0.1.2.6.0]
================

Date: 2/24/2021

Summary
-------

-  Fix for CVE-2021-3177.

.. _11501110:

[1.15.0.1.1.1.0]
================

Date: 12/23/2020

Summary
-------

-  Minor internal improvements.


.. _1150106150:

[1.15.0.1.0.615.0]
==================

Date: 11/17/2020

Summary
-------

-  Fix issue with viewing chrome trace in Neuron profile plugin in
   Chrome 80+.

Resolved Issues
---------------

-  Updated dependencies to polyfill missing APIs used by chrome trace in
   newer browser versions.


.. _1150106000:

[1.15.0.1.0.600.0]
==================

Date: 09/22/2020

Summary
-------

-  Minor internal improvements.

.. _1150105700:

[1.15.0.1.0.570.0]
==================

Date: 08/08/2020

.. _summary-1:

Summary
-------

-  Minor internal improvements.

.. _1150105130:

[1.15.0.1.0.513.0]
==================

Date: 07/16/2020

.. _summary-2:

Summary
-------

-  Minor internal improvements.

.. _1150104910:

[1.15.0.1.0.491.0]
==================

Date 6/11/2020

.. _summary-3:

Summary
-------

Fix issue where utilization was missing in the op-profile view.

Resolved Issues
---------------

-  The op-profile view in the Neuron Profile plugin now correctly shows
   the overall NeuronCore utilization.

.. _1150104660:

[1.15.0.1.0.466.0]
==================

Date 5/11/2020

.. _summary-4:

Summary
-------

Fix potential installation issue when installing both tensorboard and
tensorboard-neuron.

.. _resolved-issues-1:

Resolved Issues
---------------

-  Added tensorboard as a dependency in tensorboard-neuron. This
   prevents the issue of overwriting tensorboard-neuron features when
   tensorboard is installed after tensorboard-neuron.

Other Notes
-----------

.. _1150103920:

[1.15.0.1.0.392.0]
==================

Date 3/26/2020

.. _summary-5:

Summary
-------

Added ability to view CPU node latency in the Graphs plugin and the
Neuron Profile plugins.

Major New Features
------------------

-  Added an aggregate view in addition to the current Neuron subgraph
   view for both the Graphs plugin and the Neuron Profile plugin.
-  When visualizing a graph executed on a Neuron device, CPU node
   latencies are available when coloring the graph by "Compute time"
   using the "neuron_profile" tag.
-  The Neuron Profile plugin now has an overview page to compare time
   spent on Neuron device versus on CPU.

.. _other-notes-1:

Other Notes
-----------

-  Requires Neuron-RTD config option "enable_node_profiling" to be set
   to "true"

.. _1150103660:

[1.15.0.1.0.366.0]
==================

Date 02/27/2020

.. _summary-6:

Summary
-------

Reduced load times and fixed crashes when loading large models for
visualization.

.. _resolved-issues-2:

Resolved Issues
---------------

-  Enable large attribute filtering by default
-  Reduced load time for graphs with attributes larger than 1 KB
-  Fixed a fail to load graphs with many large attributes totaling more
   than 1 GB in size

.. _1150103150:

[1.15.0.1.0.315.0]
==================

Date 12/20/2019

.. _summary-7:

Summary
-------

No major chages or fixes. Released with other Neuron packages.

.. _1150103060:

[1.15.0.1.0.306.0]
==================

Date 12/1/2019

.. _summary-8:

Summary
-------

.. _major-new-features-1:

Major New Features
------------------

.. _resolved-issues-3:

Resolved Issues
---------------

.. _known-issues--limits:

Known Issues & Limits
---------------------

Same as prior release

.. _other-notes-2:

Other Notes
-----------

.. _1150102800:

[1.15.0.1.0.280.0]
==================

Date 11/29/2019

.. _summary-9:

Summary
-------

Initial release packaged with DLAMI.

.. _major-new-features-2:

Major New Features
------------------

N/A, initial release.

See user guide here:
https://github.com/aws/aws-neuron-sdk/blob/master/docs/neuron-tools/getting-started-tensorboard-neuron.md

.. _resolved-issues-4:

Resolved Issues
---------------

N/A - first release

.. _known-issues--limits-1:

Known Issues & Limits
---------------------

-  Must install TensorBoard-Neuron by itself, or after regular
   TensorBoard is installed. If regular Tensorboard is installed after
   TensorBoard-Neuron, it may overwrite some needed files.
-  Utilization missing in Op Profile due to missing FLOPs calculation
   (see overview page instead)
-  Neuron Profile plugin may not immediately show up on launch (try
   reloading the page)
-  Graphs with NeuronOps may take a long time to load due to attribute
   size
-  Instructions that cannot be matched to a framework layer/operator
   name show as “” (blank)
-  CPU Usage section in chrome-trace is not applicable
-  Debugger currently supports TensorFlow only
-  Visualization requires a TensorFlow-compatible graph

.. _other-notes-3:

Other Notes
-----------
