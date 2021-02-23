.. _neuron-tensorboard-rn:


TensorBoard-Neuron Release Notes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. contents:: Table of Contents
   :local:
   :depth: 1


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
