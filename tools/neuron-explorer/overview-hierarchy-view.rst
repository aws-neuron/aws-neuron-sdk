.. meta::
    :description: Learn about the Hierarchy View in Neuron Explorer for analyzing framework layers and HLO operations with zooming, highlighting, and display options.
    :date-modified: 12/02/2025

Hierarchy Viewer
===================

The Hierarchy Viewer shows an up-leveled representation of the hardware execution organized by the framework layers and HLO operations. It enables you to progressively drill down into nested layers or operators and map the execution of application level constructs to the Neuron device. This view interacts with other tools such as the Device Trace Viewer.

.. image:: /tools/profiler/images/hierarchy-view-1.gif


Zooming
-------

.. image:: /tools/profiler/images/hierarchy-view-2.png

You can zoom in on the Hierarchy Viewer in a couple of ways:

* Click-drag your mouse across the graph (support in both directions)
* Scroll down using your mouse wheel, with the mouse cursor on the x-axis
* Zoom in and out buttons in the top-right corner
* With the keyboard:
  
    * W and S for zooming in and out, respectively
    * Up and down arrow keys for zooming in and out, respectively

To zoom out, simply scroll up with your mouse wheel when you place your mouse cursor on the x-axis.

Change Displayed Layers
-----------------------

.. image:: /tools/profiler/images/hierarchy-view-3.png

The display options menu, accessed with the button in the top-right corner, allows you to selectively show or hide different layers. For instance, in the example shown above, the framework layer is hidden while displaying the hierarchy starting from HLO.

Highlighting
------------

.. image:: /tools/profiler/images/hierarchy-view-4.png

Right-clicking on an operator in Hierarchy Viewer will highlight all the corresponding instructions in the Device Trace Viewer for the operator using the same color. Multiple operators can be highlighted at once.

.. image:: /tools/profiler/images/hierarchy-view-5.png

