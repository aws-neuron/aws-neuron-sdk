.. _neuron-top-ug:

Neuron Top User Guide
=====================

.. contents::
   :local:
   :depth: 2

Overview
--------
``neuron-top`` provides useful information about NeuronCore and vCPU utilization, memory usage,
loaded models, and Neuron applications.

.. note ::
  If you are parsing ``neuron-top`` output in your automation environment, you can now replace it with ``neuron-monitor``
  (:ref:`neuron-monitor-ug`) which outputs data in a standardized, easier to parse JSON format.

Using neuron-top
----------------

Command line arguments
~~~~~~~~~~~~~~~~~~~~~~
Launching ``neuron-top`` with no arguments will show data for all ML Applications running with Neuron Runtime 2.x.
``neuron-top`` will also show data for any Neuron Runtime 1.x (``neuron-rtd`` daemon) running on the default GRPC
address - either ``$NEURON_RTD_ADDRESS`` or ``unix:/run/neuron.sock`` if that environment variable is not defined.

If more than one ``neuron-rtd`` daemons are running, each of their GRPC addresses needs to be
specified in the command line:

::

  neuron-top unix:/run/neuron2.sock unix:/run/neuron3.sock

User interface
~~~~~~~~~~~~~~


The user interface is divided in 4 sections. The data shown in these
sections only applies to the currently selected Neuron application:

|image0|

* The ``Neuroncore Utilization`` section shows the NeuronCore utilization for the
  currently selected Neuron application.

* The ``VCPU and Memory Info`` section shows:

  * ``Total system vCPU usage`` - the two percentages are user% and system%
  * ``Runtime vCPU usage`` - same breakdown
  * ``Runtime Memory Host`` - amount of host memory used by the Application and total available
  * ``Runtime Memory Device`` - amount of device memory used by the Application

* ``Loaded Models`` is a tree view which can be expanded/collapsed. The columns are:

  * ``Model ID`` - an Application-level identifier for this model instance
  * ``Host Memory`` - amount of host memory used by the model, displayed hierarchically, where
    the 'parent' value is the sum of its 'children'
  * ``Device Memory`` - amount of device memory used by the model, displayed just like the Host Memory

.. note ::
  The up/down/left/right keys can be used to navigate the tree view. The 'x' key expands/collapses the
  entire tree.

The bottom bar shows which application is currently displayed by highlighting
its tag using a yellow font and marking it using a pair of '>', '<' characters.
For ``neuron-rtd`` daemons, the GRPC address will be shown here instead of the tag:

|image1|

.. note ::
  The '1'-'9' keys select the current application.'a'/'d' selects the previous/next
  application on the bar.

.. |image0| image:: ../../images/nt-1.png
.. |image1| image:: ../../images/nt-2.png
