Neuron runtime FAQs
===================

Q: How does Neuron connect to all the Inferentia chips in an Inf1 instance?
---------------------------------------------------------------------------

By default, a single runtime process will manage all assigned
Inferentias, including running the Neuron Core Pipeline mode. if needed,
you can configure multiple KRT processes each managing a separate group
of Inferentia chips. For more details please refer to
:ref:`nrt-overview`

Q: Where can I get logging and other telemetry information?
-----------------------------------------------------------

See this document on how to collect logs: :ref:`neuron_gatherinfo`

Q: What about RedHat or other versions of Linux?
------------------------------------------------

We dont officially support it yet.

Q: What about Windows?
----------------------

Windows is not supported at this time.

Q: How can I use Neuron in a container based environment? Does Neuron work with ECS and EKS?
--------------------------------------------------------------------------------------------

ECS and EKS support is coming soon. Containers can be configured as
shown :ref:`here <neuron-containers>`.

Q: How can I take advantage of multiple NeuronCores to run multipleinferences in parallel?
------------------------------------------------------------------------------------------

Examples of this for TensorFlow are found
:ref:`here <tensorflow-tutorials>` as well as for MXNet
:ref:`here <mxnet-tutorials>`
