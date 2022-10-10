.. _container-troubleshooting:

Troubleshooting Neuron Containers
=================================

This document aims to provide more information on how to fix issues you
might encounter while using the Neuron Containers. For each
issue we will provide an explanation of what happened and what can
potentially correct the issue.


If your issue is not listed below or you have a more nuanced problem, contact
us via `issues <https://github.com/aws/aws-neuron-sdk/issues>`__ posted
to this repo, the `AWS Neuron developer
forum <https://forums.aws.amazon.com/forum.jspa?forumID=355>`__, or
through AWS support.

Neuron Container includes the following Neuron Components. For issues relating to 
these components inside the container refer the individual component troubleshooting
guides :ref:`general-troubleshooting`

* Neuron Runtime/Driver
* Pytorch/Tenosrflow/MXNet frameworks
* Libfabric/EFA 

The following are container specific issues

Neuron Device Not found
-----------------------

The neuron container expects the neuron devices to be exposed to the container as
referenced in :ref:`container-devices`. 

Please look at the container logs to see messages like below

::

   2022-Sep-08 17:55:23.0768    19:19    ERROR  TDRV:tdrv_get_dev_info                       No neuron device available


If the above message is seen then devices are not exposed to container

Solution
''''''''

* Refer :ref:`container-devices` and make sure the devices are exposed to container
* In kubernetes environment refer :ref:`k8s-specify-devices` to make sure neuron devices are requested in container spec



Contiguous Device ID's
-----------------------

Neuron runtime expects the inferentia/trainium device id's to be contigious. If the device id's
are not contiguous you might see error messages like below


::

   2022-Sep-08 21:52:11.0307     7:7     ERROR  TDRV:tdrv_init_mla_phase1                    Could not open the nd1

::

   2022-Sep-08 23:00:05.0667     8:8     ERROR   NRT:nrt_allocate_neuron_cores               Neuron cores are not contiguous


Solution
''''''''

* In the docker run command make sure the devices specified using --device are all contiguous
* If oci neuron hook is used and the env variable AWS_NEURON_VISIBLE_DEVICES is used then make sure the
devices specified are all contiguous
* In kubernetes environment with just the neuron device plugin running there is no guarantee that
the devices allocated will be contiguous. Make sure to run the neuron scheduler extension as specified in :ref:`neuron-k8-scheduler-ext`