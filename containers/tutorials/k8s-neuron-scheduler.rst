.. _k8s-neuron-scheduler-ext:

Neuron scheduler extension is required for scheduling pods that require more than one Neuron core or device resource. Refer :ref:`k8s-neuron-scheduler-flow` for details on how the neuron scheduler extension works. Neuron scheduler extension filter out nodes with non-contiguous core/device ids and enforces allocation of contiguous core/device ids for the PODs requiring it.

.. tab-set::

   .. tab-item:: Multiple Scheduler Approach

      .. include:: /containers/tutorials/k8s-multiple-scheduler.rst

   .. tab-item:: Default Scheduler Approach

      .. include:: /containers/tutorials/k8s-default-scheduler.rst