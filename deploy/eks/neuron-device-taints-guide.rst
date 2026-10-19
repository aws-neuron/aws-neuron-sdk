.. meta::
   :description: Use Neuron device taints on Amazon EKS to isolate faulty Neuron devices and nodes so the Kubernetes scheduler keeps new work off them while healthy devices keep serving.
   :keywords: Neuron device taints, EKS, Kubernetes, DRA, Neuron Observer, device health, NoExecute, NoSchedule, toleration, Trainium, UltraServer
   :date-modified: 2026-10-08

.. _neuron-device-taints:

=========================================================================
How to isolate faulty Neuron devices on Amazon EKS with device taints
=========================================================================

.. note::

   Neuron device taints are a beta feature. Details on this page may change before general availability.

Task overview
-------------

This topic shows how to use Neuron device taints on Amazon EKS to isolate an individual faulty Neuron device, or a faulty node, so that the Kubernetes scheduler keeps new work off it and can evict running work onto healthy devices. Device taints build on the :ref:`Neuron DRA driver <neuron-dra>` and work only when the DRA driver manages your Neuron devices.

What are device taints?
-----------------------

A device taint is a Kubernetes mechanism that marks an individual accelerator device as unavailable for scheduling, much like a node taint marks a whole node. Each taint has a key, a value, and an effect that tells the scheduler to keep new work off the device and, optionally, to evict work already running on it. Dynamic Resource Allocation (DRA) drivers can apply device taints to the devices they manage, and cluster admins can apply them across a cluster with a ``DeviceTaintRule``. Neuron device taints apply this mechanism to Neuron devices on Trainium platforms.

For more information, see `Device taints and tolerations <https://kubernetes.io/docs/concepts/resource-management/dynamic-resource-allocation/device-taints/>`_ in the Kubernetes documentation.

How device taint isolation works
---------------------------------

Neuron Observer runs on each Neuron node as a DaemonSet and watches the Neuron device driver's per-device health signals. On a health change, Observer pushes an event to the Neuron DRA driver, which learns of the fault within about a second and adds a *taint* to that device (or, on UltraServers, to the node). A kernel hardware-error event surfaced through Observer can taint a device within a subsecond. The Kubernetes scheduler then acts on the taint. The isolation granularity depends on the instance type:

- **Standalone Trainium instances:** the DRA driver taints only the faulty device. The node's other devices stay schedulable.
- **UltraServer instances (trn2u.\*, trn3.\*):** a claim takes a node's devices together, so the DRA driver taints the whole node. The other nodes in the UltraServer stay schedulable. No configuration is needed — the DRA driver selects device-level or node-level tainting automatically from the instance type.

All device-health taints use the single key ``neuron.aws.com/device-unhealthy``. The taint value names the specific fault, so an operator can tell from the ``ResourceSlice`` what happened. Taint effects are of three types:

* ``NoSchedule``: Blocks new workloads; leaves running ones alone.
* ``NoExecute``: Blocks new workloads and evicts running ones.
* ``None``: Purely an informational taint on the ``ResourceSlice``. It does not disrupt current/new workloads.

Workloads can opt out of a taint with a toleration. A toleration on the claim request lets a pod keep using a tainted device, and ``tolerationSeconds`` gives it a grace period before eviction. See :ref:`Tolerate a fault in your workload <neuron-device-taints-tolerate>` for more details.

Prerequisites
-------------

- **Kubernetes version:** Kubernetes 1.36 or later on the control plane, with the node AMI on a matching version. The admin-set ``DeviceTaintRule`` API requires Kubernetes 1.37 or later; see :ref:`Set manual device taints <neuron-device-taints-manual>`.
- **Neuron DRA driver:** Install and configure the :ref:`Neuron DRA driver <neuron-dra>` (version 1.3.0 or later). Device taints work only when the DRA driver manages your Neuron devices.
- **Neuron driver:** Use the Neuron driver from Neuron SDK 2.33.0 or later, which provides the per-device health signals this feature consumes. On an earlier driver, the DRA driver logs a warning and disables health monitoring.
- **Neuron Observer:** Deploy the Neuron Observer DaemonSet to your cluster. Observer ships with Neuron SDK 2.33.0 — the same release as the required Neuron driver — and watches the driver's per-device health signals, pushing fault events to the DRA driver.


Enable device health monitoring
--------------------------------

**1: Enable health monitoring on the DRA driver.**

Health monitoring is off by default. Enable it with the Neuron Helm chart:

.. code-block:: bash

   helm upgrade --install neuron-helm-chart oci://public.ecr.aws/neuron/neuron-helm-chart \
     --set "draDriver.enabled=true" \
     --set "draDriver.deviceHealthMonitoring.enabled=true"

**2: Confirm the health monitor started.**

Check the DRA driver logs and confirm it is consuming Neuron Observer events:

.. code-block:: bash

   kubectl logs -n neuron-dra-driver <driver-pod> | grep Observer
   # NeuronObserver event stream consumer started

What you'll see when a device faults
------------------------------------

When Neuron Observer reports a device fault, the DRA driver taints the affected device or node automatically, with no action required from you. What you see depends on the instance type.

.. _neuron-device-taints-standalone:

On a standalone Trainium instance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If an uncorrectable HBM error occurs on device 3, the DRA driver taints just that device on the first Observer event that reports it.

**What you'll see:** A pod starts terminating on its own.

.. code-block:: bash

   kubectl get pods -w
   NAME    READY   STATUS        RESTARTS   AGE
   pod0    1/1     Terminating   0          4m12s

**Why:** An uncorrectable HBM error hit device 3. The DRA driver tainted that one device the moment Neuron Observer reported it.

.. code-block:: bash

   kubectl logs -n neuron-dra-driver -l app=neuron-dra-driver-kubelet-plugin | grep "Applied device taint"

.. code-block:: text

   "Applied device taint from NeuronObserver FaultEvent" deviceID=3 \
     taintKey="neuron.aws.com/device-unhealthy" taintValue="hbm-uncorrectable-error" \
     taintEffect="NoExecute" components=["hbm"] suggestedAction="DO_NOT_USE" latched=true

**Confirm it yourself:** The taint is on the device in the node's ``ResourceSlice``.

.. code-block:: bash

   kubectl get resourceslices -o json \
     | jq '[.items[].spec.devices[]? | select(.taints != null) | {name, taints}]'

.. code-block:: json

   [
     {
       "name": "neuron-device-3",
       "taints": [
         {
           "key": "neuron.aws.com/device-unhealthy",
           "value": "hbm-uncorrectable-error",
           "effect": "NoExecute",
           "timeAdded": "2026-10-15T14:32:01Z"
         }
       ]
     }
   ]

For a ``NoExecute`` taint, recycle the node to return all of its devices to a healthy, schedulable state.

.. _neuron-device-taints-ultraserver:

On an UltraServer
~~~~~~~~~~~~~~~~~~

UltraServer instances group Trainium nodes together. Each node still runs its own DRA driver pod and monitors its own devices. However, because a claim takes the whole node's devices together, the node advertises a single ``neuron-node`` resource for all of them. That one resource is the only place a taint can land, so tainting is at node granularity: fault on any device affects the whole node.

**What you'll see:** The job holding the node starts terminating.

.. code-block:: bash

   kubectl get pods -w
   NAME    READY   STATUS        RESTARTS   AGE
   pod0    1/1     Terminating   0          4m12s

**Why:** An uncorrectable HBM error hit device 3. The driver log still names the device that faulted, so you can tell which one caused it.

.. code-block:: bash

   kubectl logs -n neuron-dra-driver -l app=neuron-dra-driver-kubelet-plugin | grep "Applied device taint"

.. code-block:: text

   "Applied device taint from NeuronObserver FaultEvent" deviceID=3 \
     taintKey="neuron.aws.com/device-unhealthy" taintValue="hbm-uncorrectable-error" \
     taintEffect="NoExecute" components=["hbm"] suggestedAction="DO_NOT_USE" latched=true

**Confirm it yourself:** The taint is on ``neuron-node`` rather than on an individual device.

.. code-block:: bash

   kubectl get resourceslices -o json \
     | jq '[.items[].spec.devices[]? | select(.taints != null) | {name, taints}]'

.. code-block:: json

   [
     {
       "name": "neuron-node",
       "taints": [
         {
           "key": "neuron.aws.com/device-unhealthy",
           "value": "hbm-uncorrectable-error",
           "effect": "NoExecute",
           "timeAdded": "2026-10-15T14:32:01Z"
         }
       ]
     }
   ]

What this means in practice:

- Recovery is per fault, not per device. The taint is removed only when no device on the node still reports or latches that same key and value. One device clearing while a sibling is still faulted leaves the taint in place.
- Eviction moves work to healthy nodes, not healthy devices.
- Peer nodes are flagged, not taken out of service. When a node in the same UltraServer goes unhealthy, each surviving node applies ``neuron.aws.com/peer-tainted`` to its own ``neuron-node`` resource. The effect is ``None``: it does not block scheduling and does not evict anything. Peer state is polled, so expect up to about 15 seconds before it appears or clears. Peer tainting requires the :doc:`Neuron UltraServer Operator </deploy/eks/ultraserver-operator>`.

.. _neuron-device-taints-tolerate:

Tolerate a fault in your workload
---------------------------------

By default, a ``NoExecute`` taint evicts your pod as soon as it lands. A toleration with ``tolerationSeconds`` delays that eviction, giving your workload a window to save state and exit on its own terms. Add a toleration to the device request in your ``ResourceClaimTemplate``:

.. code-block:: yaml

   apiVersion: resource.k8s.io/v1
   kind: ResourceClaimTemplate
   metadata:
     name: fault-tolerant-neurons
   spec:
     spec:
       devices:
         requests:
         - name: neurons
           exactly:
             deviceClassName: neuron.aws.com
             allocationMode: All
             tolerations:
             - key: neuron.aws.com/device-unhealthy
               operator: Exists
               effect: NoExecute
               tolerationSeconds: 120   # 2 minutes to checkpoint before eviction

With ``tolerationSeconds`` set, the pod gets that many seconds after a taint lands before eviction. Omit ``tolerationSeconds`` to tolerate the fault indefinitely. A ``NoSchedule`` taint never evicts a running pod, so most workloads need no toleration for it.

.. note::

   The grace period lets a pod shut down gracefully. Whether it can also checkpoint depends on the fault: after a data-integrity fault such as an uncorrectable memory error, the device state may be unreliable, so a checkpoint taken at that point may not be trustworthy.

.. TODO(author): Confirm from testing when SIGTERM is sent relative to the ``tolerationSeconds`` timer, the default ``tolerationSeconds`` value, and the detection-to-eviction latency to expect.

Confirm your work
-----------------

To confirm health monitoring is active, check that the DRA driver is consuming Neuron Observer events:

.. code-block:: bash

   kubectl logs -n neuron-dra-driver <driver-pod> | grep Observer


.. _neuron-device-taints-fault-mapping:

Fault-to-effect mapping
-----------------------

Every device-health taint uses the single key ``neuron.aws.com/device-unhealthy``.

The following table lists the default fault-to-effect mapping the Neuron DRA driver applies. To change it, see :ref:`Override the fault-to-effect mapping <neuron-device-taints-override>`.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Fault
     - Effect
     - Taint value
   * - hbm
     - NoExecute
     - hbm-uncorrectable-error
   * - sram
     - NoExecute
     - sram-uncorrectable-error
   * - device_unreachable
     - NoExecute
     - device-unreachable
   * - firmware_service_api
     - NoSchedule
     - firmware-service-api-error
   * - firmware_service_processor
     - NoSchedule
     - firmware-service-processor-error
   * - link_host
     - NoExecute
     - link-host-error
   * - link_intraserver
     - NoSchedule
     - link-intraserver-error
   * - link_interserver
     - NoSchedule
     - link-interserver-error
   * - link_efa
     - NoSchedule
     - link-efa-error
   * - link_switch
     - NoSchedule
     - link-switch-error
   * - fabric_error
     - NoExecute
     - fabric-error

Faults that compromise running work map to ``NoExecute`` (eviction): uncorrectable memory errors, a host-link error, a fabric error, and an unreachable device. The remaining interconnect and firmware faults map to ``NoSchedule`` (cordon): the device may still compute correctly, so the driver keeps new work off it without the disruption of eviction.

.. _neuron-device-taints-override:

Override the fault-to-effect mapping
------------------------------------

Override the default mapping to control how disruptive each fault is in your cluster.

**Create a taint-policy ConfigMap.**

List only the faults you want to change; any fault you omit keeps its default effect.

.. code-block:: yaml

   apiVersion: v1
   kind: ConfigMap
   metadata:
     name: neuron-dra-taint-policy
     namespace: neuron-dra-driver
   data:
     policy.yaml: |
       policies:
         # De-escalate: cordon on an HBM fault instead of evicting.
         hbm:
           effect: NoSchedule
         # Escalate: evict on an EFA link fault instead of cordoning.
         # Escalation requires explicit opt-in.
         link_efa:
           effect: NoExecute
           allowEscalation: true

Apply it:

.. code-block:: bash

   kubectl apply -f neuron-dra-taint-policy.yaml

Making a fault *less* disruptive than its default is allowed directly. Making a fault *more* disruptive requires ``allowEscalation: true`` on that entry, so a configuration change cannot cause fleet-wide evictions by accident.

Editing the ConfigMap is the only action required. The DRA driver watches the mounted policy file and reloads it in place when the ConfigMap changes, with no DaemonSet restart. Each update is validated before it takes effect; the driver rejects an invalid policy and keeps its last-good policy, so a bad edit does not take a node's health monitoring offline. Confirm the reload and any rejected entries in the driver log:

.. code-block:: bash

   kubectl logs -n neuron-dra-driver <driver-pod> | grep "taint policy"

.. _neuron-device-taints-manual:

Set manual device taints (DeviceTaintRule)
-------------------------------------------

In addition to health-based taints set by the DRA driver, a cluster admin can taint Neuron devices manually with the Kubernetes ``DeviceTaintRule`` API. Use this for planned maintenance, fleet-wide cordoning, or manual overrides that are independent of health monitoring.

.. note::

   The ``DeviceTaintRule`` API requires Kubernetes 1.37 or later — a higher floor than the rest of this feature, which needs Kubernetes 1.36.

A ``DeviceTaintRule`` adds a taint to devices that match a selector — by driver name, pool, device, or device class. For example, to cordon all Neuron devices on a node during maintenance:

.. code-block:: yaml

   apiVersion: resource.k8s.io/v1
   kind: DeviceTaintRule
   metadata:
     name: maintenance-node-1
   spec:
     deviceSelector:
       driver: neuron.aws.com
       pool: ip-192-168-1-10.us-west-2.compute.internal
     taint:
       key: neuron.aws.com/maintenance
       value: scheduled
       effect: NoSchedule

The taint applies as long as the ``DeviceTaintRule`` exists, and you remove it by deleting the rule. The rule's status reports how many devices and pods it affects without evicting anything.

For more information, see `Taints set by an admin <https://kubernetes.io/docs/concepts/resource-management/dynamic-resource-allocation/device-taints/#taints-set-by-an-admin>`_ in the Kubernetes documentation.

Common issues
-------------

.. rubric:: Health monitoring did not start

- **Possible solution:** Confirm the Neuron driver meets the version floor in Prerequisites and that the Neuron Observer DaemonSet is running. On an older driver, the per-device health signals are absent, so the DRA driver disables health monitoring and logs a warning. Check the driver log for that warning.

.. rubric:: A device stays tainted and does not recover

- **Possible solution:** The driver removes a taint on the first clean reading from Neuron Observer, but some faults do not clear. Uncorrectable memory errors, in particular, do not recover. For a ``NoExecute`` taint, recycle the node to return its devices to a healthy state.

Related information
-------------------

- :ref:`AWS Neuron Dynamic Resource Allocation (DRA) <neuron-dra>` — device taints require the Neuron DRA driver.
- :doc:`Neuron Helm chart </deploy/eks/helm-chart>` — install and configure the DRA driver.
- :doc:`Neuron UltraServer Operator </deploy/eks/ultraserver-operator>` — provisioning and lifecycle management for UltraServer workloads.
