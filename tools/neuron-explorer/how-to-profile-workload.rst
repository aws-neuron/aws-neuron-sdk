.. meta::
    :description: Learn how to capture a profile, launch the Neuron Explorer UI, and use the Profile Manager to analyze your workload performance.
    :date-modified: 12/02/2025

Capture and View Profiles in Neuron Explorer
================================================

Capturing Profiles
------------------
In this guide, you'll learn how to capture a profile, launch the Neuron Explorer, use the Profile Manager, and view Neuron Explorer in your IDE.

To get a better understanding of your workload's performance, you must collect the raw device traces and runtime metadata in the form of an NTFF (Neuron Trace File Format) which you can then correlate with the compiled NEFF (Neuron Executable File Format) to derive insights.

Set the following environment variables before compiling to capture more descriptive layer names and stack frame information.

.. code-block:: bash

   export XLA_IR_DEBUG=1
   export XLA_HLO_DEBUG=1

For NKI developers, set ``NEURON_FRAMEWORK_DEBUG`` in addition to the two above to enable kernel source code tracking:

.. code-block:: bash

   export NEURON_FRAMEWORK_DEBUG=1

If profiling was successful, you will see NEFF (``.neff``) and NTFF (``.ntff``) artifacts in the specified output directory similar to the following:

.. code-block:: bash

   output
   └── i-0ade06f040a13f2bf_pid_210229
       ├── 395760075800974_instid_0_vnc_0.ntff
       └── neff_395760075800974.neff

Device profiles for the first execution of each NEFF per NeuronCore are captured, and NEFF/NTFF pairs with the same prefix (for PyTorch) or unique hash (for JAX or CLI) must be uploaded together. See the section on :ref:`uploading profiles <profile-manager-upload-profile>` for more details.

PyTorch Profiling API
~~~~~~~~~~~~~~~~~~~~~

The context-managed profiling API in ``torch_neuronx.experimental.profiler`` allows you to profile specific blocks of code. To use the profiling API, import it into your application:

.. code-block:: python

   from torch_neuronx.experimental import profiler

Then, profile a block of code using the following code:

.. code-block:: python

   with torch_neuronx.experimental.profiler.profile(
           profile_type='operator',
           target='neuron_profile',
           output_dir='./output') as profiler:

Full code example:

.. code-block:: python

   import os

   import torch
   import torch.nn as nn
   import torch.nn.functional as F

   # XLA imports
   import torch_xla
   import torch_xla.core.xla_model as xm
   import torch_xla.debug.profiler as xp

   import torch_neuronx
   from torch_neuronx.experimental import profiler

   # Global constants
   EPOCHS = 2

   # Declare 3-layer MLP Model
   class MLP(nn.Module):
     def __init__(self, input_size = 10, output_size = 2, layers = [5, 5]):
         super(MLP, self).__init__()
         self.fc1 = nn.Linear(input_size, layers[0])
         self.fc2 = nn.Linear(layers[0], layers[1])
         self.fc3 = nn.Linear(layers[1], output_size)

     def forward(self, x):
         x = F.relu(self.fc1(x))
         x = F.relu(self.fc2(x))
         x = self.fc3(x)
         return F.log_softmax(x, dim=1)


   def main():
       # Fix the random number generator seeds for reproducibility
       torch.manual_seed(0)

       # XLA: Specify XLA device (defaults to a NeuronCore on Trn1 instance)
       device = xm.xla_device()

       # Start the proflier context-manager
       with torch_neuronx.experimental.profiler.profile(
           profile_type='operator',
           target='neuron_profile',
           output_dir='./output') as profiler:

           # IMPORTANT: the model has to be transferred to XLA within
           # the context manager, otherwise profiling won't work
           model = MLP().to(device)
           optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
           loss_fn = torch.nn.NLLLoss()

           # start training loop
           print('----------Training ---------------')
           model.train()
           for epoch in range(EPOCHS):
               optimizer.zero_grad()
               train_x = torch.randn(1,10).to(device)
               train_label = torch.tensor([1]).to(device)

               #forward
               loss = loss_fn(model(train_x), train_label)

               #back
               loss.backward()
               optimizer.step()

               # XLA: collect ops and run them in XLA runtime
               xm.mark_step()

       print('----------End Training ---------------')

   if __name__ == '__main__':
       main()

JAX Profiling API
~~~~~~~~~~~~~~~~~

When using the JAX context-managed profiling API, set two extra environment variables to signal the profile plugin to begin capturing device profile data when the profiling API is invoked.

.. code-block:: python

   os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
   os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = "./output"

Then, profile a block of code:

.. code-block:: python

   with jax.profiler.trace(os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"]):

Full code example:

.. code-block:: python

   from functools import partial
   import os
   import jax
   import jax.numpy as jnp

   from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
   from jax.experimental.shard_map import shard_map
   from time import sleep
   from functools import partial

   os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
   os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = "./output"

   jax.config.update("jax_default_prng_impl", "rbg")

   mesh = Mesh(jax.devices(), ('i',))

   def device_put(x, pspec):
     return jax.device_put(x, NamedSharding(mesh, pspec))

   lhs_spec = P('i', None)
   lhs = device_put(jax.random.normal(jax.random.key(0), (128, 128)), lhs_spec)

   rhs_spec = P('i', None)
   rhs = device_put(jax.random.normal(jax.random.key(1), (128, 16)), rhs_spec)


   @jax.jit
   @partial(shard_map, mesh=mesh, in_specs=(lhs_spec, rhs_spec),
            out_specs=rhs_spec)
   def matmul_allgather(lhs_block, rhs_block):
     rhs = jax.lax.all_gather(rhs_block, 'i', tiled=True)
     return lhs_block @ rhs

   with jax.profiler.trace(os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"]):
     out = matmul_allgather(lhs, rhs)
     for i in range(10):
         with jax.profiler.TraceAnnotation("my_label"+str(i)):
             out = matmul_allgather(lhs, rhs)
         sleep(0.001)


   expected = lhs @ rhs
   with jax.default_device(jax.devices('cpu')[0]):
     equal = jnp.allclose(jax.device_get(out), jax.device_get(expected), atol=1e-3, rtol=1e-3)
     print("Tensors are the same") if equal else print("Tensors are different")


.. _neuron-explorer-capture-environment-variables:
.. _neuron-explorer-non-framework-user-experience:

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

You can also control profiling with environment variables. This is useful when you can’t easily change your 
application code, such as when running an executable which calls the Neuron Runtime or in a containerized 
environment where the application code is built into the container image.

.. _neuron-explorer-core-control-variables:

Core Control Variables
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left

   * - Variable
     - Description
     - Default behavior
   * - ``NEURON_RT_INSPECT_ENABLE``
     - Set to ``1`` to enable profiling
     - Enables system profiling and disables device profiling. To control which profile types are captured, see :ref:`Profile type selection <neuron-explorer-profile-type-selection>`
   * - ``NEURON_RT_INSPECT_OUTPUT_DIR``
     - Directory for profile data output
     - Default directory for captured profile data is ``./output``

.. _neuron-explorer-profile-type-selection:

Device or System Profile Type Selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note:: 
    
    When ``NEURON_RT_INSPECT_ENABLE`` set to ``1``, ``NEURON_RT_INSPECT_SYSTEM_PROFILE`` is enabled by default (set to 1) and ``NEURON_RT_INSPECT_DEVICE_PROFILE`` is disabled by default (set to ``0``).

When ``NEURON_RT_INSPECT_ENABLE`` = 1, two different profile types are available:

.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left

   * - Variable
     - Profile type
     - Description
     - Enable capture
     - Disable capture
   * - ``NEURON_RT_INSPECT_SYSTEM_PROFILE``
     - System-level
     - Captures runtime system events and operations
     - Set to ``1``
     - Set to ``0``
   * - ``NEURON_RT_INSPECT_DEVICE_PROFILE``
     - Device-level
     - Captures detailed NeuronCore hardware metrics
     - Set to ``1``
     - Set to ``0``

.. note::

    These variables have no effect if ``NEURON_RT_INSPECT_ENABLE`` is not set to ``1``.

.. _neuron-explorer-advanced-config-vars:
  
Advanced configuration for System Profiles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left

   * - Variable
     - Profile type
     - Description
     - Default behavior
   * - ``NEURON_RT_INSPECT_SYS_TRACE_MAX_EVENTS_PER_NC``
     - System-level
     - Maximum trace events per NeuronCore before oldest events are overwritten
     - 1,000,000

.. note:: 
    
    Increasing the event limit will consume more host memory.

Capture using nccom-test with Environment Variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Profiling can be enabled using environment variables. For simplicity, we have a quick way to generate a Neuron workload through using :ref:`nccom-test <nccom-test>`. nccom-test is a benchmarking tool which is already available with Neuron AMI.

.. code-block:: shell

    export NEURON_RT_INSPECT_ENABLE=1
    export NEURON_RT_INSPECT_OUTPUT_DIR=./output
    nccom-test allr allg -b 512kb -e 512kb -r 32 -n 10 -d fp32 -w 1 -f 512

.. note::
    If you have problems with nccom-test add the --debug flag.
    If using a trn1.2xlarge instance, change -r 32 to -r 2 to use fewer neuron cores.

To understand the profiling output see this section: :ref:`Inspect Output <neuron-explorer-inspect-output>`

Capture with EKS
^^^^^^^^^^^^^^^^

Capturing a profile on EKS is most easily done through setting of environment variables as described in the section 
:ref:`Non-framework specific User Experience <neuron-explorer-non-framework-user-experience>`. By using environment 
variables, users do not need to change application code in their container image or modify their run commands. 

Update the deployment yaml to include the ``NEURON_RT_INSPECT_ENABLE`` and ``NEURON_RT_INSPECT_OUTPUT_DIR`` 
environment variables. For distributed workloads, it’s important that ``NEURON_RT_INSPECT_OUTPUT_DIR`` points to a 
directory on a shared volume which all workers have access to.

.. code-block:: yaml

    apiVersion: v1
    kind: Pod
    metadata:
    name: trn1-mlp
    spec:
    restartPolicy: Never
    schedulerName: default-scheduler
    nodeSelector:
        beta.kubernetes.io/instance-type: trn1.32xlarge
    containers:
        - name: trn1-mlp
        env:
            - name: NEURON_RT_INSPECT_ENABLE
            value: "1"
            - name: NEURON_RT_INSPECT_OUTPUT_DIR
            value: "/shared/output"
        command: ['torchrun']
        args:
            - '--nnodes=1'
            - '--nproc_per_node=32'
            - 'train_torchrun.py'
        image: ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO}:mlp
        imagePullPolicy: IfNotPresent
        resources:
            limits: 
            aws.amazon.com/neuron: 16


.. note::

    EKS users running PyTorch and JAX applications are still free to change their application code 
    and use the PyTorch or JAX Python profiling APIs if they want finer-grained control over profiling. 
    However, using the environment variables conveniently allows profiling without modifying the 
    container image or application code.


CLI
~~~

In certain cases, you may want to profile the application without requiring code modifications such as when deploying a containerized application through EKS. Note that when capturing with the CLI, profiling will be enabled for the entire lifetime of the application. If more granular control is required for profiling specific sections of the model, it is recommended to use the PyTorch or JAX APIs.

To enable profiling without code change, run your workload with the following environment variables set:

.. code-block:: bash

   export NEURON_RT_INSPECT_ENABLE=1
   export NEURON_RT_INSPECT_DEVICE_PROFILE=1
   export NEURON_RT_INSPECT_OUTPUT_DIR=./output
   python train.py

CLI reference for System Profiles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In addition to controlling profiling with environment variables, you can use the ``neuron-explorer inspect`` command line interface 
for profiling applications. This provides the same functionality as environment variables but helps you avoid typos, invalid arguments, 
and provides a useful ``--help`` command to explain available options.

.. code-block:: shell

   Usage:
   neuron-explorer [OPTIONS] inspect [inspect-OPTIONS] [userscript...]

   Application Options:
   -v, --version               Show version and exit

   Help Options:
   -h, --help                  Show this help message

   [inspect command options]
         -o, --output-dir=       Output directory for the inspection results (default: .)
         -n, --num-trace-events= Maximum number of trace events to capture when profiling. Once hitting this limit, old events are dropped

   [inspect command arguments]
   userscript:                 Run command/script that launches a Neuron workload. E.g. 'python app.py' or './runscript.sh'

Example of using System Profiles CLI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

User can provide any type of their own script to generate a Neuron workload such as Pytorch to the System Profiles CLI. 
For simplicity, we have a quick way to generate a Neuron workload 
through using ``nccom-test``. ``nccom-test`` is a benchmarking tool which is already available with Neuron AMI and ``aws-neuronx-tools`` package.

.. code-block:: shell

    ubuntu@ip-172-31-63-210:~$ neuron-explorer inspect -o inspect-output-nccom-test nccom-test allg -b 512kb -e 512kb -r 32 -n 10 -d fp32 -w 1 -f 512
    INFO[0000] Running command "nccom-test allg -b 512kb -e 512kb -r 32 -n 10 -d fp32 -w 1 -f 512" with profiling enabled
        size(B)    count(elems)    type    time:avg(us)    algbw(GB/s)    busbw(GB/s)
        524288          131072    fp32           24.15          21.71          21.03
    Avg bus bandwidth:    21.0339GB/s

.. note::
    If you have problems with nccom-test add the --debug flag.
    If using a trn1.2xlarge instance, change -r 32 to -r 2 to use fewer neuron cores.

.. _neuron-explorer-inspect-output:

``neuron-explorer inspect`` Output
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The above command traces a Neuron workload execution and saves the output to the ``inspect-output-nccom-test`` directory. 
You will see the output directory contains a single NEFF file and a device profile (NTFF) for all Neuron Cores which executed that NEFF. 
You will also see ``ntrace.pb`` and ``trace_info.pb`` files storing the system profile data.
Below showing what the outputs will look like:

.. code-block:: shell

    ubuntu@ip-172-31-63-210:~$ tree inspect-output-nccom-test
    inspect-output-nccom-test
        ├── i-012590440bb9fd263_pid_98399
        │   ├── 14382885777943380728_instid_0_vnc_0.ntff
        │   ├── 14382885777943380728_instid_0_vnc_1.ntff
        │   ├── 14382885777943380728_instid_0_vnc_10.ntff
        │   ├── 14382885777943380728_instid_0_vnc_11.ntff
        ...
        │   ├── 14382885777943380728_instid_0_vnc_8.ntff
        │   ├── 14382885777943380728_instid_0_vnc_9.ntff
        │   ├── cpu_util.pb
        │   ├── host_mem.pb
        │   ├── neff_14382885777943380728.neff
        │   ├── ntrace.pb
        │   └── trace_info.pb
        └──

    2 directories, 74 files


To view a summary of the captured profile data run the command

.. code-block:: shell

    neuron-explorer view -d inspect-output-nccom-test --output-format summary-text


.. _neuron-explorer-filtering-system-profiles:

Capture-time Filtering
----------------------

**Capture-time filtering** reduces memory usage and trace file size by only collecting specific events, but filtered data cannot be recovered later.
Configure filters before trace capture using environment variables or API functions. 
You can use NeuronCore filters to only capture events for specific NeuronCores (for example only events associated with NeuronCore 0 or all the NeuronCores on a specific NeuronDevice). 
You can use event type filters to only capture specific events (for example model execute or collectives events). 
It is possible to combine both NeuronCore and event type filters.

NeuronCore
~~~~~~~~~~

If capture is enabled for a NeuronCore then a ring buffer will be allocated in host memory for storing those core's events. Thus filtering by NeuronCore decreases host memory usage during capture.

Default Behavior
^^^^^^^^^^^^^^^^

By default, all visible NeuronCores are enabled for capture. 

Using Environment Variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell

    # Filter to capture events only from NeuronCore 0
    export NEURON_RT_INSPECT_EVENT_FILTER_NC=0

    # Filter to capture events from NeuronCores 0, 2, and 4
    export NEURON_RT_INSPECT_EVENT_FILTER_NC=0,2,4

    # Filter to capture events from a range of NeuronCores (0 through 3)
    export NEURON_RT_INSPECT_EVENT_FILTER_NC=0-3

    # Reset to default behavior
    unset NEURON_RT_INSPECT_EVENT_FILTER_NC # Back to capturing all visible cores

Using API Functions
^^^^^^^^^^^^^^^^^^^

.. code-block:: c

    #include <nrt/nrt_sys_trace.h>

    // Allocate and configure trace options
    nrt_sys_trace_config_t *config;
    nrt_sys_trace_config_allocate(&config);
    nrt_sys_trace_config_set_defaults(config);

    // Enable capture only for specific NeuronCores

    // Disable all cores since by default they are all enabled
    int num_cores = 128;
    for (int i=0; i<num_cores; i++) {
      nrt_sys_trace_config_set_capture_enabled_for_nc(config, i, false); // disable NC i
    }

    // Then enable specific cores
    nrt_sys_trace_config_set_capture_enabled_for_nc(config, 0, true);  // Enable NC 0
    nrt_sys_trace_config_set_capture_enabled_for_nc(config, 2, true);  // Enable NC 2

    // Start tracing with the configuration
    nrt_sys_trace_start(config);

    // Your application code here...

    // Stop tracing and cleanup
    nrt_sys_trace_stop();
    nrt_sys_trace_config_free(config);

Event Type
~~~~~~~~~~

Default Behavior
^^^^^^^^^^^^^^^^

By default, all event types are enabled for capture.

Getting Available Event Types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can discover all available event types using the ``nrt_sys_trace_get_event_types`` API.

.. code-block:: c

    #include <nrt/nrt_sys_trace.h>

    // Get all available event types
    const char **event_types = nullptr;
    size_t count = 0;
    NRT_STATUS status = nrt_sys_trace_get_event_types(&event_types, &count);

    if (status == NRT_SUCCESS) {
        printf("Available event types:\n");
        for (size_t i = 0; i < count; ++i) {
            printf("  %s\n", event_types[i]);
        }
        
        // Free the event types array
        for (size_t i = 0; i < count; ++i) {
            free((void*)event_types[i]);
        }
        free((void*)event_types);
    }

Using Environment Variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``NEURON_RT_INSPECT_EVENT_FILTER_TYPE`` environment variable supports:

* **Default**: If not set, all event types are captured
* **Specific event types**: Use exact event names from ``nrt_sys_trace_get_event_types()``
* **Event categories**: Use ``hardware`` or ``software`` to filter by category
* **Exclusion**: Use ``^`` prefix to exclude specific events from a category

.. code-block:: shell

    # Filter to capture only specific event types
    export NEURON_RT_INSPECT_EVENT_FILTER_TYPE=model_load,nrt_execute,runtime_execute

    # Filter to capture all hardware events
    export NEURON_RT_INSPECT_EVENT_FILTER_TYPE=hardware

    # Filter to capture all software events
    export NEURON_RT_INSPECT_EVENT_FILTER_TYPE=software

    # Filter to capture all hardware events EXCEPT cc_exec
    export NEURON_RT_INSPECT_EVENT_FILTER_TYPE=hardware,^cc_exec

    # Filter to capture all software events EXCEPT model_load
    export NEURON_RT_INSPECT_EVENT_FILTER_TYPE=software,^model_load

    # Mix categories and specific events
    export NEURON_RT_INSPECT_EVENT_FILTER_TYPE=hardware,tensor_read,tensor_write

    # Reset to default behavior
    unset NEURON_RT_INSPECT_EVENT_FILTER_TYPE  # Back to capturing all event types

The ``hardware`` group contains events that are executed on the NeuronCore. 
These are ``nc_exec_running``, ``cc_running``, ``cc_exec_barrier``, ``numerical_err``, ``nrt_model_switch``, ``timestamp_sync_point``, ``hw_notify``.
The ``software`` group contains all other events.

Using API Functions
^^^^^^^^^^^^^^^^^^^

Use the ``nrt_sys_trace_config_set_capture_enabled_for_event_type`` API to filter by event type.

.. code-block:: c

    #include <nrt/nrt_sys_trace.h>

    // Configure trace options
    nrt_sys_trace_config_t *config;
    nrt_sys_trace_config_allocate(&config);
    nrt_sys_trace_config_set_defaults(config); // By default, all event types are enabled

    // Disable specific event types (others remain enabled)
    nrt_sys_trace_config_set_capture_enabled_for_event_type(config, "device_exec", false);

    // Or disable all first, then enable only specific ones
    const char **all_event_types = nullptr;
    size_t all_count = 0;
    nrt_sys_trace_get_event_types(&all_event_types, &all_count);

    // Disable all event types first
    for (size_t i = 0; i < all_count; ++i) {
        nrt_sys_trace_config_set_capture_enabled_for_event_type(config, all_event_types[i], false);
    }

    // Enable only specific event types
    nrt_sys_trace_config_set_capture_enabled_for_event_type(config, "model_load", true);
    nrt_sys_trace_config_set_capture_enabled_for_event_type(config, "nrt_execute", true);

    // Verify which event types are enabled
    const char **enabled_types = nullptr;
    size_t enabled_count = 0;
    nrt_sys_trace_config_get_enabled_event_types(config, &enabled_types, &enabled_count);
    printf("Enabled event types: %zu\n", enabled_count);
    for (size_t i = 0; i < enabled_count; ++i) {
        printf("  %s\n", enabled_types[i]);
    }

    // Clean up memory (caller is responsible)
    for (size_t i = 0; i < enabled_count; ++i) {
        free((void*)enabled_types[i]);
    }
    free((void*)enabled_types);

    for (size_t i = 0; i < all_count; ++i) {
        free((void*)all_event_types[i]);
    }
    free((void*)all_event_types);

    // Start tracing
    nrt_sys_trace_start(config);

    // Your application code here...

    // Cleanup
    nrt_sys_trace_stop();
    nrt_sys_trace_config_free(config);


Processes-time Filtering
------------------------

**Processes-time filtering** preserves the complete trace and allows flexible analysis with different filters, but requires more memory and storage during capture.
Apply filters when viewing or processing already captured profiles. This approach allows you to 
analyze the same trace data in different ways without recapturing. The filters can be used for any 
``neuron-explorer`` output format including ``--output-format json`` and ``--output-format perfetto``.

NeuronCore
~~~~~~~~~~

Use the ``--system-trace-filter-neuron-core`` to only process events for specific NeuronCores. The IDs are local to the instance and not global IDs. 

If the ``--system-trace-filter-neuron-core`` argument is not set then events from all NeuronCores will be included in the processed trace.


**Single neuron core**

.. code-block:: shell

    neuron-explorer view -d ./output --system-trace-filter-neuron-core "0"

**Multiple neuron cores**

.. code-block:: shell

    neuron-explorer view -d ./output --system-trace-filter-neuron-core "0,1,2,3"

Event Type
~~~~~~~~~~
Use the ``--system-trace-filter-event-type`` to only process specific trace events types.

If the ``--system-trace-filter-event-type`` argument is not set then all event types will be included in the processed trace.

**Single event type**

.. code-block:: shell

    neuron-explorer view -d ./output --system-trace-filter-event-type "nrt_execute"

**Multiple event type**

.. code-block:: shell

    neuron-explorer view -d ./output --system-trace-filter-event-type "nrt_execute,nrt_load"

Instance ID
~~~~~~~~~~~

Use the ``--system-trace-filter-instance-id`` to only process events for specific ec2 instances.

If the ``--system-trace-filter-instance-id`` argument is not set then events from all instances will be included in the processed trace.

**Single instance**

.. code-block:: shell

    neuron-explorer view -d ./output --system-trace-filter-instance-id "i-abc123"

**Multiple instances**

.. code-block:: shell

    neuron-explorer view -d ./output --system-trace-filter-instance-id "i-abc123,i-def456,i-ghi789"


View Profiles
-------------

Use the ``neuron-explorer`` tool from ``aws-neuronx-tools`` to start the UI and API servers that are required for viewing profiles.

.. code-block:: bash

   neuron-explorer view

By default, the UI will be launched on port 3001 and the API server will be launched on port 3002.

If this is launched on a remote EC2 instance, use port-forwarding to enable local viewing of the profiles.

.. code-block:: bash

   ssh -i <key.pem> <user>@<ip> -L 3001:locahost:3001 -L 3002:localhost:3002

Neuron Explorer Browser UI
~~~~~~~~~~~~~~~~~~~~~~~~~~

After the above setup, navigate to ``localhost:3001`` in the browser to view the NeuronExplorer UI.

Processing only system or device profiles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To reduce processing times it is possible to skip processing of system or device profiles. Sometimes users may only be interested in one or want to start with a limited set of profiling data before exploring the full profile.

To skip processing of device profiles use the ``--ignore-device-profile`` option. To skip processing of system profiles use the ``--ignore-system-profile`` option. These options can be used with the ``--output-format`` values ``parquet`` (default), ``perfetto``, or ``json``.

For example:

.. code-block:: shell

    neuron-explorer view -d ./output --ignore-device-profile --output-format perfetto

.. _neuron-explorer-profile-manager:

Profile Manager
^^^^^^^^^^^^^^^

Profile Manager is a page for uploading artifact (NEFF, NTFF and source code) and selecting profiles to access.

.. image:: /tools/profiler/images/profile-workload-3.png

.. _profile-manager-upload-profile:



Click on the "Upload Profile" button to open the Upload Profile modal.


**Device Profile Upload**

Select "Individual Files" upload mode to upload NEFF, NTFF, and source code individually.

Select "Directory Upload" to upload profile files from a directory.

.. note::
   * "Profile name" is a required field. You cannot upload a profile with existing name unless the option "Force Upload" is checked at the bottom. Force Upload currently will overwrite the existing profile with the same name.
   * For uploading source code, the UI only supports the upload of folders, individual files, or compressed files in the gzipped tar ``.tar.gz`` archive format.

.. image:: /tools/neuron-explorer/images/device-profile-upload-ui.png


.. _profile-manager-system-profile-upload:

**System Profile Upload**

Select "Directory Upload", then in the Profile Directory drag and drop area, select the directory containing the system profile files.

The directory should contain instance sub-directories with the following: ``ntrace.pb``, ``trace_info.pb``, ``cpu_util.pb``, and ``host_mem.pb``.
For an example see the output in :ref:`neuron-explorer inspect <neuron-explorer-inspect-output>`

.. note::
   System Profile uploads only support "Directory Upload".

.. image:: /tools/neuron-explorer/images/system-profile-upload-ui.png


**Processing Status**

After uploading a profile, the processing task is shown under "User Uploaded" table. Use the "Refresh" button in the top-right to fetch the latest processing status and verify completion.


**Listing profiles**

All uploaded profiles are provided in the Profile Manager page with details such as the processing status and upload time, along with various quick access actions.

.. image:: /tools/profiler/images/profile-workload-5.png

* **Pencil button**: Rename a profile.
* **Star button**: Mark this profile as favorite profile. This profile will be shown in the User's favorites list.
* **Bulb button**: Navigate to the summary page of this profile. For more details on the summary page, see :doc:`this overview of the Neuron Explorer Summary Page </tools/neuron-explorer/overview-summary-page>`.

Clicking on the name of profile takes you to its corresponding profile page.

Neuron Explorer VSCode Extension
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The UI is also available as a VSCode extension, enabling better native integration for features such as code linking.

First, download the Visual Studio Code Extension (``.vsix``) file from https://github.com/aws-neuron/aws-neuron-sdk/releases/tag/v2.28.0.

.. grid:: 1
   :gutter: 3

   .. grid-item-card:: 
      :class-card: sd-border-2

      **Get the Neuron Explorer VSCode Extension**
      ^^^
      :download:`Neuron Explorer Visual Studio Code Extension </tools/neuron-explorer/downloads/aws-neuron.neuron-explorer-2.28.0.vsix>`

Once downloaded, open the command palette by pressing **CMD+Shift+P** (MacOS) or **Ctrl+Shift+P** (Windows), type ``> Extensions: Install from VSIX...`` and press **Enter**. When you are prompted to select a file, select ``aws-neuron.neuron-explorer-2.28.0.vsix`` and then the **Install** button (or press **Enter**) to install the extension.

.. image:: /tools/profiler/images/profile-workload-1.png

Ensure the SSH tunnel is established by following the steps above. Otherwise, specify a custom endpoint by selecting the extension in the left activity bar. Then, navigate to the "Endpoint" action on the bottom bar of your VSCode session and select "Custom endpoint", and enter ``localhost:3002``. 

.. image:: /tools/profiler/images/profile-workload-2.png

From there, navigate to the **Profile Manager** page through the extension UI in the left activity bar.

JSON Output
~~~~~~~~~~~

The ``--output-format`` json option writes processed profile data to human-readable JSON that can be used for scripting and manual inspection.

.. code-block:: shell

    neuron-explorer view -d ./output --output-format json

This will generate a ``system_profile.json`` file containing the system profile data and a ``device_profile_model_<model_id>.json`` file for each unique compiled model that was executed on a Neuron Device. 

The  system_profile.json JSON contains the following data types:

* ``trace_events``: Neuron Runtime API trace events and Framework/Application trace events containing timestamps, durations, names, and the ec2 instance-id to differentiate between events from different compute nodes in a distributed workload.

.. code-block:: json

    {
        "Neuron_Runtime_API_Event": {
            "duration": 27094,
            "group": "nrt-nc-000",
            "id": 1,
            "instance_id": "i-0f207fb2a99bd2d08",
            "lnc_idx": "0",
            "name": "nrt_tensor_write",
            "parent_id": 0,
            "process_id": "1627711",
            "size": "4",
            "tensor_id": "4900392441224765051",
            "tensor_name": "_unknown_",
            "thread_id": 1627711,
            "timestamp": 1729888371056597613,
            "type": 11
        },
        "Framework_Event": {
            "duration": 3758079,
            "group": "framework-80375131",
            "instance_id": "i-0f207fb2a99bd2d08",
            "name": "PjitFunction(matmul_allgather)",
            "process_id": "701",
            "thread_id": 80375131,
            "timestamp": 1729888382798557372,
            "type": 99999
        }
    }

* ``mem_usage``: sampled host memory usage 

.. code-block:: json

    {
        "duration": 1,
        "instance_id": "i-0f207fb2a99bd2d08",
        "percent_usage": 9.728179797845964,
        "timestamp": 1729888369286687792,
        "usage": 51805806592
    }

* ``cpu_util``: sampled CPU utilization. Results are provided per core and per ec2 instance involved in a distributed workload

.. code-block:: json

    {
        "cpu_id": "47",
        "duration": 1,
        "instance_id": "i-0f207fb2a99bd2d08",
        "timestamp": 1729888371287337243,
        "util": 2.3255813
    },


View in Perfetto
~~~~~~~~~~~~~~~~

Users can view their Neuron Explorer profiles in Perfetto. Please see :doc:`view-perfetto` for more information.

.. note::
    New Neuron Explorer features released in 2.27 and onwards may not be supported in Perfetto. For the full user experience and features set, please use the Neuron Explorer UI or VSCode Integration.


Troubleshooting
---------------

Incomplete JAX Profiles
~~~~~~~~~~~~~~~~~~~~~~~

If your JAX profile has fewer events than expected or lacks the Runtime API trace, check whether 
``jax.profiler.stop_trace`` is being called inside a ``with jax.profiler.trace`` context block. 
This can prematurely stop tracing. Use ``jax.profiler.stop_trace`` only when profiling was started 
with ``jax.profiler.start_trace``, not when using the context-managed ``with jax.profiler.trace`` API.

Also when using ``jax.profiler`` within your script ensure that the 
environment variable ``NEURON_RT_INSPECT_ENABLE`` is not set to 1. 
Additionally, ensure that ``NEURON_RT_INSPECT_OUTPUT_DIR`` is set to 
the correct output directory and this is the output directory passed to 
``with jax.profiler.trace``.

Dropped Events in System Profile
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When processing a system profile, you may see a warning indicating that some trace events were dropped during capture.

.. code-block:: shell

    WARN[0000] Warning: 1001 trace events were dropped during capture (stored 530560 out of 531561 total events). Consider increasing buffer size, reducing trace duration, or filtering events.

This means during capture the trace event buffers filled and oldest events were overwritten. If you need to avoid dropping events for the full duration of your workload consider the following adjustments:

* Increase buffer size by setting ``NEURON_RT_INSPECT_SYS_TRACE_MAX_EVENTS_PER_NC`` (see :ref:`Profile Capture Environment Variables <neuron-explorer-capture-environment-variables>`). This will increase host memory usage.
* Apply capture-time filters (NeuronCores / event types) (see :ref:`Filtering System Profiles <neuron-explorer-filtering-system-profiles>`.)
* Shorten profiled region: limit the code span under the profiling context / runtime.
