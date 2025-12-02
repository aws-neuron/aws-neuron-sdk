.. meta::
   :description: This topic explores Neuron runtime core dumps in depth, using the neuron-dump tool included in the AWS Neuron SDK.
   :date-modified: 12-02-2025
   
.. _runtime-core-dump-deep-dive:   

Deep Dive: Explore Neuron runtime core dumps
=============================================

This topic explores Neuron runtime core dumps in depth and discusses the technical details of it from the perspective of an AWS Neuron expert. Some experience in AWS NeuronCore Architecture is required to understand it in full.

What you should know before reading
------------------------------------

* :doc:`AWS NeuronCore Architecture </about-neuron/arch/neuron-hardware/neuroncores-arch>`
* :doc:`Amazon EC2 AI Chips Architecture </about-neuron/arch/neuron-hardware/neuron-devices>`
* :doc:`Generating a Neuron runtime core dump </neuron-runtime/about/core-dump>`

Overview
--------

What are Neuron Runtime core dumps?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Core dumps are a snapshot of relevant runtime and hardware state to aid in debugging issues when deploying Neuron at scale.

What problems do Neuron Runtime core dumps solve?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When deploying Neuron applications at scale, there can be infrequent and difficult to reproduce errors.
Core dumps are a mechanism to capture relevant state about these errors to aid in debugging.

Who are Neuron runtime core dumps for?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Organizations who are scaling up Neuron applications and encountering sporadic issues in the fleet.

When should Neuron Runtime core dumps be used?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Diagnoising correctness issues occuring infrequently in the fleet.

How are core dumps enabled?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Core dumps are enabled by default if an executable script ``/opt/aws/neuron/bin/neuron-dump`` exists.
The package ``aws-neuronx-tools`` provides a default implementation of ``/opt/aws/neuron/bin/neuron-dump``.
Alternatively, core dumps are enabled if users install a custom version of ``/opt/aws/neuron/bin/neuron-dump``.

If users want to disable this default behavior, core dumps are disabled by defining both ``NEURON_RT_LOCAL_CORE_DUMP_DIRECTORY`` and ``NEURON_RT_S3_CORE_DUMP_PREFIX`` to an empty string::

    export NEURON_RT_LOCAL_CORE_DUMP_DIRECTORY=""
    export NEURON_RT_S3_CORE_DUMP_PREFIX=""

Alternatively, deleting ``/opt/aws/neuron/bin/neuron-dump`` also disables core dumps.

What is the core dump generation flow?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Upon execution error:

1. Neuron runtime produces state snapshots for each rank
2. Neuron runtime invokes ``neuron-dump`` to capture instance hardware state
3. ``neuron-dump`` captures environment and hardware state
4. ``neuron-dump`` can optionally be configured to upload core dump artifacts to S3

What is included in a core dump?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- tail of runtime logs (default naming: ``nrt-<instance id>-pid-<pid>.log``)
- dump of hardware state for every participating physical NeuronCore (default naming: ``<instance id>-nd<device id>-nc<core id>-pid-<pid>-tid-<tid>-lid-<log id>``)
  - installed neuron packages
  - snapshot of instruction buffers
  - semaphore values
  - DMA state
- dump of CC core state for every participating CC core (naming: ``<instance id>-nd<device id>-cc-core-<cc core id>-pid-<pid>-tid-<tid>-lid-<log id>``)
- tail of nrt error logs for every participating process (naming: ``<instance id>-nrt-pid-<pid>.log``)

neuron-dump
-------------

What is neuron-dump?
~~~~~~~~~~~~~~~~~~~~~

``neuron-dump`` is the script responsible for capturing relevant hardware state for core dumps and uploading the core dump to Amazon S3.

How is neuron-dump distributed?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A default ``neuron-dump`` is distributed as part of the ``aws-neuronx-tools`` package.

Where is neuron-dump installed?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``/opt/aws/neuron/bin/neuron-dump``

How do users customize the information included in core dumps?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To add/remove information from core dumps, uyou must install a custom ``neuron-dump`` at ``/opt/aws/neuron/bin/neuron-dump``. If you choose to install a custom ``neuron-dump`` as part of an automated script, you must install it after you install ``aws-neuronx-tools``.

What input interface does Neuron runtime provide to neuron-dump?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Neuron runtime provides input to ``neuron-dump`` by CLI flag-value pairs.
The following CLI flags are provided to ``neuron-dump``::

    --neff-name: Name from the neff header
    --neff-uuid: UUID from the neff header

    --date-time: datetime formatted as `yyyy-mm-dd-HH-MM`. This datetime represents the epoch of the initial barrier when running a collectives execution, or it falls back to the epoch from the local process if collectives context is not available.
    --pid: The process id
    --tid: The thread id
    --log-id: A process unique id given. Each execution of neuron-dump is given a unique log id for that given runtime process. Not guaranteed to be unique across processes.

    --instance-id: The instance id
    --cluster-id: the unique identifier for a single collectives execution. `0000000000000000` if collectives information is not available.

    --error-location: The libnrt api where the error occured.
    --error-code: The libnrt api return code: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-runtime/nrt-api-guide.html#api-return-codes.

    --local-output-dir: The directory specified by `NEURON_RT_LOCAL_CORE_DUMP_DIRECTORY` with format variables replaced.
    --s3-output-prefix: The prefix specified to by `NEURON_RT_S3_CORE_DUMP_PREFIX` with format variables replaced. Only included if `NEURON_RT_S3_CORE_DUMP_PREFIX` is set.

Configuring Neuron Runtime core dumps
---------------------------------------

Where are core dumps located locally on the instance?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Neuron Runtime exposes the environment variable ``NEURON_RT_LOCAL_CORE_DUMP_DIRECTORY`` to configure the local root directory of core dumps. The default value is ``/tmp/neuron-core-dump/dt-%d-cid-%c``.

Where are core dumps uploaded to in Amazon S3?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Neuron Runtime exposes the environment variable ``NEURON_RT_S3_CORE_DUMP_PREFIX`` to configure the root directory for core dumps to be uploaded to in an s3 bucket. Neuron Runtime does not perform the upload to s3. The formatted directory is provided as an argument to ``neuron-dump`` which can be configured by the user to upload the core dump to s3.

What format variables are supported for core dump paths?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The configuration environment variables ``NEURON_RT_LOCAL_CORE_DUMP_DIRECTORY`` and ``NEURON_RT_S3_CORE_DUMP_PREFIX`` support format variable substition. Neuron Runtime substitutes these variables with information from the runtime process. The formatted directories are then passed along to ``neuron-dump``::

    %d: datetime
    %c: cluster id
    %p: the process id
    %t: the thread id
    %l: the log id
    %i: the instance id

How do users ensure the root path for core dumps are unique?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Including the format variables ``%d`` (datetime) and ``%c`` (cluster id) in the path ensures uniqueness.
As well, these values are agreed upon by all participating ranks in a collectives execution, so all ranks produce their core dump in the same directory with these set::

    export NEURON_RT_LOCAL_CORE_DUMP_DIRECTORY="/your/base/path/%d-%c"
    export NEURON_RT_S3_CORE_DUMP_PREFIX="s3://your/s3/bucket/%d-%c"
