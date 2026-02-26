.. _nccom-test:

======================
NCCOM-TEST User Guide
======================

.. contents:: Table of contents
    :local:
    :depth: 2

Overview
--------

**nccom-test** is a benchmarking tool for evaluating Collective Communication operations on AWS Trainium and Inferentia instances. It supports Trn1, Trn2, Trn3, and Inf2 instance types. The tool can assess performance across multiple instances or perform quick environment sanity checks before running more complex workloads. While single-instance benchmarking is supported for all compatible instance types, multi-instance benchmarking is limited to Trainium instances (Trn1, Trn2, and Trn3). 
To execute collective operations, **nccom-test** will generate, and then execute, NEFFs (Neuron Executable File Format) containing several collective operation instructions.

.. note::

    On Inf2 instances, only single-instance benchmarking is supported. Running a multi-node nccom-test benchmark
    will result in an error.

Using nccom-test
----------------

Here is a simple example which will run a 2 worker (ranks) all-reduce with a total size of 32MB:


.. code-block::

    nccom-test -r 2 allr
         size(B)    count(elems)     type    time(us)    algbw(GB/s)    busbw(GB/s)
        33554432        33554432    uint8         768          40.69          40.69
    Avg bus bandwidth:      40.6901GB/s


Output description
^^^^^^^^^^^^^^^^^^

The command will output a table containing several columns containing performance metrics.
There will be a line for every requested data size (by default the data size is 32MB as
seen in the previous example).

.. list-table::
    :widths: 40 260
    :header-rows: 1

    * - Column name
      - Description
    * - size(B)
      - Size in bytes for the data involved in this collective operation
    * - count(elems)
      - Number of elements in the data involved in this collective operation. For example, if **size(B)** is 4 and **type** is fp32,
        then **count** will be 1 since one single fp32 element has been processed.
    * - type
      - Data type for the processed data. Can be: **uint8**, **int8**, **uint16**, **int16**, **fp16**, **bf16**, **int32**, **uint32**, **fp32**
    * - time(us)
      - Time in microseconds representing the average of all durations for the Collective Communication operations executed during the benchmark.
    * - algbw(GB/s)
      - Algorithm bandwidth in gibibytes (1GiB = 1,073,741,824 bytes) per second which is calculated as **size(B)** / **time(us)**
    * - busbw(GB/s)
      - Bus bandwidth - bandwidth per data line in gibibytes per second - it provides a bandwidth number that is independent from the number of ranks (unlike **algbw**).
        For a more in-depth explanation on bus Bandwidth, please refer to `Bus Bandwidth Calculation`_
    * - algorithm (optional)
      - Algorithm used to execute this collective operation (e.g. Ring, Mesh, RDH)
    * - Avg bus bandwidth
      - Average of the values in the busbw column
  

.. _Bus Bandwidth Calculation:
**Bus Bandwidth Calculation:**

The purpose of bus bandwidth is to provide a number reflecting how optimally hardware is used, normalizing for different rank counts.

Given the following:

- ``r`` as the number of ranks participating in a collective operation
- ``s`` as the size of the collective operation
- ``B`` as the bus bandwidth of a single rank
- ``t`` latency of the operation

Let's take an AllGather operation as an example. To complete an AllGather operation with ``r`` ranks, each rank must transfer ``r-1`` data chunks of size ``s/r``. Therefore, with a bandwidth of ``B``, the latency (``t``)
of the operation would be:

.. code-block::

    t = ((number of chunks to transfer) * (size of each chunk)) / (bandwidth of rank)
    t = ((r-1) * (s/r)) / B

However, for a given collective operation result, we have the latency, but not the bandwidth of each rank. Rearranging to solve for bus bandwidth, we get:

.. code-block::

  B = ((r-1) * (s/r)) / t

which, given ``algbw = s / t``, can also be rewritten as:

.. code-block::

  B = ((r-1) / r) * algbw

Using this formula, we can calculate the bus bandwidth, ``B``, for an AllGather collective operation among ``r`` ranks with size ``s`` that took ``t`` seconds.

We can now directly compare the calculated bus bandwidth to the actual hardware bandwidth to see how well the hardware is being utilized. For different operations that transfer a different
number of chunks, the bandwidth calculation changes slightly, with our algbw factor ``(r-1) / r`` changing depending on the collective operation:

.. list-table::
    :widths: 40 40
    :header-rows: 1

    * - Collective Operation
      - Bus Bandwidth Factor
    * - All-Reduce
      - ``(2 * (r-1)) / r``
    * - All-Gather
      - ``(r-1) / r``
    * - Reduce-Scatter
      - ``(r-1) / r``
    * - Send-Receive
      - 1
    * - All-to-All
      - ``(r-1) / r``
    * - Permute
      - 1
    * - All-to-Allv
      - ``(r-1) / r``



CLI arguments
^^^^^^^^^^^^^

Required Arguments:
~~~~~~~~~~~~~~~~~~~

.. list-table::
    :widths: 40 80 260
    :header-rows: 1

    * - Argument
      - Default value
      - Description
    * - <cc operation>
      - N/A, required argument
      - The type of Collective Communication operation to execute for this benchmark.
        Supported types:

            - ``all_reduce`` / ``allr``: All-Reduce
            - ``all_gather`` / ``allg``: All-Gather
            - ``reduce_scatter`` / ``redsct``: Reduce-Scatter
            - ``sendrecv``: Send-Receive
            - ``alltoall``: All-to-All
            - ``permute``: Permute
            - ``alltoallv``: All-to-Allv (Currently only supported for inter-node configurations)
    * - ``-r, --nworkers``
      - N/A, required argument
      - Total number of workers (ranks) to use

Benchmark Configuration:
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
    :widths: 40 80 260
    :header-rows: 1

    * - Argument
      - Default value
      - Description
    * - ``-N, --nnodes``
      - 1
      - Total number of nodes (instances) to use. The number of workers will be divided equally across all nodes.
        If this argument is greater than 1, `MPI Execution`_ or `Slurm Execution`_ will need to be used.
    * - ``-b, --minbytes``
      - 32M
      - The starting size for the benchmark
    * - ``-e, --maxbytes``
      - 32M
      - The end size for the benchmark. **nccom-test** will run benchmarks for all sizes between ``-b, --minbytes`` and
        ``-e, --maxbytes``, increasing the size by either ``-i, --stepbytes`` or ``--f, --stepfactor`` with every run.
    * - ``-i, --stepbytes``
      - (``--maxbytes`` - ``--minbytes``) / 10
      - Amount of bytes with which to increase the benchmark's size on every subsequent run.
        For example, for this combination of arguments: ``-b 8 -e 16 -i 4``, the benchmark will
        be ran for the following sizes: 8 bytes, 12 bytes, 16 bytes.
    * - ``-f, --stepfactor``
      - N/A
      - Factor with which to increase the benchmark's size on every subsequent run.
        For example, for this combination of argument values: ``-b 8 -e 32 -f 2``, the benchmark will
        be ran for the following sizes: 8 bytes, 16 bytes, 32 bytes.

.. note::

    All arguments that take a size in bytes will also accept larger size units, for example:
    ``-f 2048`` can be written as ``-f 2kb`` or ``-f 1048576`` can be written as ``-f 1MB``.

Iteration Configuration:
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
    :widths: 40 80 260
    :header-rows: 1

    * - Argument
      - Default value
      - Description
    * - ``-n, --iters``
      - 20
      - Number of Collective Communication operations to execute during the benchmark.
    * - ``-w, --warmup_iters``
      - 5
      - Number of Collective Communication operations to execute as warmup during the benchmark. 
        The warmup operations will execute prior to any of the measured operations and their performance will be not be used calculate the reported statistics.
    * - ``-I, --neff_iters``
      - N/A
      - Number of times to execute the NEFF with Collective Communication operations during the benchmark. 
    * - ``-W, --neff_warmup_iters``
      - N/A
      - Number of times to execute the NEFF with Collective Communication operations as warmup during the benchmark. All collective operations in a warmup NEFF execution will be ignored when calculating statistics.

To execute collective operations, ``nccom-test`` will generate, and then execute, NEFFs (Neuron Executable File Format) containing several collective operation instructions.
The above flags control how many collective operations are generated, run, and measured.

There are two primary modes for controlling the number of collective operations run:

1. If neither the ``neff_iters`` nor the ``neff_warmup_iters`` flag is supplied, ``iters + warmup_iters`` will be treated as the desired total number of
   operations to be run. If necessary, ``nccom-test`` will spread this total number of operations out across several NEFFs.

2. If the user desires more control over how collectives operation execution should be organized, they should use the ``neff_iters`` and ``neff_warmup_iters``
   flags. When these flags are used, the ``iters`` and the ``warmup_iters`` flags now represent the number of operations in a single NEFF. The NEFF itself will be repeatedly run
   ``neff_iters + neff_warmup_iters`` times.

Examples:

- ``-n 15``, ``-w 5``, ``-I 10``, would result in 200 Collective Communication operations being run with 150 being measured:
  The generated NEFF will have 20 (15 measured, 5 warmup) ops and the NEFF will be run 10 times.
- ``-n 15``, ``-w 5``, ``-I 10``, ``-W 5``, would result in 300 Collective Communication operations being run with 150 being measured:
  The generated NEFF will have 20 (15 measured, 5 warmup) ops and the NEFF will be run 15 (10 measured, 5 warmup) times
    

Input/Output Data:
~~~~~~~~~~~~~~~~~~

.. list-table::
    :widths: 40 80 260
    :header-rows: 1

    * - Argument
      - Default value
      - Description
    * - ``-d, --datatype``
      - ``uint8``
      - Data type for the data used by the benchmark. Supported types: ``uint8``, ``int8``, ``uint16``, ``int16``,
        ``fp16``, ``bf16``, ``uint32``, ``int32``, ``fp32``. Input data will be zero filled, unless ``--check`` is
        provided in which case it will be filled with either pseudo-random data or ones.
    * - ``-c, --check``
      - N/A
      - If provided, validates correctness of the operations. Can additionally specify options: ``random`` (default) or ``all_ones``.
        For an explanation of these options, see `Data Integrity`_.
        This will not impact device execution time and collective operation performance (time, algbw, and busbw),
        but will slightly increase the overall execution time.
    * - ``--seed``
      - N/A
      - Seed to use while generating pseudo-random data for ``random`` correctness check with ``--check`` flag
    * - ``--unique-buff``
      - false
      - Use a unique buffer for the input and output of every collective operation. When using this flag, each collective operation in a NEFF will use a
        different in-memory input/output buffer than every other operation. For All-Gather operations run with certain algorithms (e.g. Mesh, RDH),
        there is additional handshaking for output buffers, and using unique buffers may improve collective operation performance.
    * - ``--coalesced-cc-size-ratio``
      - N/A
      - List representing the ratio with which to split the input tensor into multiple tensors for coalesced, collective operations. Given a size of ``4MB`` and a ``coalesced-cc-size-ratio`` of ``[1,2,1]``, each collective
        operation would actually consist of 3 parallel, coalesced operations of sizes: ``1MB``, ``2MB``, and ``1MB``.
    * - ``--shared-output-buff``
      - false
      - For the CC operation, use a single, shared, HBM output buffer between 2 neuron cores in the same HBM domain.
    * - ``--alltoallv-metadata``
      - N/A
      - For ``alltoallv`` collective operation, a ``json`` file containing send counts, send displacements, receive counts, and receive displacements for the collective operation. 
        Counts specify number of elements to send/receive between ranks, displacements specify where in buffer to send/receive data.
        Length of count and displacement arrays should equal size of replica group over which ``alltoallv`` collective operation is performed. 
        If one metadata entry is provided, it applies to all ranks, otherwise, specify one entry per rank. `AlltoAllV Example`_.

.. _Data Integrity:

Data Integrity:

If the ``--check`` flag is provided when running ``nccom-test``, the correctness of the CC operations will be verified. There are currently two modes for verification: ``random`` (the default used when only ``--check`` is provided)
and ``all_ones``. 

1. The ``random`` mode will fill each input tensor with pseudo-random data and then, on the CPU, calculate a expected golden output. After collective operation execution,
   the output tensor of the operation will be compared against the calculated golden tensor. For non-integral types (e.g. ``fp16``, ``fp32``), golden comparison will use tolerances.
   For operations in which all participating ranks should finish with identical outputs (e.g ``allr``, ``allg``), there will also be a check between ranks to ensure this.
   If the ``random`` check fails, input, output, and golden tensors will be saved to disk for further investigation. The ``--seed`` flag can be used to set the seed for the
   pseudo-random input tensor generation. Otherwise, the seed value will be based on the current time and logged.

2. The ``all_ones`` mode will fill each input tensor with the value ``1``. A single, golden value\: ``G``, will be calculated based on the operation. For example, the golden value\: ``G``
   for an All-Reduce with 16 ranks will be ``16``. After operation execution, ``nccom-test`` will verify each output tensor is filled with ``G``.

``random`` mode should be preferred for more rigorous verification. However, for quicker, more easily understood verification, ``all_ones`` should be preferred.

.. _MPI Execution:
MPI Execution:
~~~~~~~~~~~~~~~

.. list-table::
    :widths: 40 80 260
    :header-rows: 1

    * - Argument
      - Default value
      - Description
    * - ``-s, --hosts``
      - N/A
      - Hosts on which to run execution.
    * - ``--hosts-file``
      - N/A
      - File containing hosts on which to run execution. One host specified per line.
    * - ``--mpi-log-dir``
      - N/A
      - If specified, logs from each node in ``mpi`` multi-node benchmark will be saved to a unique file within the specified directory

To use ``mpi`` mode, provide all hosts for your invocation, either with the ``--hosts`` flag or a ``~/hosts`` file, and set the ``NEURON_RT_ROOT_COMM_ID`` environment variable to the IP address of the first host listed and any free port.
Depending on your environment, ``mpi`` may require passwordless SSH access to each host in your invocation. See the `Open MPI SSH documentation <https://docs.open-mpi.org/en/v5.0.x/launching-apps/ssh.html#launching-with-ssh>`_ for details.

Example:

``NEURON_RT_ROOT_COMM_ID=10.1.4.145:45654 nccom-test -r 64 -N 2 -d fp32 allr --hosts 10.1.4.145 10.1.4.138``

The above command will invoke a ``neuron-bench`` process on both hosts listed, to execute the collective operations, using 32 ranks from each host.
Latency data will be reported back from each host and collected on the host on which the ``nccom-test`` command was invoked. 
The host on which the ``nccom-test`` command is invoked should usually be one of the provided hosts, but it can be another unrelated host, as long as it can invoke MPI processes
on the provided hosts.


.. _Slurm Execution:
Slurm Execution:
~~~~~~~~~~~~~~~~

.. list-table::
    :widths: 40 80 260
    :header-rows: 1

    * - Argument
      - Default value
      - Description
    * - ``-S, --slurm-mode``
      - false
      - Use ``srun`` to run benchmark on ``slurm``-based cluster
    * - ``-u, --slurm-vcpus-per-node``
      - Minimum CPU count amongst all nodes
      - Number of vCPUs available per node in ``slurm`` allocation
    * - ``--slurm-setup-script``
      - N/A
      - Script to run on each node in ``slurm`` allocation before executing benchmark. Can use ``default`` to run 
        a default script installing the latest Neuron software.
    * - ``--slurm-job-id``
      - alloc
      - Specify jobId for ``slurm`` allocation to execute benchmark on. By default, will create a new allocation to execute benchmark on.
    * - ``--slurm-use-head-node-neuron-bench``
      - false
      - Copy ``neuron-bench`` binary from head node to all nodes in allocation

To use ``slurm`` mode, specify the ``--slurm-mode`` flag. When using slurm mode, ``nccom-test`` invocations should be run from the head node of the slurm cluster. 
Users can either use an existing slurm job by providing a job id, or have ``nccom-test`` allocate one for you. 
Additionally, users can provide a path to a setup script to run on each slurm node before execution. Users can alternatively specify ``default`` to use a supplied default setup script.

Examples:

``nccom-test -r 64 -N 2 allr --slurm-mode --slurm-setup-script path/to/my/custom-setup-script.sh``

The above command will execute collective operation across two nodes using slurm. Slurm will allocate a job with two nodes before beginning execution and will run the ``custom-setup-script.sh``
on each node before executing any collective operations.


``nccom-test -r 64 -N 2 allr --slurm-mode --slurm-job-id 12345``

The above command will use an existing slurm allocation (``jobId: 12345``) with no setup.


Output:
~~~~~~~

.. list-table::
    :widths: 40 80 260
    :header-rows: 1

    * - Argument
      - Default value
      - Description
    * - ``--non-interactive``
      - false
      - Do not display any animation or progress indicator.
    * - ``--report-to-json-file``
      - N/A
      - Persist config and results to specified JSON file if a filepath is provided.
    * - ``-t, --stats``
      - avg
      - Latency (time) statistics to display in the final output. Currently supports ``avg`` and any percentile (e.g ``p15``, ``p50``, ``p90``).
    * - ``--show-algorithm``
      - false
      - Show which algorithm (e.g. Ring, Mesh, RDH) was used to execute the collective operation in ``nccom-test`` output.
        Currently, any hierarchical algorithms used will be displayed as ``hier``, and will not include any sub-algorithms.
    * - ``--show-input-output-size``
      - false
      - Print or save to JSON per rank input and output sizes in B.
    * - ``--debug``
      - false
      - Show debug logs from execution of ``nccom-test`` and ``neuron-bench`` in realtime. Enables ``non-interactive`` mode implicitly.


SBUF Collectives:
~~~~~~~~~~~~~~~~~

.. list-table::
    :widths: 40 80 260
    :header-rows: 1

    * - Argument
      - Default value
      - Description
    * - ``--sb2sb``
      - false
      - Indicates whether to allocate input, output, and scratch-buffer on SBUF (rather than HBM).  This may result in improved performance.
    * - ``--input-shape``
      - N/A
      - Provide input tensor dimensions in format: ``[step0,step1][num_elem0,num_elem1]``. ``step0/num_elem0`` correspond to the free dimension of the SBUF, while ``step1/num_elem1`` correspond to the partition dimension of the SBUF.
    * - ``--output-shape``
      - N/A
      - Provide output tensor dimensions in format: ``[step0,step1][num_elem0,num_elem1]``. ``step0/num_elem0`` correspond to the free dimension of the SBUF, while ``step1/num_elem1`` correspond to the partition dimension of the SBUF.
    * - ``--cc-dim``
      - 1
      - Control dimensions of tensor concatenation. Either concatenate tensor in free dimension (``cc-dim = 0``) or concatenate in partition dimension first and wrap around in free dimension second (``cc-dim = 1``)


Replica Group:
~~~~~~~~~~~~~~

Flags to control which subset of ranks a collective operation will be executed on.

.. list-table::
    :widths: 40 80 260
    :header-rows: 1

    * - Argument
      - Default value
      - Description
    * - ``--data-parallel-dimension``
      - N/A
      - Run the given collective operation in parallel across multiple sub-groups of size ``data-parallel-dimension``. For 128 ranks and data parallel dimension of 2, 
        there would be 64 parallel collective operations happening at the same time, each with 2 ranks. Primarily intended for multi-node executions with one-rank-per-node
        replica groups.
    * - ``--custom-replica-group``
      - N/A
      - Provide the JSON file for custom-defined replica groups.
    * - ``--custom-src-target-pairs``
      - N/A
      - Provide the JSON file for custom-defined source_target_pairs for the collective permute operation.

Additional Flags:
~~~~~~~~~~~~~~~~~

.. list-table::
    :widths: 40 80 260
    :header-rows: 1

    * - Argument
      - Default value
      - Description
    
    * - ``--vcpu-pin-mode``
      - false
      - Pin CPU thread for each rank to a given CPU.  
    * - ``--data-collector-port``
      - 60006
      - If running ``nccom-test`` in multi-node mode or on another node, a data collector is used to gather latencies from all nodes in benchmark.
        Port to use for data collector.
    * - ``--data-collector-host``
      - current host
      - Hostname or IP address of node to use as data collector, all latencies from other nodes will be sent to this host

Environment Variables
^^^^^^^^^^^^^^^^^^^^^
In addition to CLI arguments, there are also several environment variables which can be used to alter how collectives run inside ``nccom-test``

.. list-table::
    :widths: 40 80 260
    :header-rows: 1

    * - Environment Variable
      - Default value
      - Description
    * - ``NEURON_LOGICAL_NC_CONFIG``
      - 2 for ``trn2`` and ``trn3``. 1 for ``inf2`` and ``trn1``
      - Controls how many physical NeuronCores are grouped to make up a logical NeuronCore.

Users may also find certain Neuron Runtime environment variables useful with ``nccom-test`` executions. See :ref:`nrt-configuration`

Examples
^^^^^^^^

.. note::

    Performance data shown in these examples should not be considered up-to-date. For the latest performance
    data, please refer to the performance section.


Single Instance Examples
~~~~~~~~~~~~~~~~~~~~~~~~

- Quick environment validation

    .. code-block::

        nccom-test -r 2 allr
            size(B)    count(elems)     type    time(us)    algbw(GB/s)    busbw(GB/s)
            33554432        33554432    uint8         768          40.69          40.69
        Avg bus bandwidth:      40.6901GB/s


    If a problem was found, it can be reported in two possible ways:

    - Immediately:

        .. code-block::

            nccom-test -r 2 allr
            Neuron DKMS Driver is not running! Read the troubleshooting guide at: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-runtime/nrt-troubleshoot.html#neuron-driver-installation-fails


    - After a benchmark attempt:

        .. code-block::

            nccom-test -r 2 allr
                 size(B)    count(elems)    type    time(us)    algbw(GB/s)    busbw(GB/s)
                33554432    Failure running neuron-bench - log file /tmp/nccom_test_log_7pqpdfjf.log
            1 errors found - test failed


        In this case, further information about the error can be found in the ``neuron-bench`` log file.

- 2 rank all-reduce on a single instance for sizes ranging from 1MiB to 1GiB with a step of 4x

    .. code-block::

        nccom-test -r 2 --minbytes 1kb --maxbytes 1gb --stepfactor 4 --datatype fp32 allr
               size(B)    count(elems)    type    time(us)    algbw(GB/s)    busbw(GB/s)
                  1024             256    fp32          58           0.02           0.02
                  4096            1024    fp32          58           0.07           0.07
                 16384            4096    fp32          58           0.26           0.26
                 65536           16384    fp32          58           1.05           1.05
                262144           65536    fp32          60           4.07           4.07
               1048576          262144    fp32          68          14.36          14.36
               4194304         1048576    fp32         107          36.51          36.51
              16777216         4194304    fp32         332          47.06          47.06
              67108864        16777216    fp32        1214          51.48          51.48
             268435456        67108864    fp32        4750          52.63          52.63
            1073741824       268435456    fp32       18930          52.83          52.83
        Avg bus bandwidth:      23.6671GB/s


- 32 rank all-gather on a single instance for sizes ranging from 1KiB to 1MiB with a step of 8x, with correctness checking


.. code-block::

        nccom-test -r 32 --minbytes 1kb --maxbytes 1mb --stepfactor 8 --datatype fp32 --check allg
        size(B)    count(elems)    type    time(us)    algbw(GB/s)    busbw(GB/s)
        1024             256    fp32         151           0.01           0.01
        8192            2048    fp32         149           0.05           0.05
       65536           16384    fp32         150           0.41           0.39
      524288          131072    fp32         179           2.73           2.64
    Avg bus bandwidth:      0.7731GB/s

- Specify the custom source target pairs as a JSON file for the collective permute operator ``--custom-src-target-pairs``.

.. code-block::

    nccom-test -r 8 --custom-src-target-pairs pairs.json permute
    size(B)    count(elems)     type    time:avg(us)    algbw(GB/s)    busbw(GB/s)
    33554432        33554432    uint8          894.24          37.52          37.52
    Avg bus bandwidth:	37.5230GB/s

    cat pairs.json
    {
        "src_target_pairs": [
            [
                [0, 1],
                [1, 0],
                [2, 3],
                [3, 2],
                [4, 4],
                [5, 5],
                [6, 6],
                [7, 7]
            ]
        ]
    }


- Reporting the input and output size explicitly with ``--show-input-output-size``.

.. code-block::

    nccom-test -r 32 --minbytes 1kb --maxbytes 1mb --stepfactor 8 --datatype fp32 --check allg --show-input-output-size
    size(B)    count(elems)    total_input_size(B)    total_output_size(B)    type    time:avg(us)    algbw(GB/s)    busbw(GB/s)
       1024             256                     32                    1024    fp32            6.16           0.17           0.16
       8192            2048                    256                    8192    fp32            6.48           1.26           1.23
      65536           16384                   2048                   65536    fp32            8.17           8.02           7.77
     524288          131072                  16384                  524288    fp32           23.16          22.64          21.93
    Avg bus bandwidth:      7.7715GB/s

- Getting percentile latency results with ``--stats``

.. code-block::

    nccom-test -r 8 --minbytes 1kb --maxbytes 1mb --stepfactor 8 --datatype fp32 --stats avg p25 p50 p90 p99 --iters 1000 allg
    size(B)    count(elems)    type    time:avg(us)    time:p25(us)    time:p50(us)    time:p90(us)    time:p99(us)    algbw(GB/s)    busbw(GB/s)
       1024             256    fp32            10.0              10              10              11              12           0.10           0.09
       8192            2048    fp32           10.22              10              10              11              12           0.80           0.70
      65536           16384    fp32           11.31              11              11              13              13           5.80           5.07
     524288          131072    fp32           14.83              14              15              16              17          35.34          30.92
    Avg bus bandwidth:	9.1966GB/s

- Example results as JSON with ``--report-to-json-file``

.. code-block::

    nccom-test -r 32 --minbytes 1kb --maxbytes 1mb --stepfactor 8 --datatype fp32 --check allg --report-to-json-file nccom-results.json
    size(B)    count(elems)    type    time:avg(us)    algbw(GB/s)    busbw(GB/s)
       1024             256    fp32            6.19           0.17           0.16
       8192            2048    fp32            6.55           1.25           1.21
      65536           16384    fp32            8.18           8.01           7.76
     524288          131072    fp32           23.11          22.69          21.98
    Avg bus bandwidth:      7.7775GB/s

    python3 -m json.tool nccom-results.json
    {
        "results": [
            {
                "size(B)": 1024,
                "count(elems)": 256,
                "type": "fp32",
                "algbw(GB/s)": 0.16553675170497603,
                "busbw(GB/s)": 0.16036372821419553,
                "time:avg(us)": 6.19
            },
            {
                "size(B)": 8192,
                "count(elems)": 2048,
                "type": "fp32",
                "algbw(GB/s)": 1.2500906056270864,
                "busbw(GB/s)": 1.21102527420124,
                "time:avg(us)": 6.55
            },
            {
                "size(B)": 65536,
                "count(elems)": 16384,
                "type": "fp32",
                "algbw(GB/s)": 8.008982241741455,
                "busbw(GB/s)": 7.758701546687035,
                "time:avg(us)": 8.18
            },
            {
                "size(B)": 524288,
                "count(elems)": 131072,
                "type": "fp32",
                "algbw(GB/s)": 22.688776793562784,
                "busbw(GB/s)": 21.97975251876395,
                "time:avg(us)": 23.11
            }
        ]
    }

- Example results with ``--show-algorithm`` flag

.. code-block::

    nccom-test -r 16 allr -b 4 -e 1gb -f 16 -d fp32 --show-algorithm
    size(B)    count(elems)    type    time:avg(us)    algbw(GB/s)    busbw(GB/s)    algorithm
            4               1    fp32          299.91           0.00           0.00         mesh
           32               8    fp32          299.69           0.00           0.00         mesh
          512             128    fp32          299.82           0.00           0.00         mesh
         8192            2048    fp32          299.74           0.03           0.05         mesh
       131072           32768    fp32          574.15           0.23           0.43         mesh
      2097152          524288    fp32          686.32           3.06           5.73          rdh
     33554432         8388608    fp32         2754.15          12.18          22.84    kangaring
    536870912       134217728    fp32         9689.51          55.41         103.89    kangaring
    Avg bus bandwidth:	16.6181GB/s


Multiple Instances Example
~~~~~~~~~~~~~~~~~~~~~~~~~~

- 64 rank all-reduce on two instances for sizes ranging from 8 bytes to 1GiB with a step of 2x, running 50 ops

    .. code-block::

        NEURON_RT_ROOT_COMM_ID=10.1.4.145:45654 nccom-test -r 64 -N 2 -b 8 -e 1GB -f 2 -n 50 -w 5 -d fp32 allr --hosts 127.0.0.1 10.1.4.138
               size(B)    count(elems)    type    time(us)    algbw(GB/s)    busbw(GB/s)
                     8               2    fp32         520           0.00           0.00
                    16               4    fp32         520           0.00           0.00
                    32               8    fp32         523           0.00           0.00
                    64              16    fp32         525           0.00           0.00
                   128              32    fp32         553           0.00           0.00
                   256              64    fp32         709           0.00           0.00
                   512             128    fp32         782           0.00           0.00
                  1024             256    fp32         840           0.00           0.00
                  2048             512    fp32         881           0.00           0.00
                  4096            1024    fp32         916           0.00           0.01
                  8192            2048    fp32        1013           0.01           0.01
                 16384            4096    fp32        1031           0.01           0.03
                 32768            8192    fp32        1174           0.03           0.05
                 65536           16384    fp32        1315           0.05           0.09
                131072           32768    fp32        1315           0.09           0.18
                262144           65536    fp32        1311           0.19           0.37
                524288          131072    fp32        1312           0.37           0.73
               1048576          262144    fp32        1328           0.74           1.45
               2097152          524288    fp32        1329           1.47           2.89
               4194304         1048576    fp32        1378           2.83           5.58
               8388608         2097152    fp32        1419           5.51          10.84
              16777216         4194304    fp32        2138           7.31          14.39
              33554432         8388608    fp32        2711          11.53          22.69
              67108864        16777216    fp32        3963          15.77          31.05
             134217728        33554432    fp32        6279          19.91          39.19
             268435456        67108864    fp32       11954          20.91          41.17
             536870912       134217728    fp32       21803          22.93          45.15
            1073741824       268435456    fp32       41806          23.92          47.09
        Avg bus bandwidth:      9.3924GB/s


.. _AlltoAllV Example:
- Specify alltoallv-metadata as JSON for ``alltoallv`` operation ``--alltoallv-metadata``.
.. code-block::

    NEURON_RT_ROOT_COMM_ID=172.32.137.79:44444 nccom-test -r 2 -N 2 -d fp32 alltoallv -b 1MB -e 1MB --hosts 127.0.0.1 172.32.253.16 --alltoallv-metadata alltoallv_metadata.json
    size(B)    count(elems)    type    time:avg(us)    algbw(GB/s)    busbw(GB/s)
    1048608          262152    fp32          955.05           1.10           0.55
    Avg bus bandwidth:	0.5490GB/s

    cat alltoallv_metadata.json
    {
      "alltoallv_metadata": [
        {
          "send_counts": [512, 1024],
          "send_displs": [0, 512],
          "recv_counts": [256, 768],
          "recv_displs": [0, 256]
        }
      ]
    }
