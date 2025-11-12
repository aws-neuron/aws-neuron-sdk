.. _nccom-test:

======================
NCCOM-TEST User Guide
======================

.. contents:: Table of contents
    :local:
    :depth: 2

Overview
--------

**nccom-test** is a benchmarking tool for evaluating Collective Communication operations on AWS Trainium and Inferentia instances. It supports Trn1, Trn2, and Inf2 instance types. The tool can assess performance across multiple instances or perform quick environment sanity checks before running more complex workloads. While single-instance benchmarking is supported for all compatible instance types, multi-instance benchmarking is limited to Trainium instances (Trn1 and Trn2). 

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
      - Size in bytes for the data involved in this operation
    * - count(elems)
      - Number of elements in the data involved in this operation. For example, if **size(B)** is 4 and **type** is fp32,
        then **count** will be 1 since one single fp32 element has been processed.
    * - type
      - Data type for the processed data. Can be: **uint8**, **int8**, **uint16**, **int16**, **fp16**, **bf16**, **int32**, **uint32**, **fp32**
    * - time(us)
      - Time in microseconds representing the P50 of all durations for the Collective Communication operations executed during the benchmark.
    * - algbw(GB/s)
      - Algorithm bandwidth in gibibytes (1GiB = 1,073,741,824 bytes) per second which is calculated as **size(B)** / **time(us)**
    * - busbw(GB/s)
      - Bus bandwidth - bandwidth per data line in gibibytes per second - it provides a bandwidth number that is independent from the number of ranks (unlike **algbw**).
        For a more in-depth explanation on bus Bandwidth, please refer to `NVIDIA's nccl-tests documentation <https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md>`_.
    * - Avg bus bandwidth
      - Average of the values in the busbw column

CLI arguments
^^^^^^^^^^^^^

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
    * - ``-r, --nworkers``
      - N/A, required argument
      - Total number of workers (ranks) to use
    * - ``-N, --nnodes``
      - 1
      - Total number of nodes (instances) to use. The number of workers will be divided equally across all nodes.
        If this argument is greater than 1, the **NEURON_RT_ROOT_COMM_ID** environment variable needs to be set to
        the host address of the instance **nccom-test** is ran on, and a free port number
        (for example: ``NEURON_RT_ROOT_COMM_ID=10.0.0.1:44444``). Additionally, either ``-s, --hosts`` needs to be provided
        or a ``~/hosts`` file needs to exist - for more details refer to the ``-s,--hosts`` description below.
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
    * - ``-n, --iters``
      - 20
      - Number of Collective Communication operations to execute during the benchmark.
    * - ``--custom-replica-group``
      - N/A
      - Provide the JSON file for custom-defined replica groups.
    * - ``--custom-src-target-pairs``
      - N/A
      - Provide the JSON file for custom-defined source_target_pairs for the collective permute operation.
    * - ``-w, --warmup_iters``
      - 5
      - Number of Collective Communication operations to execute as warmup during the benchmark
        (which won't be counted towards the result).
    * - ``-d, --datatype``
      - ``uint8``
      - Data type for the data used by the benchmark. Supported types: ``uint8``, ``int8``, ``uint16``, ``int16``,
        ``fp16``, ``bf16``, ``uint32``, ``int32``, ``fp32``. Input data will be zero filled, unless ``--check`` is
        provided (currently, only available for ``--datatype fp32``) in which case it will be filled by a repetead
        value of the requested type.
    * - ``-c, --check``
      - false
      - If provided, the corectness of the operations will be checked. This will not impact results (time, algbw and busbw)
        but will slightly increase the overall execution time.
    * - ``-s, --hosts``
      - N/A
      - Hosts on which to run execution.  Checks ``~/hosts`` if not specified.
    * - ``--non-interactive``
      - false
      - Do not display any animation or progress indicator.
    * - ``--report-to-json-file``
      - N/A
      - Persist config and results to JSON file if a filepath is provided.
    * - ``--show-input-output-size``
      - false
      - Print or save to JSON per rank input and output sizes in B.
    * - ``--sb2sb``
      - false
      - Indicates whether or not to allocate input, output, and ``scratch-buffer`` on SBUF (rather than HBM).  This may result in improved performance.

.. note::

    All arguments that take a size in bytes will also accept larger size units, for example:
    ``-f 2048`` can be written as ``-f 2kb`` or ``-f 1048576`` can be written as ``-f 1MB``.


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
