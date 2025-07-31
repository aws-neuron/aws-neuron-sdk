.. _neuron-ls-ug:

Neuron LS User Guide
---------------------

The neuron-ls command is a tool for managing Neuron devices in your instance.
This command serves two key purposes: it identifies all Neuron devices present in the current instance 
and provides information about the processes running on each device along with the command that launched that process.
To use this command, simply type ``neuron-ls`` in your terminal.

.. rubric:: neuron-ls CLI

.. program:: neuron-ls

.. option:: neuron-ls [options]

    **Available options:**

    - :option:`--wide, -w`: Displays the table in a wider format.

    - :option:`--show-all-procs, -a`: Show all processes using the Neuron Devices, including processes that aren't using
      Neuron Runtime 2.x such as ``neuron-monitor`` or ``neuron-ls`` itself.

    - :option:`--topology, -t`: Display topology information about the system's Neuron Devices.

    - :option:`--json-output, -j`: Output in JSON format.

.. note::

  ``neuron-ls`` fully supports the newly launched Trn2 instances.

Examples
^^^^^^^^

``neuron-ls`` is compatible with all Neuron instance types: inf1, inf2, trn1 and trn2.
These are a few examples on running the tool on a trn2n.48xlarge:

::

  $ neuron-ls
  instance-type: trn2n.48xlarge
  instance-id: i-aabbccdd123456789
  logical-neuroncore-config: 2
  +--------+--------+----------+--------+---------------+--------------+---------------+------+
  | NEURON | NEURON |  NEURON  | NEURON |   CONNECTED   |     PCI      |      CPU      | NUMA |
  | DEVICE | CORES  | CORE IDS | MEMORY |    DEVICES    |     BDF      |   AFFINITY    | NODE |
  +--------+--------+----------+--------+---------------+--------------+---------------+------+
  | 0      | 4      | 0-3      | 96 GB  | 12, 3, 4, 1   | 0000:cc:00.0 | 48-95,144-191 | 1    |
  | 1      | 4      | 4-7      | 96 GB  | 13, 0, 5, 2   | 0000:b5:00.0 | 48-95,144-191 | 1    |
  | 2      | 4      | 8-11     | 96 GB  | 14, 1, 6, 3   | 0000:b6:00.0 | 48-95,144-191 | 1    |
  | 3      | 4      | 12-15    | 96 GB  | 15, 2, 7, 0   | 0000:cb:00.0 | 48-95,144-191 | 1    |
  | 4      | 4      | 16-19    | 96 GB  | 0, 7, 8, 5    | 0000:6f:00.0 | 0-47,96-143   | 0    |
  | 5      | 4      | 20-23    | 96 GB  | 1, 4, 9, 6    | 0000:58:00.0 | 0-47,96-143   | 0    |
  | 6      | 4      | 24-27    | 96 GB  | 2, 5, 10, 7   | 0000:59:00.0 | 0-47,96-143   | 0    |
  | 7      | 4      | 28-31    | 96 GB  | 3, 6, 11, 4   | 0000:6e:00.0 | 0-47,96-143   | 0    |
  | 8      | 4      | 32-35    | 96 GB  | 4, 11, 12, 9  | 0000:9b:00.0 | 0-47,96-143   | 0    |
  | 9      | 4      | 36-39    | 96 GB  | 5, 8, 13, 10  | 0000:84:00.0 | 0-47,96-143   | 0    |
  | 10     | 4      | 40-43    | 96 GB  | 6, 9, 14, 11  | 0000:85:00.0 | 0-47,96-143   | 0    |
  | 11     | 4      | 44-47    | 96 GB  | 7, 10, 15, 8  | 0000:9a:00.0 | 0-47,96-143   | 0    |
  | 12     | 4      | 48-51    | 96 GB  | 8, 15, 0, 13  | 0000:f8:00.0 | 48-95,144-191 | 1    |
  | 13     | 4      | 52-55    | 96 GB  | 9, 12, 1, 14  | 0000:e1:00.0 | 48-95,144-191 | 1    |
  | 14     | 4      | 56-59    | 96 GB  | 10, 13, 2, 15 | 0000:e2:00.0 | 48-95,144-191 | 1    |
  | 15     | 4      | 60-63    | 96 GB  | 11, 14, 3, 12 | 0000:f7:00.0 | 48-95,144-191 | 1    |
  +--------+--------+----------+--------+---------------+--------------+---------------+------+

::

  $ neuron-ls --wide
  instance-type: trn2n.48xlarge
  instance-id: i-aabbccdd123456789
  logical-neuroncore-config: 2
  +--------+--------+--------+---------------+---------+--------+----------------------------------------------------------------------------------+---------+
  | NEURON | NEURON | NEURON |   CONNECTED   |   PCI   |  PID   |                                     COMMAND                                      | RUNTIME |
  | DEVICE | CORES  | MEMORY |    DEVICES    |   BDF   |        |                                                                                  | VERSION |
  +--------+--------+--------+---------------+---------+--------+----------------------------------------------------------------------------------+---------+
  | 0      | 4      | 96 GB  | 12, 3, 4, 1   | cc:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --warmup none --fixed-instance-count 64 --... | 2.0.0   |
  | 1      | 4      | 96 GB  | 13, 0, 5, 2   | b5:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --warmup none --fixed-instance-count 64 --... | 2.0.0   |
  | 2      | 4      | 96 GB  | 14, 1, 6, 3   | b6:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --warmup none --fixed-instance-count 64 --... | 2.0.0   |
  | 3      | 4      | 96 GB  | 15, 2, 7, 0   | cb:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --warmup none --fixed-instance-count 64 --... | 2.0.0   |
  | 4      | 4      | 96 GB  | 0, 7, 8, 5    | 6f:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --warmup none --fixed-instance-count 64 --... | 2.0.0   |
  | 5      | 4      | 96 GB  | 1, 4, 9, 6    | 58:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --warmup none --fixed-instance-count 64 --... | 2.0.0   |
  | 6      | 4      | 96 GB  | 2, 5, 10, 7   | 59:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --warmup none --fixed-instance-count 64 --... | 2.0.0   |
  | 7      | 4      | 96 GB  | 3, 6, 11, 4   | 6e:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --warmup none --fixed-instance-count 64 --... | 2.0.0   |
  | 8      | 4      | 96 GB  | 4, 11, 12, 9  | 9b:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --warmup none --fixed-instance-count 64 --... | 2.0.0   |
  | 9      | 4      | 96 GB  | 5, 8, 13, 10  | 84:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --warmup none --fixed-instance-count 64 --... | 2.0.0   |
  | 10     | 4      | 96 GB  | 6, 9, 14, 11  | 85:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --warmup none --fixed-instance-count 64 --... | 2.0.0   |
  | 11     | 4      | 96 GB  | 7, 10, 15, 8  | 9a:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --warmup none --fixed-instance-count 64 --... | 2.0.0   |
  | 12     | 4      | 96 GB  | 8, 15, 0, 13  | f8:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --warmup none --fixed-instance-count 64 --... | 2.0.0   |
  | 13     | 4      | 96 GB  | 9, 12, 1, 14  | e1:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --warmup none --fixed-instance-count 64 --... | 2.0.0   |
  | 14     | 4      | 96 GB  | 10, 13, 2, 15 | e2:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --warmup none --fixed-instance-count 64 --... | 2.0.0   |
  | 15     | 4      | 96 GB  | 11, 14, 3, 12 | f7:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --warmup none --fixed-instance-count 64 --... | 2.0.0   |
  +--------+--------+--------+---------------+---------+--------+----------------------------------------------------------------------------------+---------+

::

  $ neuron-ls --show-all-procs
  instance-type: trn2n.48xlarge
  instance-id: i-aabbccdd123456789
  logical-neuroncore-config: 2
  +--------+--------+--------+---------------+---------+--------+------------------------------------------+---------+
  | NEURON | NEURON | NEURON |   CONNECTED   |   PCI   |  PID   |                 COMMAND                  | RUNTIME |
  | DEVICE | CORES  | MEMORY |    DEVICES    |   BDF   |        |                                          | VERSION |
  +--------+--------+--------+---------------+---------+--------+------------------------------------------+---------+
  | 0      | 4      | 96 GB  | 12, 3, 4, 1   | cc:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --... | 2.0.0   |
  |        |        |        |               |         | 269192 | neuron-ls --show-all-procs               | NA      |
  +--------+--------+--------+---------------+---------+--------+------------------------------------------+---------+
  | 1      | 4      | 96 GB  | 13, 0, 5, 2   | b5:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --... | 2.0.0   |
  |        |        |        |               |         | 269192 | neuron-ls --show-all-procs               | NA      |
  +--------+--------+--------+---------------+---------+--------+------------------------------------------+---------+
  | 2      | 4      | 96 GB  | 14, 1, 6, 3   | b6:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --... | 2.0.0   |
  |        |        |        |               |         | 269192 | neuron-ls --show-all-procs               | NA      |
  +--------+--------+--------+---------------+---------+--------+------------------------------------------+---------+
  | 3      | 4      | 96 GB  | 15, 2, 7, 0   | cb:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --... | 2.0.0   |
  |        |        |        |               |         | 269192 | neuron-ls --show-all-procs               | NA      |
  +--------+--------+--------+---------------+---------+--------+------------------------------------------+---------+
  | 4      | 4      | 96 GB  | 0, 7, 8, 5    | 6f:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --... | 2.0.0   |
  |        |        |        |               |         | 269192 | neuron-ls --show-all-procs               | NA      |
  +--------+--------+--------+---------------+---------+--------+------------------------------------------+---------+
  | 5      | 4      | 96 GB  | 1, 4, 9, 6    | 58:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --... | 2.0.0   |
  |        |        |        |               |         | 269192 | neuron-ls --show-all-procs               | NA      |
  +--------+--------+--------+---------------+---------+--------+------------------------------------------+---------+
  | 6      | 4      | 96 GB  | 2, 5, 10, 7   | 59:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --... | 2.0.0   |
  |        |        |        |               |         | 269192 | neuron-ls --show-all-procs               | NA      |
  +--------+--------+--------+---------------+---------+--------+------------------------------------------+---------+
  | 7      | 4      | 96 GB  | 3, 6, 11, 4   | 6e:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --... | 2.0.0   |
  |        |        |        |               |         | 269192 | neuron-ls --show-all-procs               | NA      |
  +--------+--------+--------+---------------+---------+--------+------------------------------------------+---------+
  | 8      | 4      | 96 GB  | 4, 11, 12, 9  | 9b:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --... | 2.0.0   |
  |        |        |        |               |         | 269192 | neuron-ls --show-all-procs               | NA      |
  +--------+--------+--------+---------------+---------+--------+------------------------------------------+---------+
  | 9      | 4      | 96 GB  | 5, 8, 13, 10  | 84:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --... | 2.0.0   |
  |        |        |        |               |         | 269192 | neuron-ls --show-all-procs               | NA      |
  +--------+--------+--------+---------------+---------+--------+------------------------------------------+---------+
  | 10     | 4      | 96 GB  | 6, 9, 14, 11  | 85:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --... | 2.0.0   |
  |        |        |        |               |         | 269192 | neuron-ls --show-all-procs               | NA      |
  +--------+--------+--------+---------------+---------+--------+------------------------------------------+---------+
  | 11     | 4      | 96 GB  | 7, 10, 15, 8  | 9a:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --... | 2.0.0   |
  |        |        |        |               |         | 269192 | neuron-ls --show-all-procs               | NA      |
  +--------+--------+--------+---------------+---------+--------+------------------------------------------+---------+
  | 12     | 4      | 96 GB  | 8, 15, 0, 13  | f8:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --... | 2.0.0   |
  |        |        |        |               |         | 269192 | neuron-ls --show-all-procs               | NA      |
  +--------+--------+--------+---------------+---------+--------+------------------------------------------+---------+
  | 13     | 4      | 96 GB  | 9, 12, 1, 14  | e1:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --... | 2.0.0   |
  |        |        |        |               |         | 269192 | neuron-ls --show-all-procs               | NA      |
  +--------+--------+--------+---------------+---------+--------+------------------------------------------+---------+
  | 14     | 4      | 96 GB  | 10, 13, 2, 15 | e2:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --... | 2.0.0   |
  |        |        |        |               |         | 269192 | neuron-ls --show-all-procs               | NA      |
  +--------+--------+--------+---------------+---------+--------+------------------------------------------+---------+
  | 15     | 4      | 96 GB  | 11, 14, 3, 12 | f7:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --... | 2.0.0   |
  |        |        |        |               |         | 269192 | neuron-ls --show-all-procs               | NA      |
  +--------+--------+--------+---------------+---------+--------+------------------------------------------+---------+

::

  $ neuron-ls --topology
  instance-type: trn2n.48xlarge
  instance-id: i-aabbccdd123456789
  logical-neuroncore-config: 2
  +--------+--------+--------+---------------+---------+
  | NEURON | NEURON | NEURON |   CONNECTED   |   PCI   |
  | DEVICE | CORES  | MEMORY |    DEVICES    |   BDF   |
  +--------+--------+--------+---------------+---------+
  | 0      | 4      | 96 GB  | 12, 3, 4, 1   | cc:00.0 |
  | 1      | 4      | 96 GB  | 13, 0, 5, 2   | b5:00.0 |
  | 2      | 4      | 96 GB  | 14, 1, 6, 3   | b6:00.0 |
  | 3      | 4      | 96 GB  | 15, 2, 7, 0   | cb:00.0 |
  | 4      | 4      | 96 GB  | 0, 7, 8, 5    | 6f:00.0 |
  | 5      | 4      | 96 GB  | 1, 4, 9, 6    | 58:00.0 |
  | 6      | 4      | 96 GB  | 2, 5, 10, 7   | 59:00.0 |
  | 7      | 4      | 96 GB  | 3, 6, 11, 4   | 6e:00.0 |
  | 8      | 4      | 96 GB  | 4, 11, 12, 9  | 9b:00.0 |
  | 9      | 4      | 96 GB  | 5, 8, 13, 10  | 84:00.0 |
  | 10     | 4      | 96 GB  | 6, 9, 14, 11  | 85:00.0 |
  | 11     | 4      | 96 GB  | 7, 10, 15, 8  | 9a:00.0 |
  | 12     | 4      | 96 GB  | 8, 15, 0, 13  | f8:00.0 |
  | 13     | 4      | 96 GB  | 9, 12, 1, 14  | e1:00.0 |
  | 14     | 4      | 96 GB  | 10, 13, 2, 15 | e2:00.0 |
  | 15     | 4      | 96 GB  | 11, 14, 3, 12 | f7:00.0 |
  +--------+--------+--------+---------------+---------+


  Neuron Device Topology
        *        *        *        *      
        │        │        │        │      
        ▼        ▼        ▼        ▼      
  *––►[ 0 ]◄––►[ 1 ]◄––►[ 2 ]◄––►[ 3 ]◄––*
        ▲        ▲        ▲        ▲      
        │        │        │        │      
        ▼        ▼        ▼        ▼      
  *––►[ 4 ]◄––►[ 5 ]◄––►[ 6 ]◄––►[ 7 ]◄––*
        ▲        ▲        ▲        ▲      
        │        │        │        │      
        ▼        ▼        ▼        ▼      
  *––►[ 8 ]◄––►[ 9 ]◄––►[10 ]◄––►[11 ]◄––*
        ▲        ▲        ▲        ▲      
        │        │        │        │      
        ▼        ▼        ▼        ▼      
  *––►[12 ]◄––►[13 ]◄––►[14 ]◄––►[15 ]◄––*
        ▲        ▲        ▲        ▲      
        │        │        │        │      
        *        *        *        *      

  Legend:

          *––► = Wrap-around link
::

  $ neuron-ls -j
  [
    {
        "neuron_device": 0,
        "bdf": "cc:00.0",
        "cpu_affinity": "48-95,144-191",
        "numa_node": "1",
        "connected_to": [
            12,
            3,
            4,
            1
        ],
        "nc_count": 4,
        "logical_neuroncore_config": 2,
        "memory_size": 103079215104,
        "neuroncore_ids": [
            0,
            1,
            2,
            3
        ],
        "neuron_processes": [
            {
                "pid": 113985,
                "command": "neuron-bench exec --run-as-cc-neff --...",
                "neuron_runtime_version": "2.0.0"
            }
        ]
    },
    ...
    {
        "neuron_device": 15,
        "bdf": "f7:00.0",
        "cpu_affinity": "48-95,144-191",
        "numa_node": "1",
        "connected_to": [
            11,
            14,
            3,
            12
        ],
        "nc_count": 4,
        "logical_neuroncore_config": 2,
        "memory_size": 103079215104,
        "neuroncore_ids": [
            60,
            61,
            62,
            63
        ],
        "neuron_processes": [
            {
                "pid": 113985,
                "command": "neuron-bench exec --run-as-cc-neff --...",
                "neuron_runtime_version": "2.0.0"
            }
        ]
    }
  ]

Field Definitions
^^^^^^^^^^^^^^^^^

-  instance-type: Type of instance on which neuron-ls is running.
-  instance-id: EC2 ID of the instance on which neuron-ls is running.
-  logical-neuroncore-config: (only available on trn2 instances) the current logical NeuronCore configuration; for more information refer to :ref:`logical-neuroncore-config`
-  NEURON DEVICE / neuron_device: Logical ID assigned to the Neuron Device.
-  NEURON CORES / nc_count: Number of NeuronCores present in the Neuron Device.
-  NEURON CORE IDS / neuroncore_ids: Range or list of individual NeuronCore IDs belonging to the device, used with ``NEURON_RT_VISIBLE_CORES`` for selective core usage.
-  NEURON MEMORY / memory_size: Amount DRAM memory in Neuron Device.
-  CONNECTED DEVICES / connected_to: Logical ID of Neuron Devices connected to this
   Neuron Device.
-  PCI BDF / bdf: PCI Bus Device Function (BDF) ID of the device.
-  CPU AFFINITY / cpu_affinity: CPU cores that per NeuronCore proxy threads are pinned to
-  NUMA NODE / numa_node: NUMA (Non-Uniform Memory Access) node associated with the Neuron Device
-  PID / pid: ID of the process using this Neuron Device.
-  COMMAND / command: Command used to launch the process using this
   Neuron Device.
-  RUNTIME VERSION / neuron_runtime_version: Version of Neuron Runtime (if applicable) for
   the application using this Neuron Device.
