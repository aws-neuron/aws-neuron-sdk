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
        "connected_to": [
            12,
            3,
            4,
            1
        ],
        "nc_count": 4,
        "logical_neuroncore_config": 2,
        "memory_size": 103079215104,
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
        "connected_to": [
            11,
            14,
            3,
            12
        ],
        "nc_count": 4,
        "logical_neuroncore_config": 2,
        "memory_size": 103079215104,
        "neuron_processes": [
            {
                "pid": 113985,
                "command": "neuron-bench exec --run-as-cc-neff --...",
                "neuron_runtime_version": "2.0.0"
            }
        ]
    }
  ]
-  instance-type: Type of instance on which neuron-ls is running.
-  instance-id: EC2 ID of the instance on which neuron-ls is running.
-  logical-neuroncore-config: (only available on trn2 instances) the current logical NeuronCore configuration; for more information refer to :ref:`logical-neuroncore-config`
-  NEURON DEVICE / neuron_device: Logical ID assigned to the Neuron Device.
-  NEURON CORES / nc_count: Number of NeuronCores present in the Neuron Device.
-  NEURON MEMORY / memory_size: Amount DRAM memory in Neuron Device.
-  CONNECTED DEVICES / connected_to: Logical ID of Neuron Devices connected to this
   Neuron Device.
-  PCI BDF / bdf: PCI Bus Device Function (BDF) ID of the device.
-  PID / pid: ID of the process using this NeuronDevice.
-  COMMAND / command: Command used to launch the process using this
   Neuron Device.
-  RUNTIME VERSION / neuron_runtime_version: Version of Neuron Runtime (if applicable) for
   the application using this Neuron Device.
