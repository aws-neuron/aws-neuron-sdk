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

.. note::

  ``neuron-ls`` fully supports the newly launched Trn2 instances.

Examples
^^^^^^^^

On Inf1, Inf2, Trn1 and Trn2 instances ``neuron-ls`` works similarly.
Following is an example output of ``neuron-ls`` on an Trn2n.48xlarge instance.

::

  $ neuron-ls
  +--------+--------+---------+--------+---------------+---------+
  | NEURON | NEURON | LOGICAL | NEURON |   CONNECTED   |   PCI   |
  | DEVICE | CORES  |   NC    | MEMORY |    DEVICES    |   BDF   |
  +--------+--------+---------+--------+---------------+---------+
  | 0      | 8      | 4       | 96 GB  | 12, 3, 4, 1   | cc:00.0 |
  | 1      | 8      | 4       | 96 GB  | 13, 0, 5, 2   | b5:00.0 |
  | 2      | 8      | 4       | 96 GB  | 14, 1, 6, 3   | b6:00.0 |
  | 3      | 8      | 4       | 96 GB  | 15, 2, 7, 0   | cb:00.0 |
  | 4      | 8      | 4       | 96 GB  | 0, 7, 8, 5    | 6f:00.0 |
  | 5      | 8      | 4       | 96 GB  | 1, 4, 9, 6    | 58:00.0 |
  | 6      | 8      | 4       | 96 GB  | 2, 5, 10, 7   | 59:00.0 |
  | 7      | 8      | 4       | 96 GB  | 3, 6, 11, 4   | 6e:00.0 |
  | 8      | 8      | 4       | 96 GB  | 4, 11, 12, 9  | 9b:00.0 |
  | 9      | 8      | 4       | 96 GB  | 5, 8, 13, 10  | 84:00.0 |
  | 10     | 8      | 4       | 96 GB  | 6, 9, 14, 11  | 85:00.0 |
  | 11     | 8      | 4       | 96 GB  | 7, 10, 15, 8  | 9a:00.0 |
  | 12     | 8      | 4       | 96 GB  | 8, 15, 0, 13  | f8:00.0 |
  | 13     | 8      | 4       | 96 GB  | 9, 12, 1, 14  | e1:00.0 |
  | 14     | 8      | 4       | 96 GB  | 10, 13, 2, 15 | e2:00.0 |
  | 15     | 8      | 4       | 96 GB  | 11, 14, 3, 12 | f7:00.0 |
  +--------+--------+---------+--------+---------------+---------+

::
    
  $ neuron-ls --wide
  +--------+--------+---------+--------+---------------+---------+--------+----------------------------------------------------------------------------------+---------+
  | NEURON | NEURON | LOGICAL | NEURON |   CONNECTED   |   PCI   |  PID   |                                     COMMAND                                      | RUNTIME |
  | DEVICE | CORES  |   NC    | MEMORY |    DEVICES    |   BDF   |        |                                                                                  | VERSION |
  +--------+--------+---------+--------+---------------+---------+--------+----------------------------------------------------------------------------------+---------+
  | 0      | 8      | 4       | 96 GB  | 12, 3, 4, 1   | cc:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --warmup none --fixed-instance-count 64 --... | 2.0.0   |
  | 1      | 8      | 4       | 96 GB  | 13, 0, 5, 2   | b5:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --warmup none --fixed-instance-count 64 --... | 2.0.0   |
  | 2      | 8      | 4       | 96 GB  | 14, 1, 6, 3   | b6:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --warmup none --fixed-instance-count 64 --... | 2.0.0   |
  | 3      | 8      | 4       | 96 GB  | 15, 2, 7, 0   | cb:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --warmup none --fixed-instance-count 64 --... | 2.0.0   |
  | 4      | 8      | 4       | 96 GB  | 0, 7, 8, 5    | 6f:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --warmup none --fixed-instance-count 64 --... | 2.0.0   |
  | 5      | 8      | 4       | 96 GB  | 1, 4, 9, 6    | 58:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --warmup none --fixed-instance-count 64 --... | 2.0.0   |
  | 6      | 8      | 4       | 96 GB  | 2, 5, 10, 7   | 59:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --warmup none --fixed-instance-count 64 --... | 2.0.0   |
  | 7      | 8      | 4       | 96 GB  | 3, 6, 11, 4   | 6e:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --warmup none --fixed-instance-count 64 --... | 2.0.0   |
  | 8      | 8      | 4       | 96 GB  | 4, 11, 12, 9  | 9b:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --warmup none --fixed-instance-count 64 --... | 2.0.0   |
  | 9      | 8      | 4       | 96 GB  | 5, 8, 13, 10  | 84:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --warmup none --fixed-instance-count 64 --... | 2.0.0   |
  | 10     | 8      | 4       | 96 GB  | 6, 9, 14, 11  | 85:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --warmup none --fixed-instance-count 64 --... | 2.0.0   |
  | 11     | 8      | 4       | 96 GB  | 7, 10, 15, 8  | 9a:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --warmup none --fixed-instance-count 64 --... | 2.0.0   |
  | 12     | 8      | 4       | 96 GB  | 8, 15, 0, 13  | f8:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --warmup none --fixed-instance-count 64 --... | 2.0.0   |
  | 13     | 8      | 4       | 96 GB  | 9, 12, 1, 14  | e1:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --warmup none --fixed-instance-count 64 --... | 2.0.0   |
  | 14     | 8      | 4       | 96 GB  | 10, 13, 2, 15 | e2:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --warmup none --fixed-instance-count 64 --... | 2.0.0   |
  | 15     | 8      | 4       | 96 GB  | 11, 14, 3, 12 | f7:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --warmup none --fixed-instance-count 64 --... | 2.0.0   |
  +--------+--------+---------+--------+---------------+---------+--------+----------------------------------------------------------------------------------+---------+  

::

  $ neuron-ls --show-all-procs
  +--------+--------+---------+--------+---------------+---------+--------+------------------------------------------+---------+
  | NEURON | NEURON | LOGICAL | NEURON |   CONNECTED   |   PCI   |  PID   |                 COMMAND                  | RUNTIME |
  | DEVICE | CORES  |   NC    | MEMORY |    DEVICES    |   BDF   |        |                                          | VERSION |
  +--------+--------+---------+--------+---------------+---------+--------+------------------------------------------+---------+
  | 0      | 8      | 4       | 96 GB  | 12, 3, 4, 1   | cc:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --... | 2.0.0   |
  |        |        |         |        |               |         | 269192 | neuron-ls --show-all-procs               | NA      |
  +--------+--------+---------+--------+---------------+---------+--------+------------------------------------------+---------+
  | 1      | 8      | 4       | 96 GB  | 13, 0, 5, 2   | b5:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --... | 2.0.0   |
  |        |        |         |        |               |         | 269192 | neuron-ls --show-all-procs               | NA      |
  +--------+--------+---------+--------+---------------+---------+--------+------------------------------------------+---------+
  | 2      | 8      | 4       | 96 GB  | 14, 1, 6, 3   | b6:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --... | 2.0.0   |
  |        |        |         |        |               |         | 269192 | neuron-ls --show-all-procs               | NA      |
  +--------+--------+---------+--------+---------------+---------+--------+------------------------------------------+---------+
  | 3      | 8      | 4       | 96 GB  | 15, 2, 7, 0   | cb:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --... | 2.0.0   |
  |        |        |         |        |               |         | 269192 | neuron-ls --show-all-procs               | NA      |
  +--------+--------+---------+--------+---------------+---------+--------+------------------------------------------+---------+
  | 4      | 8      | 4       | 96 GB  | 0, 7, 8, 5    | 6f:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --... | 2.0.0   |
  |        |        |         |        |               |         | 269192 | neuron-ls --show-all-procs               | NA      |
  +--------+--------+---------+--------+---------------+---------+--------+------------------------------------------+---------+
  | 5      | 8      | 4       | 96 GB  | 1, 4, 9, 6    | 58:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --... | 2.0.0   |
  |        |        |         |        |               |         | 269192 | neuron-ls --show-all-procs               | NA      |
  +--------+--------+---------+--------+---------------+---------+--------+------------------------------------------+---------+
  | 6      | 8      | 4       | 96 GB  | 2, 5, 10, 7   | 59:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --... | 2.0.0   |
  |        |        |         |        |               |         | 269192 | neuron-ls --show-all-procs               | NA      |
  +--------+--------+---------+--------+---------------+---------+--------+------------------------------------------+---------+
  | 7      | 8      | 4       | 96 GB  | 3, 6, 11, 4   | 6e:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --... | 2.0.0   |
  |        |        |         |        |               |         | 269192 | neuron-ls --show-all-procs               | NA      |
  +--------+--------+---------+--------+---------------+---------+--------+------------------------------------------+---------+
  | 8      | 8      | 4       | 96 GB  | 4, 11, 12, 9  | 9b:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --... | 2.0.0   |
  |        |        |         |        |               |         | 269192 | neuron-ls --show-all-procs               | NA      |
  +--------+--------+---------+--------+---------------+---------+--------+------------------------------------------+---------+
  | 9      | 8      | 4       | 96 GB  | 5, 8, 13, 10  | 84:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --... | 2.0.0   |
  |        |        |         |        |               |         | 269192 | neuron-ls --show-all-procs               | NA      |
  +--------+--------+---------+--------+---------------+---------+--------+------------------------------------------+---------+
  | 10     | 8      | 4       | 96 GB  | 6, 9, 14, 11  | 85:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --... | 2.0.0   |
  |        |        |         |        |               |         | 269192 | neuron-ls --show-all-procs               | NA      |
  +--------+--------+---------+--------+---------------+---------+--------+------------------------------------------+---------+
  | 11     | 8      | 4       | 96 GB  | 7, 10, 15, 8  | 9a:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --... | 2.0.0   |
  |        |        |         |        |               |         | 269192 | neuron-ls --show-all-procs               | NA      |
  +--------+--------+---------+--------+---------------+---------+--------+------------------------------------------+---------+
  | 12     | 8      | 4       | 96 GB  | 8, 15, 0, 13  | f8:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --... | 2.0.0   |
  |        |        |         |        |               |         | 269192 | neuron-ls --show-all-procs               | NA      |
  +--------+--------+---------+--------+---------------+---------+--------+------------------------------------------+---------+
  | 13     | 8      | 4       | 96 GB  | 9, 12, 1, 14  | e1:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --... | 2.0.0   |
  |        |        |         |        |               |         | 269192 | neuron-ls --show-all-procs               | NA      |
  +--------+--------+---------+--------+---------------+---------+--------+------------------------------------------+---------+
  | 14     | 8      | 4       | 96 GB  | 10, 13, 2, 15 | e2:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --... | 2.0.0   |
  |        |        |         |        |               |         | 269192 | neuron-ls --show-all-procs               | NA      |
  +--------+--------+---------+--------+---------------+---------+--------+------------------------------------------+---------+
  | 15     | 8      | 4       | 96 GB  | 11, 14, 3, 12 | f7:00.0 | 268911 | neuron-bench exec --run-as-cc-neff --... | 2.0.0   |
  |        |        |         |        |               |         | 269192 | neuron-ls --show-all-procs               | NA      |
  +--------+--------+---------+--------+---------------+---------+--------+------------------------------------------+---------+

::

  $ neuron-ls --topology
  +--------+--------+---------+--------+---------------+---------+
  | NEURON | NEURON | LOGICAL | NEURON |   CONNECTED   |   PCI   |
  | DEVICE | CORES  |   NC    | MEMORY |    DEVICES    |   BDF   |
  +--------+--------+---------+--------+---------------+---------+
  | 0      | 8      | 4       | 96 GB  | 12, 3, 4, 1   | cc:00.0 |
  | 1      | 8      | 4       | 96 GB  | 13, 0, 5, 2   | b5:00.0 |
  | 2      | 8      | 4       | 96 GB  | 14, 1, 6, 3   | b6:00.0 |
  | 3      | 8      | 4       | 96 GB  | 15, 2, 7, 0   | cb:00.0 |
  | 4      | 8      | 4       | 96 GB  | 0, 7, 8, 5    | 6f:00.0 |
  | 5      | 8      | 4       | 96 GB  | 1, 4, 9, 6    | 58:00.0 |
  | 6      | 8      | 4       | 96 GB  | 2, 5, 10, 7   | 59:00.0 |
  | 7      | 8      | 4       | 96 GB  | 3, 6, 11, 4   | 6e:00.0 |
  | 8      | 8      | 4       | 96 GB  | 4, 11, 12, 9  | 9b:00.0 |
  | 9      | 8      | 4       | 96 GB  | 5, 8, 13, 10  | 84:00.0 |
  | 10     | 8      | 4       | 96 GB  | 6, 9, 14, 11  | 85:00.0 |
  | 11     | 8      | 4       | 96 GB  | 7, 10, 15, 8  | 9a:00.0 |
  | 12     | 8      | 4       | 96 GB  | 8, 15, 0, 13  | f8:00.0 |
  | 13     | 8      | 4       | 96 GB  | 9, 12, 1, 14  | e1:00.0 |
  | 14     | 8      | 4       | 96 GB  | 10, 13, 2, 15 | e2:00.0 |
  | 15     | 8      | 4       | 96 GB  | 11, 14, 3, 12 | f7:00.0 |
  +--------+--------+---------+--------+---------------+---------+
  
  
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

-  NEURON DEVICE: Logical ID assigned to the Neuron Device.
-  NEURON CORES: Number of NeuronCores present in the Neuron Device.
-  LOGICAL NC: (Only for Trn2) Number of Logical NeuronCores present in Neuron Devices.
-  NEURON MEMORY: Amount DRAM memory in Neuron Device.
-  CONNECTED DEVICES: Logical ID of Neuron Devices connected to this
   Neuron Device.
-  PCI BDF: PCI Bus Device Function (BDF) ID of the device.
-  PID: ID of the process using this NeuronDevice.
-  COMMAND: Command used to launch the process using this
   Neuron Device.
-  RUNTIME VERSION: Version of Neuron Runtime (if applicable) for
   the application using this Neuron Device.
