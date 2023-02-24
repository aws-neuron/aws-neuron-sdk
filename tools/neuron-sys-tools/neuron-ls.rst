.. _neuron-ls-ug:

Neuron LS User Guide
---------------------

To identify number of Neuron Devices in a given instance use the
``neuron-ls`` command. ``neuron-ls`` will also show which processes
are using each Device, including the command used to launch each of
those processes.

.. rubric:: neuron-ls CLI

.. program:: neuron-ls

.. option:: neuron-ls [options]

    **Available options:**

    - :option:`--wide, -w`: Displays the table in a wider format.

    - :option:`--show-all-procs, -a`: Show all processes using the Neuron Devices, including processes that aren't using
      Neuron Runtime 2.x such as ``neuron-monitor`` or ``neuron-ls`` itself.

    - :option:`--topology, -t`: Display topology information about the system's Neuron Devices.

.. note::

  ``neuron-ls`` fully supports the newly launched inf2 instances.


Examples
^^^^^^^^

First we will show the output of ``neuron-ls`` on an Inf1.6xlarge instance.

::

  $ neuron-ls
  +--------+--------+--------+-----------+--------------+-------+------------------------------------------+---------+
  | NEURON | NEURON | NEURON | CONNECTED |     PCI      |  PID  |                 COMMAND                  | RUNTIME |
  | DEVICE | CORES  | MEMORY |  DEVICES  |     BDF      |       |                                          | VERSION |
  +--------+--------+--------+-----------+--------------+-------+------------------------------------------+---------+
  | 0      | 4      | 8 GB   | 1         | 0000:00:1c.0 | 23518 | neuron-app01 infer --input-data-direc... | 2.0.0   |
  |        |        |        |           |              | 23531 | neuron-app02 infer --input-data-direc... | 2.0.0   |
  +--------+--------+--------+-----------+--------------+-------+------------------------------------------+---------+
  | 1      | 4      | 8 GB   | 2, 0      | 0000:00:1d.0 | 23595 | neuron-app01 infer --input-data-direc... | 2.0.0   |
  +--------+--------+--------+-----------+--------------+-------+------------------------------------------+---------+
  | 2      | 4      | 8 GB   | 3, 1      | 0000:00:1e.0 | 23608 | neuron-app02 infer --input-data-direc... | 2.0.0   |
  +--------+--------+--------+-----------+--------------+-------+------------------------------------------+---------+
  | 3      | 4      | 8 GB   | 2         | 0000:00:1f.0 | NA    | NA                                       | NA      |
  +--------+--------+--------+-----------+--------------+-------+------------------------------------------+---------+

  $ neuron-ls --wide
  +--------+--------+--------+-----------+--------------+-------+----------------------------------------------------------------------------------+---------+
  | NEURON | NEURON | NEURON | CONNECTED |     PCI      |  PID  |                                     COMMAND                                      | RUNTIME |
  | DEVICE | CORES  | MEMORY |  DEVICES  |     BDF      |       |                                                                                  | VERSION |
  +--------+--------+--------+-----------+--------------+-------+----------------------------------------------------------------------------------+---------+
  | 0      | 4      | 8 GB   | 1         | 0000:00:1c.0 | 23518 | neuron-app01 infer --input-data-directory ~/my_input_data --inference-count 5... | 2.0.0   |
  |        |        |        |           |              | 23531 | neuron-app02 infer --input-data-directory ~/my_input_data --inference-count 5... | 2.0.0   |
  +--------+--------+--------+-----------+--------------+-------+----------------------------------------------------------------------------------+---------+
  | 1      | 4      | 8 GB   | 2, 0      | 0000:00:1d.0 | 23595 | neuron-app01 infer --input-data-directory ~/my_input_data --inference-count 5... | 2.0.0   |
  +--------+--------+--------+-----------+--------------+-------+----------------------------------------------------------------------------------+---------+
  | 2      | 4      | 8 GB   | 3, 1      | 0000:00:1e.0 | 23608 | neuron-app02 infer --input-data-directory ~/my_input_data --inference-count 5... | 2.0.0   |
  +--------+--------+--------+-----------+--------------+-------+----------------------------------------------------------------------------------+---------+
  | 3      | 4      | 8 GB   | 2         | 0000:00:1f.0 | NA    | NA                                                                               | NA      |
  +--------+--------+--------+-----------+--------------+-------+----------------------------------------------------------------------------------+---------+

  $ neuron-ls --show-all-procs
  +--------+--------+--------+-----------+--------------+-------+------------------------------------------+---------+
  | NEURON | NEURON | NEURON | CONNECTED |     PCI      |  PID  |                 COMMAND                  | RUNTIME |
  | DEVICE | CORES  | MEMORY |  DEVICES  |     BDF      |       |                                          | VERSION |
  +--------+--------+--------+-----------+--------------+-------+------------------------------------------+---------+
  | 0      | 4      | 8 GB   | 1         | 0000:00:1c.0 | 23518 | neuron-app01 infer --input-data-direc... | 2.0.0   |
  |        |        |        |           |              | 23531 | neuron-app02 infer --input-data-direc... | 2.0.0   |
  |        |        |        |           |              | 23764 | neuron-monitor                           | NA      |
  |        |        |        |           |              | 23829 | neuron-ls --show-all-procs               | NA      |
  +--------+--------+--------+-----------+--------------+-------+------------------------------------------+---------+
  | 1      | 4      | 8 GB   | 2, 0      | 0000:00:1d.0 | 23595 | neuron-app01 infer --input-data-direc... | 2.0.0   |
  |        |        |        |           |              | 23764 | neuron-monitor                           | NA      |
  |        |        |        |           |              | 23829 | neuron-ls --show-all-procs               | NA      |
  +--------+--------+--------+-----------+--------------+-------+------------------------------------------+---------+
  | 2      | 4      | 8 GB   | 3, 1      | 0000:00:1e.0 | 23608 | neuron-app02 infer --input-data-direc... | 2.0.0   |
  |        |        |        |           |              | 23764 | neuron-monitor                           | NA      |
  |        |        |        |           |              | 23829 | neuron-ls --show-all-procs               | NA      |
  +--------+--------+--------+-----------+--------------+-------+------------------------------------------+---------+
  | 3      | 4      | 8 GB   | 2         | 0000:00:1f.0 | 23764 | neuron-monitor                           | NA      |
  |        |        |        |           |              | 23829 | neuron-ls --show-all-procs               | NA      |
  +--------+--------+--------+-----------+--------------+-------+------------------------------------------+---------+

    $ neuron-ls --topology
  +--------+--------+--------+-----------+---------+
  | NEURON | NEURON | NEURON | CONNECTED |   PCI   |
  | DEVICE | CORES  | MEMORY |  DEVICES  |   BDF   |
  +--------+--------+--------+-----------+---------+
  | 0      | 4      | 8 GB   | 1         | 00:1c.0 |
  | 1      | 4      | 8 GB   | 2, 0      | 00:1d.0 |
  | 2      | 4      | 8 GB   | 3, 1      | 00:1e.0 |
  | 3      | 4      | 8 GB   | 2         | 00:1f.0 |
  +--------+--------+--------+-----------+---------+

  Neuron Device Topology

    [ 0 ]◄––►[ 1 ]◄––►[ 2 ]◄––►[ 3 ]

On Trn1 and Inf2 instances ``neuron-ls`` works similarly. Below is an example displaying the topology for a Trn1.32xlarge instance.

::

 $ neuron-ls --topology
 +--------+--------+--------+---------------+---------+
 | NEURON | NEURON | NEURON |   CONNECTED   |   PCI   |
 | DEVICE | CORES  | MEMORY |    DEVICES    |   BDF   |
 +--------+--------+--------+---------------+---------+
 | 0      | 2      | 32 GB  | 12, 3, 4, 1   | 00:04.0 |
 | 1      | 2      | 32 GB  | 13, 0, 5, 2   | 00:05.0 |
 | 2      | 2      | 32 GB  | 14, 1, 6, 3   | 00:06.0 |
 | 3      | 2      | 32 GB  | 15, 2, 7, 0   | 00:07.0 |
 | 4      | 2      | 32 GB  | 0, 7, 8, 5    | 00:08.0 |
 | 5      | 2      | 32 GB  | 1, 4, 9, 6    | 00:09.0 |
 | 6      | 2      | 32 GB  | 2, 5, 10, 7   | 00:0a.0 |
 | 7      | 2      | 32 GB  | 3, 6, 11, 4   | 00:0b.0 |
 | 8      | 2      | 32 GB  | 4, 11, 12, 9  | 00:0c.0 |
 | 9      | 2      | 32 GB  | 5, 8, 13, 10  | 00:0d.0 |
 | 10     | 2      | 32 GB  | 6, 9, 14, 11  | 00:0e.0 |
 | 11     | 2      | 32 GB  | 7, 10, 15, 8  | 00:0f.0 |
 | 12     | 2      | 32 GB  | 8, 15, 0, 13  | 00:10.0 |
 | 13     | 2      | 32 GB  | 9, 12, 1, 14  | 00:11.0 |
 | 14     | 2      | 32 GB  | 10, 13, 2, 15 | 00:12.0 |
 | 15     | 2      | 32 GB  | 11, 14, 3, 12 | 00:13.0 |
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
 

-  NEURON DEVICE: Logical ID assigned to the Neuron Device.
-  NEURON CORES: Number of NeuronCores present in the Neuron Device.
-  NEURON MEMORY: Amount DRAM memory in Neuron Device.
-  CONNECTED DEVICES: Logical ID of Neuron Devices connected to this
   Neuron Device.
-  PCI BDF: PCI Bus Device Function (BDF) ID of the device.
-  PID: ID of the process using this NeuronDevice.
-  COMMAND: Command used to launch the process using this
   Neuron Device.
-  RUNTIME VERSION: Version of Neuron Runtime (if applicable) for
   the application using this Neuron Device.
