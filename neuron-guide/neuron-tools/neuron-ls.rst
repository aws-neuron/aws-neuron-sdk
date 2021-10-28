.. _neuron-ls-ug:

Neuron LS User Guide
---------------------

To identify number of Neuron Devices in a given instance use the
``neuron-ls`` command. ``neuron-ls`` will also show which processes
are using each Device, including the command used to launch each of
those processes.

If you are running Neuron Runtime 1.x neuron-rtd daemons, run the
command as ``sudo`` to get their addresses and version numbers.

.. rubric:: neuron-ls CLI

.. program:: neuron-ls

.. option:: neuron-ls [options]

    **Available options:**

    - :option:`--wide, -w`: Displays the table in a wider format.

    - :option:`--show-all-procs, -a`: Show all processes using the Neuron Devices, including processes that aren't using
      Neuron Runtime 2.x such as ``neuron-monitor`` or ``neuron-ls`` itself.

.. note::

  If you're running Neuron Runtime 1.x ``neuron-rtd`` daemons, use ``sudo`` with the
  ``neuron-ls`` command to obtain their address and version information. Without ``sudo``,
  the ``neuron-rtd`` processes will be considered generic processes using one or more Devices
  and will only be displayed when running ``neuron-ls`` with the ``--show-all-procs`` option.

Examples
^^^^^^^^

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


The above output is taken from an Inf1.6xlarge instance.

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
