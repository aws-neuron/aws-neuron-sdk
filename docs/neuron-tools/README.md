# Neuron Tools

# Identifying Neuron Devices

To identify number of Neuron Devices in a given instance use the `neuron-ls` command.

```
$ neuron-ls
+--------------+---------+--------+--------+--------------+-----------------------+---------+---------+
|   PCI BDF    | LOGICAL | NEURON | MEMORY | CONNECTED TO |        RUNTIME        | RUNTIME | RUNTIME |
|              |   ID    | CORES  |        |              |        ADDRESS        |   PID   | VERSION |
+--------------+---------+--------+--------+--------------+-----------------------+---------+---------+
| 0000:00:1c.0 |       0 |      4 | 8 GB   | [1]          | unix:/run/neuron.sock |   17902 | 1.0.x.x |
| 0000:00:1d.0 |       1 |      4 | 8 GB   | [0 2]        | unix:/run/neuron.sock |   17902 | 1.0.x.x |
| 0000:00:1e.0 |       2 |      4 | 8 GB   | [1 3]        | unix:/run/neuron.sock |   17902 | 1.0.x.x |
| 0000:00:1f.0 |       3 |      4 | 8 GB   | [2]          | unix:/run/neuron.sock |   17902 | 1.0.x.x |
+--------------+---------+--------+--------+--------------+-----------------------+---------+---------+
```

The above output is taken from an Inf1.6xlarge instance.
- PCI BDF           ->  PCI Bus Device Function (BDF) ID of the device.
- LOGICAL ID        ->  Logical ID assigned to the device. This id can be used when configuring multiple runtime to use different devices.
- NEURON CORES      ->  Number of NeuronCores present in the NeuronDevice.
- CONNECTED TO      ->  Shows other NeuronDevices connected to this NeuronDevice. (Only connected devices can be used in a neuron core group(NCG) configuration.)
- RUNTIME ADDRESS   ->  Shows address of runtime process using this NeuronDevice.
- RUNTIME PID       ->  Shows process id of runtime process using this NeuronDevice.
- RUNTIME VERSION   ->  Shows version of runtime process using this NeuronDevice.

# NeuronCore Groups
Multiple NeuronCores(NC) can be combined to form a NeuronCore Group (NCG).
Neuron framework layer will automatically create a default NeuronCore Group.
To view list of available NCGs the following command can be used.
```
$ neuron-cli list-ncg
Device count 4 NC count 16
Found 4 NCG's
+--------+----------+--------------------+----------------+
| NCG ID | NC COUNT | DEVICE START INDEX | NC START INDEX |
+--------+----------+--------------------+----------------+
|      1 |        2 |                  0 |              0 |
|      2 |        4 |                  0 |              2 |
|      3 |        3 |                  1 |              2 |
|      4 |        1 |                  2 |              1 |
+--------+----------+--------------------+----------------+
```
The above examples shows there are 4 NCG created on the system with the following grouping
NCG ID 1: Device0:(Core0, Core1)
NCG ID 2: Device0:(Core2, Core3), Device1:(Core0, Core1)
NCG ID 3: Device1:(Core2, Core3), Device2:(Core0)
NCG ID 2: Device1:(Core1)

# Listing Models
Multiple models can be loaded into a single NCG but only one can be in READY state at any given moment.
Inference can be performed only on the models in the READY state.

The `neuron-cli list-model` command should be used to view all the models.
```
$ neuron-cli list-model
+----------------------------------------------+----------+--------------+---------------+-------+-------+----------------------+
|                     UUID                     | MODEL ID | MODEL STATUS | NEURON DEVICE |  NC   |  NC   |         NAME         |
|                                              |          |              |  START INDEX  | INDEX | COUNT |                      |
+----------------------------------------------+----------+--------------+---------------+-------+-------+----------------------+
| 63c43dd60b0411eaa9160288cac7f65c30           |    10011 | STANDBY      |             1 |     0 |     1 | test0_1_concat_multi |
| 63c43dd60b0411eaa9160288cac7f65c30330808637f |    10010 | STANDBY      |             0 |     0 |     1 | test0_1_concat_multi |
| 63c43dd60b0411eaa9160288cac7f65ce078         |    10009 | READY        |             1 |     1 |     1 | test0_1_concat_multi |
| 63c43dd60b0411eaa9160288cac7f65ca05f         |    10008 | READY        |             1 |     0 |     1 | test0_1_concat_multi |
| 6a9726                                       |    10007 | READY        |             0 |     2 |     2 | onv_h1_2tpb_cpu_2tpb |
| 529c31da0b0411ea95730288cac7f65cb03b0afc627f |    10006 | READY        |             0 |     0 |     0 | t-test0_5conv_h1_cpu |
+----------------------------------------------+----------+--------------+---------------+-------+-------+----------------------+
```

- UUID          ->  UUID generated for this model during compile time.
- MODEL ID      ->  Neuron runtime identifier for this model.
- MODEL STATUS  ->  READY   = The model is loaded on to the NeuronDevice and active on the NeuronCore. (Inference can be done only on models with READY state)
                    STANDBY = The model is loaded on to the NeuronDevice but another model is currently active on the NeuronCore. (A model switch is needed to start inference)

# View Resource Usage
Each model loaded consumes different amount of memory (host and device), NeuronCore and CPU usage.
The `neuron-top` command can be used to view the memory and NeuronCore usage.

```
$ neuron-top
neuron-top - 2020-02-12 23:03:15
NN Models: 2 total, 2 running
Number of VNCs tracked: 16

0000:00:1c.0 Utilizations: Neuron core0 0.00%, Neuron core1 0.00%, Neuron core2 0.00%, Neuron core3 0.00%,
0000:00:1e.0 Utilizations: Neuron core0 0.00%, Neuron core1 0.00%, Neuron core2 0.00%, Neuron core3 0.00%,

Model ID   Model Name                                                      UUID                               Node ID   Subgraph   Exec. Unit       Host Mem   Device Mem   Neuron core %
10018      1.0.6801.0-/home/ubuntu/benchmarking/compiler_workdir/rn50      d12cf238420d11ea8e270afe835c0a32   3         0          0000:00:1e.0:0   33554816   135290880    0.00
10017      1.0.6801.0-/home/ubuntu/benchmarking/compiler_workdir/rn50      d12cf238420d11ea8e270afe835c0a32   3         0          0000:00:1c.0:0   33554816   135290880    0.00

```
In the above output:
- Model ID      ->  Unique Identifier for models loaded in the Neuron device
- Model Name    ->  Neuron Compiler Version-compiler work directory/User defined model name
- Node ID       ->  For Internal use only
- UUID          ->  Unique Id assigned by the Neuron Compiler for a Model
- Exec. Unit    ->  BDF of Neuron Device followed by the Neuron Core ID, b:d:f.NC
- Host Mem      ->  Host memory consumed by the Model in bytes
- Device Mem    ->  Neuron Device memory consumed by the Model in bytes
- Neuron Core % ->  Utilization % of the neuron core at sample time.  If there are no active inferences this value will be 0.

# Gather Debugging information
Please refer to [Neuron Gatherinfo](./tutorial-neuron-gatherinfo.md)
