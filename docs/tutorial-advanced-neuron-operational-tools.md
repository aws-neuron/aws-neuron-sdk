# Advanced Usage: Neuron Tools

# Identifying Inferentia devices

To identify number of Inferentia devices in given instance `neuron-ls` command can be used.

```
$ neuron-ls
+--------------+---------+--------+-----------+-----------+---------+------+------+
|   PCI BDF    | LOGICAL | NEURON |  MEMORY   |  MEMORY   |   DMA   | EAST | WEST |
|              |   ID    | CORES  | CHANNEL 0 | CHANNEL 1 | ENGINES |      |      |
+--------------+---------+--------+-----------+-----------+---------+------+------+
| 0000:00:1f.0 |       0 |      4 | 4096 MB   | 4096 MB   |      12 |    0 |    0 |
+--------------+---------+--------+-----------+-----------+---------+------+------+
```

The above output is taken from a inf1.xl instance.
The first column shows the PCI Bus Device Function ID.
The second column shows logical ID assigned to the device. This logical ID is used during multiple Neuron-rtd configuration.
The third columns shows the number of NeuronCores in the inferentia device.
The last two columns shows the connection to any other inferentia devices; since this is a single inferentia device, those are empty.

# Neuron Groups
Multiple NeuronCore can be combined to form a NeuronGroup.
Neuron framework layer would automatically create a default NeuronGroup.
To view list of available NCG the following command can be used.
```
$ neuron-cli list-ncg
MLA count 1 NC count 4
+-------+-----------------+----------+-----------------+----------------+
| NCG ID| VNC START INDEX | NC COUNT | MLA START INDEX | NC START INDEX |
+-------+-----------------+----------+-----------------+----------------+
|     1 |               0 |        1 |               0 |              0 |
|     2 |               1 |        1 |               0 |              1 |
|     3 |               2 |        2 |               0 |              2 |
+-------+-----------------+----------+-----------------+----------------+
```
The above output shows there 3 NCG and 2 NCG contains single NC and one NCG has 2 NC.

If for some reason, app is crashed, the framework created NCGs would be left behind.
To delete it `neuron-cli destroy-ncg` can be used.

# Listing Models
Models can be loaded into a NCG.
Multiple models can be loaded into a NCG but only one can be in STARTED state at a given moment.
Inference can be done only on the models with STARTED state.

To view all the models `neuron-cli list-model` can be used.
```
$ neuron-cli list-model
Found 3 models
10003 MODEL_STATUS_LOADED 1
10001 MODEL_STATUS_STARTED 1
10002 MODEL_STATUS_STARTED 1
```

In the above output 10001 and 10002 are unique identifier for models loaded in Inferentia.

To start/stop/unload a model `neuron-cli start/stop/unload` command can be used.

# View Resource Usage
Each model loaded in Inferentia consumes different amount of memory(host and device), NeuronCore and CPU usage.
This usage can be viewed by using neuron-top.
```
$ neuron-top
neuron-top - 2019-11-13 23:57:08
NN Models: 3 total, 2 running
Number of VNCs tracked: 2
0000:00:1f.0 Utilizations: Neuron core0 0.00, Neuron core1 0.00, Neuron core2 0, Neuron core3 0,
DLR Model   Node ID   Subgraph   Exec. Unit       Host Mem   MLA Mem     Neuron core %
10003       1         0          0000:00:1f.0:0   384        135660544   0.00
10001       3         0          0000:00:1f.0:0   384        67633152    0.00
10002       1         0          0000:00:1f.0:1   384        135660544   0.00
```
