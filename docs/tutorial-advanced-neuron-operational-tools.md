# Neuron Tools

# Identifying Inferentia devices

To identify number of Inferentia devices in a given instance use the `neuron-ls`.

```
$ neuron-ls
+--------------+---------+--------+-----------+-----------+---------+------+------+
|   PCI BDF    | LOGICAL | NEURON |  MEMORY   |  MEMORY   |   DMA   | EAST | WEST |
|              |   ID    | CORES  | CHANNEL 0 | CHANNEL 1 | ENGINES |      |      |
+--------------+---------+--------+-----------+-----------+---------+------+------+
| 0000:00:1f.0 |       0 |      4 | 4096 MB   | 4096 MB   |      12 |    0 |    0 |
+--------------+---------+--------+-----------+-----------+---------+------+------+
```

The above output is taken Inf1.xlarge instance.
The first column shows the PCI Bus Device Function ID.
The second column shows logical ID assigned to the device. This logical ID is used during multiple Neuron-rtd configuration.
The third columns shows the number of NeuronCores in the inferentia device.
The last two columns shows the connection to any other inferentia devices; since this is a single inferentia device, those are empty.

# Neuron Groups
Multiple NeuronCores(NC) can be combined to form a NeuronCore Group (NCG).
Neuron framework layer would automatically create a default NeuronCore Group.
To view list of available NCGs the following command can be used.
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

If for some reason, need to delete the framework created NCGs `neuron-cli destroy-ncg` can be used.

# Listing Models
Models can be loaded into a NCG.
Multiple models can be loaded into a single NCG but only one can be in STARTED state at a given moment.
Inference can be done only on the models with STARTED state.

To view all the models `neuron-cli list-model` can be used.
```
$ neuron-cli list-model
Found 3 models
10003 MODEL_STATUS_LOADED 1
10001 MODEL_STATUS_STARTED 1
10002 MODEL_STATUS_STARTED 1
```

In the above output 10001 and 10002 are unique identifier for models loaded in Neuron device.

To start/stop/unload a model `neuron-cli start/stop/unload` command can be used.

# View Resource Usage
Each model loaded consumes different amount of memory (host and device), NeuronCore and CPU usage.
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

TODO describe VNC
TODO change output with example that shows utilization
