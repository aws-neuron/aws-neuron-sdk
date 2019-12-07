# Neuron Tools

# Identifying Inferentia devices

To identify number of Inferentia devices in a given instance use the `neuron-ls` command.

```
$ neuron-ls
+--------------+---------+--------+-----------+-----------+------+------+
|   PCI BDF    | LOGICAL | NEURON |  MEMORY   |  MEMORY   | EAST | WEST |
|              |   ID    | CORES  | CHANNEL 0 | CHANNEL 1 |      |      |
+--------------+---------+--------+-----------+-----------+------+------+
| 0000:00:1f.0 |       0 |      4 | 4096 MB   | 4096 MB   |    0 |    0 |
+--------------+---------+--------+-----------+-----------+------+------+
```

The above output is taken from an Inf1.xlarge instance.
The first column shows the PCI Bus Device Function (BDF) ID.
The second column shows logical ID assigned to the device. This logical ID is used during Neuron-rtd configuration.
The third column shows the number of NeuronCores in the inferentia device.
The last two columns show the connection to any other inferentia devices; since this is a single inferentia device, those are empty.

# NeuronCore Groups
Multiple NeuronCores(NC) can be combined to form a NeuronCore Group (NCG).
Neuron framework layer will automatically create a default NeuronCore Group.
To view list of available NCGs the following command can be used.
```
$ neuron-cli list-ncg
Device 1 NC count 4
+-------+----------+--------------------+----------------+
| NCG ID| NC COUNT | DEVICE START INDEX | NC START INDEX |
+-------+----------+--------------------+----------------+
|     1 |        1 |                  0 |              0 |
|     2 |        1 |                  0 |              1 |
|     3 |        2 |                  0 |              2 |
+-------+-----------------+----------+-----------------+----------------+
```

If there is a need to delete the framework created NCGs the `neuron-cli destroy-ncg` command can be used.

# Listing Models
Multiple models can be loaded into a single NCG but only one can be in STARTED state at any given moment.
Inference can be done only on the models with a STARTED state.

The `neuron-cli list-model` command should be used to view all the models.
```
$ neuron-cli list-model
Found 3 models
10003 MODEL_STATUS_LOADED 1
10001 MODEL_STATUS_STARTED 1
10002 MODEL_STATUS_STARTED 1
```

In the above output 10001 and 10002 are unique identifiers for models loaded in a Neuron device.

The command `neuron-cli start/stop/unload` can be used to start/stop/unload a model.

# View Resource Usage
Each model loaded consumes different amount of memory (host and device), NeuronCore and CPU usage.
The `neuron-top` command can be used to biew the memory usage.

```
$ neuron-top
neuron-top - 2019-11-13 23:57:08
NN Models: 3 total, 2 running
Number of VNCs tracked: 2
0000:00:1f.0 Utilizations: Neuron core0 0.00, Neuron core1 0.00, Neuron core2 0.00, Neuron core3 0.00,
DLR Model   Node ID   Subgraph   Exec. Unit       Host Mem   Device Mem     Neuron core %
10003       1         0          0000:00:1f.0:0   384        135660544      0.00
10001       3         0          0000:00:1f.0:0   384        67633152       0.00
10002       1         0          0000:00:1f.0:1   384        135660544      0.00
```


# Gather Debugging information
Please refer to [Neuron Gatherinfo](./tutorial-neuron-gatherinfo.md)
