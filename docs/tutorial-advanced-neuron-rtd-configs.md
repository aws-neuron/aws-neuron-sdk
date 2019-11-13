# Tutorial: Advanced Configurations for Neuron-RTD on an Inf1 instance

##  Steps Overview:

1. Prerequisite
2. Select your deployment configuration. Examples given here:
    1. Single Neuron-RTD for all present Inferentia devices in the instance
    2. 4 x Inferentia with 1 Neuron-RTD per Inferentia

## Step1: Prerequisite

[Getting started:  Installing and Configuring Neuron-RTD on an Inf1 instance](./getting-started-neuron-rtd.md)

## Step2 : Configure Neuron-RTD

### Single Neuron-RTD for all INferntia devices present

The default configuration sets up a single Neuron-RTD daemon for all present Inferentias in the instance. This can be modified if desired by configuring additional Neuron-RTD mappings to each set of Inferentia chips desired:

### Single Neuron-RTD per Inferentia in an instance with 4 Inferentia (6xl):

Steps Overview:

Step 1: stop the current Neuron-rtd
Step2: use the Neuron utility neuron-ls to enumerate the set of Inferentia chips in the system 
Step3: create a configuration file for each Neuron-rtd you wish to launch, with the  1 or more  Inferentia chips desired to be mapped to that Neuron-rtd instance, and the listening port for it.

Find the Logical ID for each Inferentia:

```
     >/opt/aws/neuron/bin/neuron-ls
+--------------+---------+--------+-----------+-----------+---------+------+------+
|   PCI BDF    | LOGICAL | NEURON |  MEMORY   |  MEMORY   |   DMA   | EAST | WEST |
|              |   ID    | CORES  | CHANNEL 0 | CHANNEL 1 | ENGINES |      |      |
+--------------+---------+--------+-----------+-----------+---------+------+------+
| 0000:00:1f.0 |       0 |      4 | 4096 MB   | 4096 MB   |      12 |    0 |    1 |
+--------------+---------+--------+-----------+-----------+---------+------+------+ 
| 0000:00:1e.0 |       1 |      4 | 4096 MB   | 4096 MB   |      12 |    1 |    1 |
+--------------+---------+--------+-----------+-----------+---------+------+------+ 
| 0000:00:1d.0 |       2 |      4 | 4096 MB   | 4096 MB   |      12 |    1 |    1 |
+--------------+---------+--------+-----------+-----------+---------+------+------+ 
| 0000:00:1c.0 |       3 |      4 | 4096 MB   | 4096 MB   |      12 |    1 |    0 |
+--------------+---------+--------+-----------+-----------+---------+------+------+ 
```

```

```


Stop Neuron-RTD and create configuration files for each desired Neuron-RTD(must be in json format):

```
>sudo systemctl stop neuron-rtd

>sudo tee /opt/aws/neuron/bin/nrtd0.json > /dev/null << EOF
{
"name": "nrtd0",
"server_port": "unix:/run/neuron.sock0",
"infa_devices": [0]
}
EOF

>sudo tee /opt/aws/neuron/bin/nrtd1.json > /dev/null << EOF
{
"name": "nrtd1",
"server_port": "unix:/run/neuron.sock1",
"infa_devices": [1]
}
EOF

>sudo tee /opt/aws/neuron/bin/nrtd2.json > /dev/null << EOF
{
"name": "nrtd2",
"server_port": "unix:/run/neuron.sock2",
"infa_devices": [2]
}
EOF

>sudo tee /opt/aws/neuron/bin/nrtd3.json > /dev/null << EOF
{
"name": "nrtd3",
"server_port": "unix:/run/neuron.sock3",
"infa_devices": [3]
}
EOF

sudo chmod 755 /opt/aws/neuron/bin/nrdt0.json
sudo chmod 755 /opt/aws/neuron/bin/nrtd1.json
sudo chmod 755 /opt/aws/neuron/bin/nrtd2.json
sudo chmod 755 /opt/aws/neuron/bin/nrtd3.json
```

Start the services:

```
>sudo systemctl start neuron-rtd@nrtd0
>sudo systemctl start neuron-rtd@nrtd1
>sudo systemctl start neuron-rtd@nrtd2
>sudo systemctl start neuron-rtd@nrtd3
```

Verify the services are up and running. This example shows one of the Neuron-RTD daemons (Neuron-RTD0):

```
>sudo systemctl status neuron-rtd@nrtd0
● neuron-rtd@nrtd0.service - Neuron Runtime Daemon nrtd0
   Loaded: loaded (/lib/systemd/system/neuron-rtd@.service; disabled; vendor preset: enabled)
   Active: active (running) since Wed 2019-11-13 00:24:25 UTC; 8s ago
 Main PID: 32446 (neuron-rtd)
    Tasks: 14 (limit: 4915)
   CGroup: /system.slice/system-neuron\x2drtd.slice/neuron-rtd@nrtd0.service
           └─32446 /opt/aws/neuron/bin/neuron-rtd -i nrtd0 -c /opt/aws/neuron/config/neuron-rtd.config

Nov 13 00:23:39 ip-10-1-255-226 neuron-rtd[32446]: nrtd[32446]: [TDRV:reset_mla] Resetting 0000:00:1f.0
Nov 13 00:23:39 ip-10-1-255-226 nrtd[32446]: [TDRV:reset_mla] Resetting 0000:00:1f.0
Nov 13 00:24:00 ip-10-1-255-226 neuron-rtd[32446]: nrtd[32446]: [hal] request seq: 3, cmd: 1 timed out
Nov 13 00:24:00 ip-10-1-255-226 nrtd[32446]: [hal] request seq: 3, cmd: 1 timed out
Nov 13 00:24:25 ip-10-1-255-226 neuron-rtd[32446]: nrtd[32446]: [TDRV:tdrv_init_one_mla_phase2] Initialized Inferentia: 0000:00:1f.0
Nov 13 00:24:25 ip-10-1-255-226 nrtd[32446]: [TDRV:tdrv_init_one_mla_phase2] Initialized Inferentia: 0000:00:1f.0
Nov 13 00:24:25 ip-10-1-255-226 neuron-rtd[32446]: E1113 00:24:25.605502817   32446 socket_utils_common_posix.cc:197] check for SO_REUSEPORT: {"created":"@1573604665.605493059","description":"SO_REUSEPORT unavailab
Nov 13 00:24:25 ip-10-1-255-226 systemd[1]: Started Neuron Runtime Daemon nrtd0.
Nov 13 00:24:25 ip-10-1-255-226 neuron-rtd[32446]: nrtd[32446]: [NRTD:RunServer] Server listening on unix:/run/neuron.sock0
Nov 13 00:24:25 ip-10-1-255-226 nrtd[32446]: [NRTD:RunServer] Server listening on unix:/run/neuron.sock0
lines 1-18/18 (END)
```


