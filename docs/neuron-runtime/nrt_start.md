</br>
</br>

Please view our documentation at **[https://awsdocs-neuron.readthedocs-hosted.com/](https://awsdocs-neuron.readthedocs-hosted.com/)** 

**Please note this file will be deprecated.**

</br>
</br>



# Getting started:  Installing and Configuring Neuron-RTD

In this getting started guide you will learn how to install Neuron runtime, and configure it for inference. 


## Step 1: Launch an Inf1 Instance and Install runtime packages

1. Select an AMI of your choice, which may be Ubuntu 16.x, Ubuntu 18.x, or Amazon Linux 2 based. Refer to the [Neuron installation guide](../neuron-install-guide.md) for details. 
2. Select an Inf1 instance size of your choice (see https://aws.amazon.com/ec2/instance-types/inf1/)
3. Install the following packages `aws-neuron-dkms aws-neuron-runtime-base aws-neuron-runtime aws-neuron-tools`. Refer to the [Neuron installation guide](../neuron-install-guide.md) for details. 

## Step 2: Configure Neuron-RTD
You can choose your Neuron-RTD mode, either select to run a single instance of the Neuron runtime, or mutiple instances which may be desired to provide your application capabilities like isolation or load balancing.


### Single Neuron-RTD
The default configuration sets up a single Neuron-RTD daemon for all present Neuron devices in the instance.
With the default configuration:
1. Runtime API server listens on a single UDS endpoint `unix:/run/neuron.sock`
2. A single runtime daemon(multi threaded) handles all the inference requests.


### Multiple Neuron-RTD
Multiple runtime daemon might be preferred in some cases for isolation or for load balancing.

When configuring multiple Neuron-RTD, a configuration file needs to be created to specify the API server endpoint (UDP or TCP port) and logical device id it should manage.

The following steps explains configuring four Neuron-RTD on an inf1.6xl instance and let each daemon to manage 1 Neuron device.

#### Identify logical IDs of Inferentia devices
Use `neuron-ls` to enumerate the set of Inferentia chips avaliable in the system.

```bash
/opt/aws/neuron/bin/neuron-ls
+--------+--------+--------+-----------+--------------+---------+---------+---------+
| NEURON | NEURON | NEURON | CONNECTED |     PCI      | RUNTIME | RUNTIME | RUNTIME |
| DEVICE | CORES  | MEMORY |  DEVICES  |     BDF      | ADDRESS |   PID   | VERSION |
+--------+--------+--------+-----------+--------------+---------+---------+---------+
| 0      | 4      | 8 GB   | 1         | 0000:00:1c.0 | NA      | 12410   | NA      |
| 1      | 4      | 8 GB   | 2, 0      | 0000:00:1d.0 | NA      | 12410   | NA      |
| 2      | 4      | 8 GB   | 3, 1      | 0000:00:1e.0 | NA      | 12410   | NA      |
| 3      | 4      | 8 GB   | 2         | 0000:00:1f.0 | NA      | 12410   | NA      |
+--------+--------+--------+-----------+--------------+---------+---------+---------+
```

neuron-rtd can manage one or more devices. Select contigous Inferentia devices to be managed by a single neuron-rtd. 


#### Create a configuration file for each instance
Create a configuration file for each Neuron-rtd you wish to launch, with one or more Inferentia chips desired to be mapped to that Neuron-rtd instance, and the listening port for it.

```bash
sudo tee /opt/aws/neuron/bin/nrtd0.json > /dev/null << EOF
{
"name": "nrtd0",
"server_port": "unix:/run/neuron.sock0",
"devices": [0]
}
EOF

sudo tee /opt/aws/neuron/bin/nrtd1.json > /dev/null << EOF
{
"name": "nrtd1",
"server_port": "unix:/run/neuron.sock1",
"devices": [1]
}
EOF

sudo tee /opt/aws/neuron/bin/nrtd2.json > /dev/null << EOF
{
"name": "nrtd2",
"server_port": "unix:/run/neuron.sock2",
"devices": [2]
}
EOF

sudo tee /opt/aws/neuron/bin/nrtd3.json > /dev/null << EOF
{
"name": "nrtd3",
"server_port": "unix:/run/neuron.sock3",
"devices": [3]
}
EOF

sudo chmod 755 /opt/aws/neuron/bin/nrtd0.json
sudo chmod 755 /opt/aws/neuron/bin/nrtd1.json
sudo chmod 755 /opt/aws/neuron/bin/nrtd2.json
sudo chmod 755 /opt/aws/neuron/bin/nrtd3.json
```


#### Start the services
##### Stop the default service
```bash
sudo systemctl stop neuron-rtd
```

##### Start the new services
```bash
sudo systemctl start neuron-rtd@nrtd0
sudo systemctl start neuron-rtd@nrtd1
sudo systemctl start neuron-rtd@nrtd2
sudo systemctl start neuron-rtd@nrtd3
```

Verify the services are up and running. This example shows one of the Neuron-RTD daemons (Neuron-RTD0):

```bash
sudo systemctl status neuron-rtd@nrtd0
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


