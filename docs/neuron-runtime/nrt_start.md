# Getting started:  Installing and Configuring Neuron-RTD

In this getting started guide you will learn how to install Neuron runtime, and configure it for inference. If you'd like to install the Neuron runtime into your own AMI of choice please start with Step 1. If you plan to use a pre-built Deep Learning AMI (recommended) follow these instructions: https://docs.aws.amazon.com/dlami/latest/devguide/launch-config.html. Using DLAMI is recommended as it comes pre-installed with all of the needed Neuron packages. When using DLAMI, you can start with step 3 below.


## Step 1: Launch an Inf1 Instance

Steps Overview: 

1. Select an AMI of your choice, which may be Ubuntu 16.x, Ubuntu 18.x, Amazon Linux 2 based. 
2. Select an Inf1 instance size of your choice (see https://aws.amazon.com/ec2/instance-types/)

## Step 2: Install Neuron-RTD

Steps Overview:

1. Modify yum/apt repository configurations to point to the Neuron repository.
2. Install Neuron-RTD

### Modify yum/apt repository configurations to point to the Neuron repository.


To know your ubuntu version, type this grep cmd. It should be an 18.* or 16.*

```
grep -iw version /etc/os-release
```

### UBUNTU 16

```
sudo tee /etc/apt/sources.list.d/neuron.list > /dev/null <<EOF
deb https://apt.repos.neuron.amazonaws.com xenial main
EOF

wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | sudo apt-key add -
 
sudo apt-get update
sudo apt-get install aws-neuron-runtime
sudo apt-get install aws-neuron-tools
```

**UBUNTU 18**

```
sudo tee /etc/apt/sources.list.d/neuron.list > /dev/null <<EOF
deb https://apt.repos.neuron.amazonaws.com bionic main
EOF

wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | sudo apt-key add -
 
sudo apt-get update
sudo apt-get install aws-neuron-runtime
sudo apt-get install aws-neuron-tools
```

### RPM (AmazonLinux, Centos)

```
sudo tee /etc/yum.repos.d/neuron.repo > /dev/null <<EOF
[neuron]
name=Neuron YUM Repository
baseurl=https://yum.repos.neuron.amazonaws.com
enabled=1
EOF

sudo rpm --import https://yum.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB
sudo yum install aws-neuron-runtime
sudo yum install aws-neuron-tools
```

## Step3 : Configure Neuron-RTD
You can choose your Neuron-RTD mode, either select to run a single instance of the Neuron runtime, or mutiple instances which may be desired to provide your application capabilities like isolation or load balancing.


### Single Neuron-RTD

The default configuration sets up a single Neuron-RTD daemon for all present Neuron devices in the instance.
With the default configuration:
1. Runtime API server listens on a single UDS endpoint `unix:/run/neuron.sock`
2. A single runtime daemon(multi threaded) handles all the inference requests.


### Multiple Neuron-RTD
Multiple runtime daemon might be preferred in some cases for isolation or for load balancing.
The following steps explains configuring 4 Neuron-RTD on a Inf1.6xl instance and let each daemon to manage 1 Neuron device.
When configuring multiple Neuron-RTD, a configuration file needs to be created to specify the API server endpoint (UDP or TCP port) and logical device id it should manage.


## Useful commands

### Identify logical IDs of Inferentia devices
Use `neuron-ls` to enumerate the set of Inferentia chips in the system.

```bash
/opt/aws/neuron/bin/neuron-ls
+--------------+---------+--------+-----------+-----------+------+------+
|   PCI BDF    | LOGICAL | NEURON |  MEMORY   |  MEMORY   | EAST | WEST |
|              |   ID    | CORES  | CHANNEL 0 | CHANNEL 1 |      |      |
+--------------+---------+--------+-----------+-----------+------+------+
| 0000:00:1f.0 |       0 |      4 | 4096 MB   | 4096 MB   |    0 |    1 |
+--------------+---------+--------+-----------+-----------+------+------+ 
| 0000:00:1e.0 |       1 |      4 | 4096 MB   | 4096 MB   |    1 |    1 |
+--------------+---------+--------+-----------+-----------+------+------+ 
| 0000:00:1d.0 |       2 |      4 | 4096 MB   | 4096 MB   |    1 |    1 |
+--------------+---------+--------+-----------+-----------+------+------+ 
| 0000:00:1c.0 |       3 |      4 | 4096 MB   | 4096 MB   |    1 |    0 |
+--------------+---------+--------+-----------+-----------+------+------+ 
```

neuron-rtd can manage one or more devices. Select contigous Inferentia devices to be managed by a single neuron-rtd. 


### Create a configuration file for each instance
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

>sudo tee /opt/aws/neuron/bin/nrtd3.json > /dev/null << EOF
{
"name": "nrtd3",
"server_port": "unix:/run/neuron.sock3",
"devices": [3]
}
EOF

sudo chmod 755 /opt/aws/neuron/bin/nrdt0.json
sudo chmod 755 /opt/aws/neuron/bin/nrtd1.json
sudo chmod 755 /opt/aws/neuron/bin/nrtd2.json
sudo chmod 755 /opt/aws/neuron/bin/nrtd3.json
```


### Start the services
#### Stop the default service
```bash
sudo systemctl stop neuron-rtd
```

#### Start the new services
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


