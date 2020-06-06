# Troubleshooting Neuron Runtime

This document aims to provide more information on how to fix issues you might encounter while using the Neuron Runtime. For each issue we will provide an explanation of what happened and what can potentially correct the issue. 

If you haven't read it already, please familiarize yourself with our [getting started](./nrt_start.md) documentation and usage examples.  If your issue is still not resolved or you have a more nuanced problem, contact us via [issues](https://github.com/aws/aws-neuron-sdk/issues) posted to this repo, the [AWS Neuron developer forum](https://forums.aws.amazon.com/forum.jspa?forumID=355), or through AWS support.

---

### Topics

**What is going wrong?**  

[Runtime installation failed](#installation-failed)

[Neuron Runtime services fail to start](#neuron-services-fail-to-start)

[Load model failure](#load-model-failure)

[Inferences are failing](#inferences-are-failing)

<br/>

**Additional helpers:**

[Getting Started Guide](./nrt_start.md)

[Error Codes](#error-codes)

---



## Runtime installation failed

There's a few reasons this might occur.  Here's a list of things to double check on:

#### 1. Supported OS

Ensure you're attempting installation on a supported operating system.  Neuron SDK currently supports Ubuntu 16, Ubuntu 18, and Amazon Linux 2.

#### 2. Check configuration of Neuron repos

1. Does the yum/apt repository configuration point to the correct Neuron repository?

* Amazon Linux 2:

```bash
sudo tee /etc/yum.repos.d/neuron.repo > /dev/null <<EOF
[neuron]
name=Neuron YUM Repository
baseurl=https://yum.repos.neuron.amazonaws.com
enabled=1
EOF
```

* Ubuntu 16:

```bash
sudo tee /etc/apt/sources.list.d/neuron.list > /dev/null <<EOF
deb https://apt.repos.neuron.amazonaws.com xenial main
EOF
```

* Ubuntu 18:

```bash
sudo tee /etc/apt/sources.list.d/neuron.list > /dev/null <<EOF
deb https://apt.repos.neuron.amazonaws.com bionic main
EOF
```

2. Is the public key registered?

- Amazon Linux 2:

```bash
sudo rpm --import https://yum.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB
```

- Ubuntu 16, Ubuntu 18

```bash
wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | sudo apt-key add -
```

3. Package list is updated?

- Ubuntu 16, Ubuntu 18

```bash
sudo apt-get update
```

#### 3. Check the package version

1. What version is attempting to be installed?

- Amazon Linux, CentOS

```bash
sudo yum list | grep aws-neuron-runtime
aws-neuron-runtime.x86_64              1.0.3978.0-1                  neuron     
aws-neuron-runtime-base.x86_64         1.0.3438.0-1                  neuron
```

- Ubuntu 16, Ubuntu 18

```bash
sudo apt list | grep aws-neuron-runtime
aws-neuron-runtime/unknown 1.0.3978.0 amd64
aws-neuron-runtime-base/unknown 1.0.3438.0 amd64
```

2. If there is a known issue with a version, we will captured it in the release notes and/or create an issue in our GitHub repo.  Please check the [release notes](../../release-notes/neuron-runtime.md) for more details on the version you're running or installing.



#### 4. neuron-rtd or neuron-discovery failed to start during installation

See [Neuron Runtime services fail to start](#neuron-services-fail-to-start) for more help.  The components are likely installed at this point, but you're experiencing a problem related to the startup, which is attempted during install.

---



## Neuron Runtime services fail to start

There are two different runtime services, neuron-rtd and neuron-discovery, that are needed for correct functionality of an Inf1 instance.  If neuron-discovery is failing, you might have a system configuration issue. Try running on a different instance and if the problem persists contact us via [issues](https://github.com/aws/aws-neuron-sdk/issues), the [AWS Neuron developer forum](https://forums.aws.amazon.com/forum.jspa?forumID=355), or through AWS support.

If the neuron-rtd service is failing to start, you may be experiencing failure due to (1) a conflict with another instance of neuron-rtd, (2) insufficient number of hugepages allocated by the OS, or (3) a lack of system privileges needed to start the service.  Read on for more details on each scenario.



### Another Instance of Runtime is Running

##### What Went Wrong?

A new instance of neuron-rtd cannot start if another neuron-rtd is already running and bound to the same Neuron devices.  Please read on for how to detect this scenario, but if you're having trouble configuring two or more runtimes on the same Inf1 instance, see detailed config instructions [here](./nrt_start.md#multiple-neuron-rtd).

##### How To Find Out?

Check for error messages in syslog similar to:

```bash
nrtd[nnnnn]: .... Failed to lock MLA. File /run/infa/infa-0000:00:1f.0.json is locked
```

##### How To Fix?

Terminate the current neuron-rtd that is already running before starting the new instance. 

```bash
sudo systemctl stop neuron-rtd
```



### Insufficient amount of hugepages available for the Runtime

##### What Went Wrong?

Runtime requires 128 2MB hugepages per Inferentia.  If you have less than this, the neuron-rtd service is going to fail to start and emit errors to the syslog indicating it failed due to hugetlb allocation.  

The most common cause of this error is reuse of an AMI built on an instance type with less Inferentias than the instance type it was later launched on.  For example, if you built an AMI with aws-neuron-runtime using an inf1.xlarge, but then used the same AMI on an inf1.6xlarge, neuron-rtd would fail because the original hugepage setting was created for a single Neuron device.

##### How To Find Out?

Check for error messages in syslog similar to:

```bash
nrtd[nnnnn]: ....  Failed to mmap with hugetlb
nrtd[nnnnn]: ....  Attempt to preallocate 128 hugetlb pages failed!
```

##### How To Fix?

Detailed information on how to configure the number of hugepages on an instance is documented [here](./nrt_start.md#step-3-configure-nr_hugepages).  If you're hitting this issue due to your AMI being built on an instance that has less Inferntias than the target it's lauched on, there's two ways to fix the issue.  Either update the configuration of the AMI to be specific to the Inf1 instance type, or create a script to set the number of hugepages at boot.  The scripted approach is currently part of the DLAMI if you need an example to follow.  Please see further instruction on how to configure the number of hugepages [here](./nrt_start.md#step-3-configure-nr_hugepages).

TODO: provide a script.

### Incorrect User privileges

Trying to start Runtime without being root/sudo results in an authentication password request.

```bash
$ systemctl start neuron-rtd
==== AUTHENTICATING FOR org.freedesktop.systemd1.manage-units ===
Authentication is required to start 'Neuron-rtd.service'.
Authenticating as: Ubuntu (ubuntu)
Password:
```

This will fail due to lack of root privileges needed for system memory allocations.  The neuron-rtd needs the CAP_SYS_ADMIN capability, but will drop all elevated capabilities immediately following the memory allocations.

##### What Went Wrong?

Runtime was attempted to be started as a non-root user.

##### How To Fix?

Ensure neuron-rtd is always started as sudo or root.

----



## Load Model Failure

There are a variety of reasons for a model load to fail. The most common ones are listed below.
If the solutions below are insufficient, please reach out to the Neuron team by posting the relevant details on GitHub [issues](https://github.com/aws/aws-neuron-sdk/issues).

### Neff couldn't be extracted

##### What went wrong?

Host ran out of disk space while trying to extract the NEFF object

##### How to find out?

Syslog will show an error similar to the following:

```
nrtd[nnnnn]: ....  Failed to untar (tar -xsvf /tmp/neff.XXXXX -C /tmp/neff.YYYYY > /dev/null)
```

##### How to fix?

Increase /tmp space by removing unused files or taking other measures to increase the available disk size under /tmp.

### Unsupported NEFF Version

##### What Went Wrong?

The version of the NEFF file is incompatible with the version of Neuron Runtime that has received it.  The NEFF is generated by the compiler and the Neuron Runtime is intended to support multiple NEFF versions; however, this may require updating the runtime to gain support for newer NEFF formats.

##### How To Find Out?

Check for error messages in syslog similar to:

```
nrtd[nnnnn]: ....  NEFF version mismatch supported: 1.1 received: 2.0
```

Error Code: 10

##### How To Fix?

Use compatible versions of Neuron Compiler and Runtime. Updating to the latest version of both Neuron Compiler and Neuron Runtime is the simplest solution.  If updating one of the two is not an option, please refer to the [release notes](../../release-notes/neuron-runtime.md) of the Neuron Runtime to determine NEFF version support. 

### Invalid NEFF

##### What Went Wrong?

Validation is performed on the NEFF file before attempting to load it.  When that validation fails, it usually indicates that the compiler produced an invalid NEFF (possible bug in the Neuron Compiler).

##### How To Find Out?

Check for error messages in syslog similar to:

```
nrtd[nnnn]: .... Failed .... neff.json
nrtd[nnnn]: .... Failed/Unsupported/Invalid .... NEFF
nrtd[nnnn]: .... Wrong NEFF file size
nrtd[nnnn]: .... NEFF upload failed
```

Error Code: 2

##### How To Fix?

Try recompiling with the latest version of Neuron Compiler. If that does not work, report issue to Neuron by posting the relevant details on GitHub [issues](https://github.com/aws/aws-neuron-sdk/issues).

### Bad Memory Access by NEFF

##### What Went Wrong?

To ensure the execution of the NEFF, neuron-rtd is monitoring for illegal and unaligned access to Inferentia memory.  When this occurs, the NEFF will fail to load.

##### How To Find Out?

Check for error messages in syslog similar to:

```
nrtd[nnnn]:.... address ... must be X byte aligned
```

Error Code: 2

##### How To Fix?

Report issue to Neuron by posting the relevant details on GitHub [issues](https://github.com/aws/aws-neuron-sdk/issues).

### Insufficient resources

##### What Went Wrong?

Loading the NEFF requires more host or Inferentia resources (usually memory on the host or Inferentia) then available on the instance.  This issue may be contextual in that other applications or models consumed the needed resources before the current NEFF could be loaded.

##### How To Find Out?

Check for error messages in syslog similar to:

```
nrtd[nnnn]:.... Failed to allocate buffer in MLA DRAM for 
nrtd[nnnn]:.... Failed to alloc hugetlb
```

Error Code: 4

##### How To Fix?

As the error is contextual to what's going on with your instance, the exact next step is unclear.  Here are some ideas on what might help free up space.  Start by unloading some of the loaded models.  If this is not the issue, you may need to increase the number of huge pages on the instance (instructions for this are [here](./nrt_start.md#step-3-configure-nr_hugepages)).  If you're still stuck, moving to a larger Inf1 instance size may provide the additional resources needed.

### Insufficient number of NeuronCores available to load a NEFF

##### What Went Wrong?

The NEFF requires more NeuronCores than available on the instance.

##### How To Find Out?

Check for error messages in syslog similar to:

```
nrtd[nnnn]:.... Requested number of NCs: X exceeds the total available number: Y
nrtd[nnnn]:.... Insufficient number of VNCs: X, required: Y
```

Error Code: 9

##### How To Fix?

The NeuronCores may be in use by models you are not actively using.  Ensure you've unloaded models you're not using and deleted unused NCGroups.  If this is still a problem, moving to a larger Inf1 instance size with additional NeuronCores may help.

---



## Inferences are failing

### Wrong Model Id

##### What Went Wrong?

An inference request had a model id that is invalid or not in a running state.

##### How To Find Out?

Check for error messages in syslog similar to:

```
nrtd[nnnnn]:....Failed to find model: 10001
```

##### How To Fix?

Ensure your application is only inferring against models that are running on the Inferentia.



### Bad or incorrect inputs

NEFF contains information of the number of input feature maps required by the model.  If inputs to the model don't match the expected number/size of the input, inferene will fail.

##### What Went Wrong?

Mismatch in either the number of expected inputs or the size of the inputs.

##### How To Find Out?

Check for error messages in syslog similar to:

```
nrtd[nnnnn]: ....  Wrong number of ifmaps, expected: X, received: Y
nrtd[nnnnn]: ....  Invalid data length for [input:0], received X, expected Y
```

##### How To Fix?

Ensure the correct number of inputs and correct sizes are used.



### Numerical errors on the Model 

##### What Went Wrong?

The inference generated NaNs during execution, which is usually an indication of model or input errors.

##### How To Find Out?

Check for error messages in syslog similar to:

```
nrtd[nnnnn]: ....  Error notifications found on NC .... INFER_ERROR_SUBTYPE_NUMERICAL
```

##### How To Fix?

Report issue to Neuron by posting the relevant details on GitHub [issues](https://github.com/aws/aws-neuron-sdk/issues).



### Inference Timeout

##### What Went Wrong?

It's possible that the Neuron Compiler built a NEFF with errors, e.g. the NEFF might describe incorrect internal data flows or contain incorrect instruction streams.  When this happens, it will potentially result in a timeout during inference.

##### How To Find Out?

Check for error messages in syslog similar to:

```
nrtd[nnnnn]: ....  Error: DMA completion timeout in ....
```

Error Code: 5

##### How To Fix?

Report issue to Neuron by posting the relevant details on GitHub [issues](https://github.com/aws/aws-neuron-sdk/issues).

---


### Error codes

Non-exhaustive list of errors returned by the Neuron Runtime.  We've only captured the returns that are most relevant to debug.  When these errors occur, an entry is logged to syslog and most ML framework clients will surface the error code with a specific message from the runtime.

| Code | Error                | Most likely explanation(s)                                   |
| ---- | -------------------- | ------------------------------------------------------------ |
| 2    | Invalid              | The NEFF provided or inference requested is invalid in some way.  Either incomplete, corrupt, missing, or has a mismatch between model and input tensor name/size. |
| 4    | Resource             | Not enough of something we needed; usually memory on the host, memory in the device, or NeuronCores. |
| 5    | Timeout              | Something went sideways during inference and breached our expected threshold for a response. |
| 6    | Hardware Error       | ?!?!  Most likely a GRPC call failed.  If you suspect a real hardware problem, please let us know asap. |
| 7    | Queue full           | Requests for inference have exceeded our capacity to service them.  Tapping the breaks. |
| 8    | Failed state         | An inference was submitted for a model that is in Failed state.  The model encountered unrecoverable error and cannot be used |
| 9    | Resource NC          | Specifically, there's not enough NCs to service the request  |
| 10   | Unsupported Version  | This runtime doesn't support that version of a compiled NN, also known as a NEFF.  The error message provided will be explicit in this case. |
| 1002 | Bad input            | Invalid input has been submitted to infer, e.g. missing some of the input tensors, incorrect input tensor sizes |
| 1003 | Numerical errors     | Inference is completed, but numerical errors were reported while executing the inference, e.g. NaN |
| 1004 | Non-numerical errors | Inference is completed, but non-numerical errors were reported while executing the inference, e.g. event double-clear. |








