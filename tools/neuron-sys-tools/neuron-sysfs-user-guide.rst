.. _neuron-sysfs-ug:

Neuron Sysfs User Guide
=======================

.. contents:: Table of contents
    :local:
    :depth: 3

Introduction
------------
The kernel provides a few ways in which userspace programs can get system information from the kernel space. Sysfs is one common way to do so. It is a virtual filesystem typically mounted on the ``/sys`` directory and contains information about hardware devices attached to the system and about drivers handling those devices. By navigating the hierarchical structure of the sysfs filesystem and viewing the information provided by its files and directories, you can gather valuable information that can help diagnose and resolve a wide range of hardware and system issues.

Thus a sysfs filesystem is set up per Neuron Device under ``/sys/devices/virtual/neuron_device`` to give you an insight into the Neuron Driver and Runtime at system level. By performing several simple CLIs such as reading or writing to a sysfs file, you can get information such as Runtime status, memory usage, Driver info etc. You can even create your own shell scripts to query Runtime and Driver statistics from sysfs and generate customized reports.

This user guide will first explain the Neuron sysfs structure and then introduce many ways where you can perform diagnostic works with Neuron sysfs.


Neuron Sysfs Filesystem Structure
---------------------------------
High Level Overview
^^^^^^^^^^^^^^^^^^^

Here is the high level structure of the Neuron sysfs filesystem, where the total and present counters are not shown:

.. code-block:: bash

  /sys/devices/virtual/neuron_device/
  ├── neuron0/
  │   ├── subsystem
  │   ├── uevent
  │   ├── core_count
  │   ├── reset
  │   ├── power/
  │   │   ├── async
  │   │   ├── control
  │   │   ├── runtime_active_time
  │   │   ├── runtime_active_kids
  │   │   └── ...
  │   ├── info/
  │   │   ├── notify_delay
  │   │   └── architecture/
  │   │       ├── arch_type
  │   │       ├── device_name
  │   │       └── instance_type
  │   ├── neuron_core0/
  │   │   ├── info/
  │   │   │   └── architecture/
  │   │   │       └── arch_type
  │   │   ├── stats/
  │   │   │   ├── status/
  │   │   │   │   ├── exec_bad_input
  │   │   │   │   ├── hw_error
  │   │   │   │   ├── infer_failed_to_queue
  │   │   │   │   ├── resource_nc_error
  │   │   │   │   ├── unsupported_neff_version
  │   │   │   │   ├── failure
  │   │   │   │   ├── infer_completed_with_error
  │   │   │   │   ├── invalid_error
  │   │   │   │   ├── success
  │   │   │   │   ├── generic_error
  │   │   │   │   ├── infer_completed_with_num_error
  │   │   │   │   ├── resource_error
  │   │   │   │   └── timeout
  │   │   │   ├── memory_usage/
  │   │   │   │   ├── device_mem
  │   │   │   │   └── host_mem
  │   │   │   └── other_info/
  │   │   │       ├── inference_count
  │   │   │       ├── model_load_count
  │   │   │       └── reset_count
  │   │   └── ...
  │   ├── neuron_core1/
  │   │   ├── info/
  │   │   │   └── ...
  │   │   └── stats/
  │   │       └── ...
  │   └── ...
  ├── neuron1
  ├── neuron2
  ├── neuron3
  └── ...


Each Neuron Device is represented as a directory under ``/sys/devices/virtual/neuron_device/``, where ``neuron0/`` represents Neuron Device 0, ``neuron1/`` represents Neuron Device 1, etc. Each NeuronCore is represented as a directory under a Neuron Device directory, represented as ``neuron_core{0,1,2,...}``. Metrics such as Runtime and Driver info and statistics are collected as per NeuronCore in two directories under the NeuronCore directory, i.e. ``info/`` and ``stats/``.

Most of the metrics belong to a category called “counter.” Each counter is represented as a directory, which holds two numerical values as two files: total and present. The total value starts accumulating metrics when the Driver is loaded; the present value records the last changed metric value. Each counter has the same filesystem structure like this:

.. code-block:: dash

    /sys/devices/virtual/neuron_device/neuron0/neuron_core0/status/
    ├── exec_bad_input/
    │   ├── total
    │   └── present
    ├── hw_error/
    │   ├── total
    │   └── present
    ├── infer_failed_to_queue/
    │   ├── total
    │   └── present
    └── ...



Description for Each Metric
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``info/``: this directory stores hardware information. All of them are not counter types:

* ``notify_delay``: Controls delays between notifications from Neuron Device.  Current settings are on (``0``) or off (``-1``).  Off by default. 
* ``arch_type``: Architecture type of the Neuron Device. Sample architecture types are v1, v2, and v3. You can only read the value but not change it.
* ``instance_type``: Instance type of the Neuron Device. Sample instance types are Inf1, Inf2, and Trn1. You can only read the value but not change it.
* ``device_type``: Neuron Device type. Sample Neuron Device types are Inferentia, Inferentia2, and Trainium1. You can only read the value but not change it.


``stats/``: this directory stores Neuron Runtime and Driver statistics. It contains three subdirectories: ``status/``, ``memory_usage/``, and ``other_info/``.

* ``status/``: this directory stores the number of each return status of API calls. As explained in :ref:`The LIBNRT API Return Codes <nrt_api>`, every API call returns an NRT_STATUS value, which represents the return status of that API call. Our sysfs filesystem stores all ``NRT_STATUS`` as subdirectories under the ``status/`` directory. They all have the counter structure. Thus each ``NRT_STATUS`` subdirectory holds two values (total and present) and records the number of times you receive a certain ``NRT_STATUS``. The following is description for each ``NRT_STATUS`` subdirectory. You should see the description align with what is described in :ref:`The LIBNRT API Return Codes <nrt_api>`.

* ``memory_usage/``: this directory contains memory usage statistics, as per device and per host, all of which are counters. In this directory, total counters refer to the amount of memory you are using now, and present counters refer to the last memory allocation or deallocation amount:

  * ``device_mem/{total, present}``: the amount of memory that Neuron Runtime uses for weights, instructions and DMA rings.
  * ``host_mem/{total, present}``: the amount of memory that Neuron Runtime uses for input and output tensors.

* ``other_info/``: this directory contains statistics that are not included by ``status/`` and ``memory_usage/``. All of them are not counter types:

  * ``inference_count``: number of successful inferences
  * ``model_load_count``:  number of successful model loads
  * ``reset_count``: number of successful device resets


Read and Write to Metrics
^^^^^^^^^^^^^^^^^^^^^^^^^

Reading a sysfs file gives the value for the corresponding metric. You can use the cat command to view the contents of the sysfs files.: 

.. code-block:: bash

  ubuntu@ip-xxx-xx-xx-xxx:~$ sudo cat /sys/devices/virtual/neuron_device/neuron0/neuron_core0/stats/status/failure/total 
  0
  ubuntu@ip-xxx-xx-xx-xxx:~$ sudo cat /sys/devices/virtual/neuron_device/neuron0/neuron_core0/info/architecture/arch_type 
  NCv2

Sysfs metrics of counter type are write to clear. You can write any value to the file, and the metric will be set to 0:

.. code-block:: bash

  ubuntu@ip-xxx-xx-xx-xxx:~$ echo 1 | sudo tee /sys/devices/virtual/neuron_device/neuron0/neuron_core0/stats/status/failure/total 
  1

Note
^^^^

All files under ``/sys/devices/virtual/neuron_device/neuron0/power`` such as ``runtime_active_kids`` or ``runtime_status`` are related to generic device power management. They are not created or controlled by our sysfs metrics. The word ``runtime`` in these files does not refer to Neuron Runtime.

.. _troubleshoot_via_sysfs:
How to Troubleshoot via Sysfs
-----------------------------
You can perform simple and easy tasks to troubleshoot your ML jobs with one or a few CLIs to read or write the sysfs filesystem.
You can do aggregations across all NeuronCores and all Neuron Device to get a summarized view using your scripts. 


You can also use the Sysfs notification feature to wait passively (without wasting CPU cycles) for changes to the values of Sysfs files. To use this feature, you need to implement a user-space program that calls the poll() function on the Sysfs file that you want to wait on. 
The poll() function has the following signature: ``unsigned int (*poll) (struct file *, struct poll_table_struct *)``.
By default, the Sysfs notification feature is turned off when the driver is loaded. To enable notifications, you can set the value of ``/sys/devices/virtual/neuron_device/neuron0/info/notify_delay`` to 0. To disable notifications, you can set it to -1. Please note that enabling this feature can impact performance.

Here is a sample user space program using poll():

.. code-block:: dash

	#include <fcntl.h>
	#include <poll.h>
	#include <unistd.h>
	#include <stdio.h>
	#include <stdlib.h>

	int main(int argc, char * argv[])
	{
		char readbuf[128];
		int attr_fd = -1; 
		struct pollfd pfd;
		int retval = 0;
		ssize_t read_bytes;

		if (argc < 2) {
			fprintf(stderr, "Error: Please specify sysfs file path\n");
			exit(1);
		}   
		attr_fd = open(argv[1], O_RDONLY, 0); 
		if (attr_fd < 0) {
			perror(argv[1]);
			exit(2);
		}   

		read_bytes = read(attr_fd, readbuf, sizeof(readbuf));
		if (read_bytes < 0) {
			perror(argv[1]);
			exit(3);
		}   
		printf("%.*s", (int)read_bytes, readbuf);

		pfd.fd = attr_fd;
		pfd.events = POLLERR | POLLPRI;
		pfd.revents = 0;
		while ((retval = poll(&pfd, 1, 100)) >= 0) {
			if (pfd.revents & (POLLERR | POLLPRI)) {
				pfd.revents = 0;

				lseek(attr_fd, 0, SEEK_SET);
				read_bytes = read(attr_fd, readbuf, sizeof(readbuf));
				if (read_bytes < 0) {
					perror(argv[1]);
					exit(4);
				}
				printf("%.*s", (int)read_bytes, readbuf);
			}
		}
		return 0;
	}


