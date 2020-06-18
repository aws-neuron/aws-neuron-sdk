# Neuron Tools release notes

This documents lists the release notes for AWS Neuron tools. Neuron tools are used for debugging, profiling and gathering inferentia system information.

# Known Issues and Limitations 06/18/2020

* neuron-top has a visible screen stutter as the number of loaded models increases above 40. This is only a visual issue with no impact on performance. The issue is caused by the re rendering the UI on screen refresh. We will fix this in a future release.

# [1.0.9171.0]

Date: 6/18/2020

## Major New Features

* n/a

## Improvements

* n/a

## Resolved Issues

*  In the earlier version yum downgrade aws-neuron-runtime-base removed the neuron-discovery service unit files. This results in neuron-discovery start failures. Please upgrade aws-neuron-runtime-base to this version if using yum.


# [1.0.9043.0]

Date: 06/11/2020

## Summary
* Enhancements to neuron-cli to improve loading of large models
* Fix aws-neuron-runtime-base uninstall to cleanup all the relevant files
* Migrated neuron-discovery service to use IMDSv2 to query instance type

## Major New Features
* Added new commandline options to **neuron-cli** to improve the performance on loading large models
    #### --ncg-id \<value>
    
    Legal values for ncg-id:

    * "-1": runtime will create the NCG (default)
    * "0": NCG will be created by neuron-cli
    * ">=1": Model will be loaded to the NCG id specified

    
    
    During model load, neuron-cli parses the NEFF file for parameters needed to create an NCG. The runtime will parse the same NEFF file a second time during the load.  Allowing the runtime to create the NCG reduces load time by skipping the redundant parse in neuron-cli.
    
    
    
    #### --enable-direct-file-load
    By default, neuron-cli loads models into its own memory and streams the model to the Neuron Runtime using GRPC.  When the '--enable-direct-file-load' flag is passed, the load operation will skip the copy and only pass the filepath of the model to the Neuron Runtime.  This saves time and memory during model loads.


## Resolved Issues
* None


# [1.0.8550.0]

Date: 5/15/2020

## Summary
* Point fix for installation and startup errors of neuron-discovery service in the aws-neuron-runtime-base package.

Please update to aws-neuron-runtime-base package version 1.0.7173 or newer:
```
# Ubuntu 18 or 16:
sudo apt-get update
sudo apt-get install aws-neuron-runtime-base

# Amazon Linux, Centos, RHEL
sudo yum update
sudo yum install aws-neuron-runtime-base
```

## Major New Features
* None

## Resolved Issues
* Installation of aws-neuron-runtime-base version 1.0.7044 fails to successfully move service files into the service folder.  Release of aws-neuron-runtime-base version 1.0.7173 fixes this installation issue.
* Added a dependency on the networking service in the neuron-discovery service to avoid potential for discovery to start before networking.  If networking starts first, neuron-discovery will fail to start.  


# [1.0.8131.0]

Date: 5/11/2020

## Summary

## Major New Features

* All tools now support use of an environment variable (NEURON_RTD_ADDRESS) to specify the runtime address or by explicitly specifying the address with the -a flag.  Not specifying an address will continue to rely on default address set during installation.  
* When run as root, neuron-ls output will now include runtime details (address, pid, and version).
```
$ sudo neuron-ls
+--------------+---------+--------+-----------+-----------+------+------+-----------------------+---------+---------+
|   PCI BDF    | LOGICAL | NEURON |  MEMORY   |  MEMORY   | EAST | WEST |        RUNTIME        | RUNTIME | RUNTIME |
|              |   ID    | CORES  | CHANNEL 0 | CHANNEL 1 |      |      |        ADDRESS        |   PID   | VERSION |
+--------------+---------+--------+-----------+-----------+------+------+-----------------------+---------+---------+
| 0000:00:1c.0 |       0 |      4 | 4096 MB   | 4096 MB   |    1 |    0 | unix:/run/neuron.sock |    8871 | 1.0.x.x |
| 0000:00:1d.0 |       1 |      4 | 4096 MB   | 4096 MB   |    1 |    1 | unix:/run/neuron.sock |    8871 | 1.0.x.x |
| 0000:00:1e.0 |       2 |      4 | 4096 MB   | 4096 MB   |    1 |    1 | unix:/run/neuron.sock |    8871 | 1.0.x.x |
| 0000:00:1f.0 |       3 |      4 | 4096 MB   | 4096 MB   |    0 |    1 | unix:/run/neuron.sock |    8871 | 1.0.x.x |
+--------------+---------+--------+-----------+-----------+------+------+-----------------------+---------+---------+
```

## Resolved Issues
* Backwards compatibility of neuron-top with older versions of Neuron Runtime is now restored.

## Known Issues and Limitations
* neuron-top has a visible screen stutter as the number of loaded models increases above 40. This is only a visual issue with no impact on performance. The issue is caused by the re rendering the UI on screen refresh. We will fix this in a future release.

# [1.0.6554.0]

Date: 3/26/2020

## Summary

Fixed the issue where neuron-top was negatively impacting inference throughput.

## Major New Features

N/A

## Resolved Issues

* neuron-top no longer has a measurable impact on inference throughput regardless of instance size.
  * This version of neuron-top requires Neuron Runtime version 1.0.6222.0 or newer. Backwards compatibility will be fixed in the next release.
* neuron-top now correctly shows when a model is unloaded.

## Known Issues and Limitations

* neuron-top has a visible screen stutter as the number of loaded models increases above 40. This is only a visual issue with no impact on performance. The issue is caused by the re rendering the UI on screen refresh. We will fix this in a future release.


# [1.0.5832.0]

Date: 2/27/2020

## Summary

Improved neuron-cli output to display device placement information about each model.

## Major New Features

N/A

## Resolved Issues

N/A

## Known Issues and Limitations

* neuron-top consumes one vCPU to monitor hardware resources, which might affect performance of the system on inf1.xlarge.  Using a larger instance size will not have the same limitation.  In a future release we will improve this for smaller instance sizes.


# [1.0.5165.0]

Date: 1/27/2020

## Summary

Improved neuron-top load time, especially when a large amount of models are loaded.

## Major New Features

N/A

## Resolved Issues

N/A

## Known Issues and Limitations

* neuron-top consumes one vCPU to monitor hardware resources, which might affect performance of the system on inf1.xlarge.  Using a larger instance size will not have the same limitation.  In a future release we will improve this for smaller instance sizes.

## Other Notes



# [1.0.4587.0]

Date: 12/20/2019

## Summary

Minor bug fixes to neuron-top and neuron-ls.

## Major New Features

## Resolved Issues

* neuron-top: now shows model name and uuid to help distinguish which model is consuming resources.  Previously only showed model id.
* neuron-ls: lists device memory size correctly in MB

## Known Issues and Limitations

## Other Notes


# [1.0.4250.0]

Date:  12/1/2019

## Summary

## Major New Features

## Resolved Issues

* neuron-top may take longer to start and refresh when numerous models are loaded
* neuron-top may crash when trying to calculate the utilization of the devices

## Known Issues and Limitations

## Other Notes

# [1.0.3657.0]

Date:  11/25/2019


## Major New Features

N/A, this is the first release.

## Resolved Issues

N/A, this is the first release.

## Known Issues and Limits

* neuron-top may take longer to start and refresh when numerous models are loaded. 
    * Workaround: Unload the models not in use before using neuron-top
* neuron-top may crash when trying to calculate the utilization of the devices. 

## Other Notes


