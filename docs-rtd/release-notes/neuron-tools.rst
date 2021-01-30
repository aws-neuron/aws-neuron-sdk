Neuron Tools Release Notes
^^^^^^^^^^^^^^^^^^^^^^^^^^

This documents lists the release notes for AWS Neuron tools. Neuron
tools are used for debugging, profiling and gathering inferentia system
information.

Known Issues and Limitations 01/30/2020
=======================================

-  neuron-top has a visible screen stutter as the number of loaded
   models increases above 40. This is only a visual issue with no impact
   on performance. The issue is caused by the re rendering the UI on
   screen refresh. We will fix this in a future release.

.. _1420:

[1.4.2.0]
=========

Date: 01/30/2020

Major New Features
------------------

-  Adds **neuron-htop** as a new front-end over neuron-monitor.  New tool is an improved interface over neuron-top with an intention of eventually replacing the older version of neuron-top in the future.  Try it out and provide any feedback/issues on GitHub.


Improvements
------------

-  Extended support for uncompressed neffs to all tools.

Resolved Issues
---------------

-  Runtime memory usage error was not clearing in neuron-monitor.


.. _1310:

[1.3.1.0]
=========

Date: 12/23/2020

Improvements
------------

-  Minor internal enhancement to **neuron-monitor** to help track system resources used by neuron-monitor.
 

.. _1270:

[1.2.7.0]
=========

Date: 11/17/2020

Major New Features
------------------

-  **neuron-monitor** now provides system-wide memory usage statistics.
   Many JSON field names have been updated. We've added a new sample
   script which exports most of **neuron-monitor**'s metrics to a
   `Prometheus <https://prometheus.io/>`__ monitoring server.
   Additionally, we also provided a :neuron-monitor-src:`sample Grafana
   dashboard <neuron-monitor-grafana.json>` - in
   JSON format - which can be imported to a
   `Grafana <https://grafana.com/>`__ instance via its `web
   interface <https://grafana.com/docs/grafana/latest/dashboards/export-import/#importing-a-dashboard>`__.
   This dashboard can then present the metric data made available to
   Prometheus by **neuron-monitor**. More details on how to use
   **neuron-monitor** with this new feature can be found in the :ref:`neuron-monitor-ug`.

-  Neuron tools updated the NeuronCore utilization metric to include all
   inf1 compute engines and DMAs. The new metric definition is more
   comprehensive and provides a better representation of execution
   efficiency.

Resolved Issues
---------------

-  Fixed a memory leak in **neuron-monitor** when attempting to connect
   to the GRPC address of a Neuron Runtime which is not running.

.. _112280:

[1.1.228.0]
===========

Date: 10/22/2020

.. _major-new-features-1:

Major New Features
------------------

-  n/a

Improvements
------------

-  All the tools now use nd0:nc0 to identify NeuronDevice and NeuronCore
   instead of bdf.
-  ``neuron-cli list-model`` now shows NCG Id for each loaded model.
-  ``neuron-top`` columns are reordered to show usage details first.
-  ``neuron-top`` shows weights in human readable format(MB, GB).

.. _resolved-issues-1:

Resolved Issues
---------------

-  ``neuron-top`` now correctly shows NC usage if multiple models are
   loaded onto the same NC.



.. _10110540:

[1.0.11054.0]
=============

Date: 09/22/2020

Major New Features
------------------

Beta release of **neuron-monitor** for streaming metric information
about inference execution from your inf1. We provided a sample script
for connecting neuron-monitor output directly into CloudWatch. Usage of
the new tool is a simple one-liner:

::

   neuron-monitor | neuron-monitor-cloudwatch.py --namespace neuron_monitor_test --region us-west-2

More details on how to use **neuron-monitor** can be found in the :ref:`neuron-monitor-ug`.

Improvements
------------

-  neuron-ls now shows connected devices as a list. This information can
   be used when creating a neuron core group.

Resolved Issues
---------------

-  n/a

.. _10106160:

[1.0.10616.0]
=============

Date: 08/19/2020

.. _major-new-features-1:

Major New Features
------------------

-  n/a

.. _improvements-1:

Improvements
------------

-  Various minor improvements.

.. _resolved-issues-1:

Resolved Issues
---------------

-  n/a

.. _10102720:

[1.0.10272.0]
=============

Date: 08/08/2020

.. _major-new-features-2:

Major New Features
------------------

-  n/a

.. _improvements-2:

Improvements
------------

-  Various minor improvements.

.. _resolved-issues-2:

Resolved Issues
---------------

-  n/a

.. _10101820:

[1.0.10182.0]
=============

Date: 08/05/2020

.. _major-new-features-3:

Major New Features
------------------

-  n/a

.. _improvements-3:

Improvements
------------

-  Various minor improvements.

.. _resolved-issues-3:

Resolved Issues
---------------

-  n/a

.. _1097000:

[1.0.9700.0]
============

Date: 07/16/2020

.. _major-new-features-4:

Major New Features
------------------

-  n/a

.. _improvements-4:

Improvements
------------

-  neuron-ls now supports JSON output format through a new command line
   option --json-output.

.. _resolved-issues-4:

Resolved Issues
---------------

-  n/a

.. _1090430:

[1.0.9043.0]
============

Date: 06/11/2020

Summary
-------

-  Enhancements to neuron-cli to improve loading of large models
-  Fix aws-neuron-runtime-base uninstall to cleanup all the relevant
   files
-  Migrated neuron-discovery service to use IMDSv2 to query instance
   type

.. _major-new-features-5:

Major New Features
------------------

-  Added new commandline options to **neuron-cli** to improve the
   performance on loading large models

   .. rubric:: --ncg-id <value>
      :name: --ncg-id-value

   Legal values for ncg-id:

   -  "-1": runtime will create the NCG (default)

   -  "0": NCG will be created by neuron-cli

   -  ">=1": Model will be loaded to the NCG id specified

   During model load, neuron-cli parses the NEFF file for parameters
   needed to create an NCG. The runtime will parse the same NEFF file a
   second time during the load. Allowing the runtime to create the NCG
   reduces load time by skipping the redundant parse in neuron-cli.

   .. rubric:: --enable-direct-file-load
      :name: --enable-direct-file-load

   By default, neuron-cli loads models into its own memory and streams
   the model to the Neuron Runtime using GRPC. When the
   '--enable-direct-file-load' flag is passed, the load operation will
   skip the copy and only pass the filepath of the model to the Neuron
   Runtime. This saves time and memory during model loads.

.. _resolved-issues-5:

Resolved Issues
---------------

-  None

.. _1085500:

[1.0.8550.0]
============

Date: 5/15/2020

.. _summary-1:

Summary
-------

-  Point fix for installation and startup errors of neuron-discovery
   service in the aws-neuron-runtime-base package.

Please update to aws-neuron-runtime-base package version 1.0.7173 or
newer:

::

   # Ubuntu 18 or 16:
   sudo apt-get update
   sudo apt-get install aws-neuron-runtime-base

   # Amazon Linux, Centos, RHEL
   sudo yum update
   sudo yum install aws-neuron-runtime-base

.. _major-new-features-6:

Major New Features
------------------

-  None

.. _resolved-issues-6:

Resolved Issues
---------------

-  Installation of aws-neuron-runtime-base version 1.0.7044 fails to
   successfully move service files into the service folder. Release of
   aws-neuron-runtime-base version 1.0.7173 fixes this installation
   issue.

-  Added a dependency on the networking service in the neuron-discovery
   service to avoid potential for discovery to start before networking.
   If networking starts first, neuron-discovery will fail to start.

.. _1081310:

[1.0.8131.0]
============

Date: 5/11/2020

.. _summary-2:

Summary
-------

.. _major-new-features-7:

Major New Features
------------------

-  All tools now support use of an environment variable
   (NEURON_RTD_ADDRESS) to specify the runtime address or by explicitly
   specifying the address with the -a flag. Not specifying an address
   will continue to rely on default address set during installation.
-  When run as root, neuron-ls output will now include runtime details
   (address, pid, and version).

::

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

.. _resolved-issues-7:

Resolved Issues
---------------

-  Backwards compatibility of neuron-top with older versions of Neuron
   Runtime is now restored.

Known Issues and Limitations
----------------------------

-  neuron-top has a visible screen stutter as the number of loaded
   models increases above 40. This is only a visual issue with no impact
   on performance. The issue is caused by the re rendering the UI on
   screen refresh. We will fix this in a future release.

.. _1065540:

[1.0.6554.0]
============

Date: 3/26/2020

.. _summary-3:

Summary
-------

Fixed the issue where neuron-top was negatively impacting inference
throughput.

.. _major-new-features-8:

Major New Features
------------------

N/A

.. _resolved-issues-8:

Resolved Issues
---------------

-  neuron-top no longer has a measurable impact on inference throughput
   regardless of instance size.

   -  This version of neuron-top requires Neuron Runtime version
      1.0.6222.0 or newer. Backwards compatibility will be fixed in the
      next release.

-  neuron-top now correctly shows when a model is unloaded.

.. _known-issues-and-limitations-1:

Known Issues and Limitations
----------------------------

-  neuron-top has a visible screen stutter as the number of loaded
   models increases above 40. This is only a visual issue with no impact
   on performance. The issue is caused by the re rendering the UI on
   screen refresh. We will fix this in a future release.

.. _1058320:

[1.0.5832.0]
============

Date: 2/27/2020

.. _summary-4:

Summary
-------

Improved neuron-cli output to display device placement information about
each model.

.. _major-new-features-9:

Major New Features
------------------

N/A

.. _resolved-issues-9:

Resolved Issues
---------------

N/A

.. _known-issues-and-limitations-2:

Known Issues and Limitations
----------------------------

-  neuron-top consumes one vCPU to monitor hardware resources, which
   might affect performance of the system on inf1.xlarge. Using a larger
   instance size will not have the same limitation. In a future release
   we will improve this for smaller instance sizes.

.. _1051650:

[1.0.5165.0]
============

Date: 1/27/2020

.. _summary-5:

Summary
-------

Improved neuron-top load time, especially when a large amount of models
are loaded.

.. _major-new-features-10:

Major New Features
------------------

N/A

.. _resolved-issues-10:

Resolved Issues
---------------

N/A

.. _known-issues-and-limitations-3:

Known Issues and Limitations
----------------------------

-  neuron-top consumes one vCPU to monitor hardware resources, which
   might affect performance of the system on inf1.xlarge. Using a larger
   instance size will not have the same limitation. In a future release
   we will improve this for smaller instance sizes.

Other Notes
-----------

.. _1045870:

[1.0.4587.0]
============

Date: 12/20/2019

.. _summary-6:

Summary
-------

Minor bug fixes to neuron-top and neuron-ls.

.. _major-new-features-11:

Major New Features
------------------

.. _resolved-issues-11:

Resolved Issues
---------------

-  neuron-top: now shows model name and uuid to help distinguish which
   model is consuming resources. Previously only showed model id.
-  neuron-ls: lists device memory size correctly in MB

.. _known-issues-and-limitations-4:

Known Issues and Limitations
----------------------------

.. _other-notes-1:

Other Notes
-----------

.. _1042500:

[1.0.4250.0]
============

Date: 12/1/2019

.. _summary-7:

Summary
-------

.. _major-new-features-12:

Major New Features
------------------

.. _resolved-issues-12:

Resolved Issues
---------------

-  neuron-top may take longer to start and refresh when numerous models
   are loaded
-  neuron-top may crash when trying to calculate the utilization of the
   devices

.. _known-issues-and-limitations-5:

Known Issues and Limitations
----------------------------

.. _other-notes-2:

Other Notes
-----------

.. _1036570:

[1.0.3657.0]
============

Date: 11/25/2019

.. _major-new-features-13:

Major New Features
------------------

N/A, this is the first release.

.. _resolved-issues-13:

Resolved Issues
---------------

N/A, this is the first release.

Known Issues and Limits
-----------------------

-  neuron-top may take longer to start and refresh when numerous models
   are loaded.

   -  Workaround: Unload the models not in use before using neuron-top

-  neuron-top may crash when trying to calculate the utilization of the
   devices.

.. _other-notes-3:

Other Notes
-----------
