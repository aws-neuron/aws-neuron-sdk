.. _neuron-profile-ug:

Neuron Profile User Guide
=========================

.. contents:: Table of contents
    :local:
    :depth: 2

Overview
--------

**neuron-profile** is a tool to profile and analyze performance of a ML model compiled with the Neuron compiler
and run on Neuron devices.

.. note::

    Please use the ``aws-neuronx-tools`` package from Neuron SDK 2.11 or higher.


Installation
------------

``neuron-profile`` comes as part of the ``aws-neuronx-tools`` package, and will be installed to ``/opt/aws/neuron/bin``.

The Neuron web profile viewer utilizes InfluxDB OSS 2.x to store time series data for the profiled workloads during postprocessing.
Please follow the instructions provided at https://portal.influxdata.com/downloads/ for the correct OS.  A sample installation
of InfluxDB is provided below.

Ubuntu
~~~~~~

::

    # Neuron
    . /etc/os-release
    sudo tee /etc/apt/sources.list.d/neuron.list > /dev/null <<EOF
    deb https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main
    EOF

    wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | sudo apt-key add -
    sudo apt-get update -y
    sudo apt-get install aws-neuronx-runtime-lib aws-neuronx-dkms -y
    sudo apt-get install aws-neuronx-tools -y

    # InfluxDB
    wget -q https://repos.influxdata.com/influxdata-archive_compat.key
    echo '393e8779c89ac8d958f81f942f9ad7fb82a25e133faddaf92e15b16e6ac9ce4c influxdata-archive_compat.key' | sha256sum -c && cat influxdata-archive_compat.key | gpg --dearmor | sudo tee /etc/apt/trusted.gpg.d/influxdata-archive_compat.gpg > /dev/null
    echo 'deb [signed-by=/etc/apt/trusted.gpg.d/influxdata-archive_compat.gpg] https://repos.influxdata.com/debian stable main' | sudo tee /etc/apt/sources.list.d/influxdata.list

    sudo apt-get update && sudo apt-get install influxdb2 influxdb2-cli -y
    sudo systemctl start influxdb
    influx setup
    # Fill in the information to finish the setup

AL2
~~~

::

    # Neuron
    sudo tee /etc/yum.repos.d/neuron.repo > /dev/null <<EOF
    [neuron]
    name=Neuron YUM Repository
    baseurl=https://yum.repos.neuron.amazonaws.com
    enabled=1
    metadata_expire=0
    EOF

    sudo rpm --import https://yum.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB
    sudo yum install aws-neuronx-runtime-lib aws-neuronx-dkms -y
    sudo yum install aws-neuronx-tools -y

    # InfluxDB
    cat <<EOF | sudo tee /etc/yum.repos.d/influxdata.repo
    [influxdata]
    name = InfluxData Repository - Stable
    baseurl = https://repos.influxdata.com/stable/\$basearch/main
    enabled = 1
    gpgcheck = 1
    gpgkey = https://repos.influxdata.com/influxdata-archive_compat.key
    EOF

    sudo yum install influxdb2 influxdb2-cli -y
    sudo systemctl start influxdb
    influx setup
    # Fill in the information to finish the setup



Capturing a profile
-------------------

The ``neuron-profile`` tool can both capture and post-process profiling information.
In the simplest mode, it takes a compiled model (a NEFF), executes it, and saves the profile results to a NTFF (``profile.ntff`` by default).
For this example, we assume a NEFF is already available as ``file.neff``

::

    $ neuron-profile capture -n file.neff -s profile.ntff

Processing and viewing the profile results
------------------------------------------

The ``view`` subcommand of ``neuron-profile`` will handle post-processing the profiling data and starting up an HTTP server that users can
navigate to in order to see profiling results.

Viewing a single profile
~~~~~~~~~~~~~~~~~~~~~~~~

The first way to invoke ``neuron-profile view`` is to pass both the NEFF and the NTFF to this command.
It will post-process these artifacts and print out a direct link to the profile view.

::

    $ neuron-profile view -n file.neff -s profile.ntff
    View profile at http://0.0.0.0:3001/profile/n_fdc71a0b582ee3009711a96e59958af921243921
    ctrl-c to exit

Viewing multiple profiles
~~~~~~~~~~~~~~~~~~~~~~~~~

Alternatively, when post-processing multiple profiles, it may be desirable to have a persistent server running while processing results in the background.
In this case, we can skip passing arguments to the command, which will direct users to the main page listing all available profiles.

::

    $ neuron-profile view
    View a list of profiles at http://0.0.0.0:3001/

In a separate window, we can kick off the post-processing without launching another server by passing the ``--ingest-only`` flag.

::

    $ neuron-profile view -n file.neff -s profile.ntff --ingest-only
    Profile "n_47cf9972d42798d236caa68952d0d29a76d8bd66" is ready to view

``n_47cf9972d42798d236caa68952d0d29a76d8bd66`` is the bucket where the data is stored.  We can find this profile at ``localhost:3001/profile/<bucket>``.

Accessing the profiles
~~~~~~~~~~~~~~~~~~~~~~

If ``neuron-profile view`` is run on a remote instance, you may need to use port forwarding to access the profiles.

From the local machine, SSH to the remote instance and forward ports 3001 (the default ``neuron-profile`` HTTP server port) and 8086 (the default
influxdb port).  Then in the browser, go to ``localhost:3001`` to view the profiles.

::

    $ ssh <user>@<ip> -L 3001:localhost:3001 -L 8086:localhost:8086


Understanding a Neuron profile
------------------------------

The section provides a quick overview on what features and information are available through the Neuron web profile viewer.

For more information on terms used, please check out the :ref:`neuron_hw_glossary`.

Timeline
~~~~~~~~

|neuron-profile-web-timeline|

The execution timeline is plotted based on the elapsed nanoseconds since the start of execution.

Starting from the bottom, the ``TensorMatrix Utilization`` shows the efficiency of the TensorEngine, and
the ``Pending DMA Count`` and ``DMA Throughput`` rows show the DMA activity.  In general, we want these to be as high
as possible, and in some cases may help give clues as to whether the workload is memory or compute bound.

Next are the individual NeuronCore engine executions.  These rows show the start and end times for instructions executed by each
engine, and clicking on one of these bars will show more detailed information, as well as any dependencies that were found.
For models involving collective compute operations, you will additionally see rows labeled with ``CC-core``, which are used to synchronize
the CC operations.

Towards the top is the DMA activity.  These can include the transfers of input and output tensors, intermediate tensors, and any
additional spilling or loading to and from the on-chip SRAM memory.


Features
~~~~~~~~

The following are some useful features that may help with navigating a profile:

- Dragging your cursor across a portion of the timeline will zoom in to the selected window, providing a more in depth view of the execution during that time period.
- Hovering over a point will reveal a subset of information associated with it.
- Clicking a point will open a text box below the timeline with all the information associated with it.
- Right-clicking a point will drop a marker at a certain location.  This marker will persist when zooming in and out.

  - All marker information can be found by clicking the ``Annotations`` button.
  - Markers can be saved and loaded by using a provided name for the marker set.
  - Individual markers can be renamed or deleted in this menu as well.

- The ``Edit view settings`` can be used to further customize the timeline view.  For example, changing the ``Instruction Grouping`` dropdown option to "Layer" will re-color the timeline based on the associated framework layer name.

Additionally, there are various summary buttons that can be clicked to provide more information on the model/NEFF, such as the input and output tensors,
number of FLOPs, and the start and end of a framework layer.

|neuron-profile-web-summaries|


CLI reference
-------------

.. rubric:: neuron-profile capture

.. program:: neuron-profile

.. option:: neuron-profile capture [parameters] [inputs...]

    Takes a given compiled NEFF, executes it, and collect the profile results.
    When no inputs are provided, all-zero inputs are used, which may result in inf or NaNs.
    It is recommended to use ``--ignore-inference

    - :option:`-n,--neff` (string): the compiled NEFF to profile

    - :option:`-s,--session-file` (string): the file to store profile session information in

    - :option:`--ignore-exec-errors`: ignore errors during execution

    - :option:`inputs` (positional args): List of inputs in the form of <NAME> <FILE_PATH> separated by space. Eg IN1 x.npy IN2 y.npy


.. option:: neuron-profile view [parameters]

    - :option:`-n,--neff-path` (string): the compiled NEFF file location

    - :option:`-s,--session-file` (string): the profile results NTFF file location

    - :option:`--db-endpoint` (string): the endpoint of InfluxDB (default: ``http://localhost:8086``)

    - :option:`--db-org` (string): the org name of InfluxDB

    - :option:`--port` (int): the port number of the http server (default: 3001)

    - :option:`--force`: force overwrite an existing profile in the database


Troubleshooting
---------------

InfluxDB not installed
~~~~~~~~~~~~~~~~~~~~~~

::

    $ neuron-profile view -n file.neff -s profile.ntff
    ERRO[0001] To install influxdb, go to https://portal.influxdata.com/downloads/ and follow the instructions there
    influxdb not setup correctly: exec: "influx": executable file not found in $PATH

::

    $ neuron-profile view -n file.neff -s profile.ntff
    ERRO[0000]                                              
    influxdb token not setup correctly: exit status 1
    Try executing "systemctl start influxdb" and "influx setup"

Running ``neuron-profile view`` without InfluxDB installed will result in an error and a pointer to the InfluxDB installation instructions.
Please follow the provided instructions and retry.

Too many open files
~~~~~~~~~~~~~~~~~~~

::

    influxdb2client E! Write error: internal error: unexpected error writing points to database: [shard 10677] open /home/ubuntu/.influxdbv2/engine/data/7caae65aaa48380d/autogen/10677/index/0/MANIFEST: too many open files

InfluxDB will encounter "too many open files" and out of memory errors after a few hundred buckets have been created.
Two ways to solve this are to delete unused buckets or increase the system file descriptor limit.

To increase the file descriptor limit, add the following lines to ``/etc/security/limits.d/efa.conf`` and ``/etc/security/limits.conf``:

::

    *               soft    nofile      1048576
    *               hard    nofile      1048576

Add the following lines to /etc/sysctl.conf

::

    fs.file-max = 197341270
    vm.max_map_count=1048576

Commit changes by running ``sudo sysctl -p``.


.. |neuron-profile-web-timeline| image:: /images/neuron-profile-web-timeline_2-11.png
.. |neuron-profile-web-summaries| image:: /images/neuron-profile-web-summaries_2-11.png
