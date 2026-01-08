.. meta::
   :description: Setup and get started guide for new Neuron SDK profiler
   :date_updated: 12/02/2025

.. _new-neuron-profiler-setup:

Get Started with Neuron Explorer
========================================

Neuron Explorer is a kernel profiling tool included with the AWS Neuron SDK that provides detailed performance insights for machine learning workloads running on AWS Trainium and Inferentia instances. This guide walks you through how to set up and launch Neuron Explorer, including the web-based UI for interactive analysis. By the end of this guide, you'll be able to visualize and analyze performance data for your models directly in your browser.

Overview
---------

In this guide, you'll launch an AWS Trainium or Inferentia EC2 instance using the AWS Deep Learning AMI (DLAMI) for Neuron, install and verify Neuron Explorer, start both the API and UI servers, and set up secure SSH tunneling to view the Neuron Explorer interface in your local browser.

Use this tool when you want to collect, inspect, and visualize Neuron profiling data from model training or inference jobs running on Neuron-compatible instances. At a high level, you will:

1. Launch a Neuron DLAMI instance
2. Verify Neuron Explorer installation
3. Start the Neuron Explorer servers
4. Configure SSH tunneling
5. Access the Neuron Explorer UI locally


Prerequisites
--------------

* An AWS account with permissions to launch EC2 instances.
* Access to an AWS Trainium or Inferentia instance type (such as trn1.2xlarge, inf2.xlarge).
* AWS Neuron DLAMI with the latest Neuron SDK preinstalled.
* SSH key pair (``.pem`` file) to securely connect to your EC2 instance.
* Local machine with SSH client and web browser installed.


Before you begin
-----------------

Complete these steps before starting the task in this document:

1. Make sure you have an active AWS account and `a default VPC available in your region <https://console.aws.amazon.com/vpc/>`_. 
2. Create or locate your SSH key pair (``.pem`` file) that allows access to your EC2 instance.

Instructions
-------------

1. Launch a Neuron-compatible EC2 instance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Launch an EC2 instance with either a Trainium or Inferentia instance type using the AWS Neuron DLAMI.
You can do this from the AWS Management Console or CLI. For more instructions on how to launch an instance with Neuron DLAMI, refer to the instructions here.

**Expected outcome**

Your instance should start and appear in the EC2 dashboard as "Running."


2. Verify that Neuron Explorer is installed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once you've connected to your EC2 instance with SSH, verify that Neuron Explorer and the associated tools are installed:

.. code-block:: bash

   apt list --installed | grep neuronx-tools

**Expected outcome**

You should see neuronx-tools listed among the installed packages, confirming that Neuron Explorer is available on your instance.


3. Launch the API and UI SPA servers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Start the Neuron Explorer web servers using the following command:

.. code-block:: bash

   neuron-explorer view -v 2 --data-path ./parquet_files --ui-mode latest

This command starts:

* The UI SPA (Single Page Application) server (default port: 3001)
* The API server (default port: 3002)


**Expected outcome**

You'll see terminal logs confirming that both the UI and API servers are running.

.. note::

**Command Update:** 

The command has been renamed from ``neuron-profiler`` to ``neuron-explorer``. If you're using an older version of the library, use ``neuron-profiler view`` instead.


4. Set up SSH tunneling
^^^^^^^^^^^^^^^^^^^^^^^^

By default, Neuron Explorer runs locally on the EC2 instance. To securely access it from your local computer, you must create SSH tunnels for ports 3001 and 3002.

Run the following command from your local machine terminal (replace placeholders such as ``your-key`` and ``public_ip_address_of_your_instance``):

.. code-block:: bash

   ssh -i ~/your-key.pem -L 3001:localhost:3001 -L 3002:localhost:3002 ubuntu@[public_ip_address_of_your_instance_] -fN

**Explanation:**

* ``-L 3001:localhost:3001`` forwards the UI server.
* ``-L 3002:localhost:3002`` forwards the API server.
* ``-fN`` keeps the tunnel open in the background.


**Expected outcome**

No error messages should appear, indicating that your SSH tunnels are active.

.. note::
   Replace ``ubuntu`` with the appropriate username for your AMI (for example, ``ec2-user`` on Amazon Linux).

5. Connect to the Neuron Explorer UI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once your tunnel is active, open your preferred web browser and navigate to:

.. code-block:: text

   http://localhost:3001


**Expected outcome**

The Neuron Explorer UI loads in your browser, displaying an interactive dashboard for exploring profiling data.

Confirm your work
------------------

You've successfully set up Neuron Explorer! To confirm everything is working:

1. The browser should display the Neuron Explorer interface.
2. The terminal running the profiler command should show log activity when interacting with the UI.
3. You can explore profiling sessions from your ``./parquet_files`` directory.

If all these checks pass, you are ready to begin analyzing performance data using Neuron Explorer.

Common issues
---------------

If you encounter an error or other issue while working through this task, here are some commonly encountered issues and how to address them:

* **Neuron Explorer UI doesn't load**: Check that your SSH tunnel is configured correctly. Make sure ports 3001 and 3002 are forwarded using the ``-L`` flags in your SSH command, and verify the EC2 instance is running.
* **No profiling data displayed**: Double-check that the directory passed to ``--data-path`` contains valid .parquet profiling files generated by a prior Neuron profiling run.
* **neuron-explorer command not found**: Ensure that Neuron SDK is installed. Please ensure that you have launched your instance with Neuron DLAMI or you have set up your instance based on the instructions mentioned here.
* **Connection refused on port 3001 or 3002**: Confirm that your EC2 security group allows outbound traffic and that the SSH tunnel was created from your local machine, not from inside the instance.
