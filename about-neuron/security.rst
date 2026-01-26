.. meta::
    :description: Security disclosures and notification for the AWS Neuron SDK.
    :date-modified: 01/27/2026

.. _security:

Neuron Security Disclosures
===========================

If you think you've found a potential security issue, please do not post it in the Issues. Instead, please follow the instructions here
(https://aws.amazon.com/security/vulnerability-reporting/) or email AWS
security directly (`mailto:aws-security@amazon.com <mailto:aws-security@amazon.com>`__).

Important Security Information for Trainium Hardware
-----------------------------------------------------

Trainium hardware is designed to optimize performance for machine learning workloads. To deliver high performance, applications with access to Trainium devices have unrestricted access to instance physical memory.

What this means for your deployment:

* Instance-level isolation is maintained: AWS EC2 ensures Trainium devices cannot access physical memory of other EC2 instances.
* As a best practice to prevent unrestricted access to host physical memory by any user/application, we recommend implementing a permission model where:

   * A dedicated system group owns the device nodes
   * Only explicitly authorized user are added to this group
   * Device permissions prevent access by users outside the group
  
Customer responsibility: Ensure that only trusted applications have access to Tranium devices on Trainium instances. For more information, see `the AWS Shared Responsibility Model <https://aws.amazon.com/compliance/shared-responsibility-model/>`__.

Example Implementation Steps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The steps below are an example you can follow to implement a security group using udev rules:

1. Create a dedicated security group (in this example, ``neuron``): ``sudo groupadd -r neuron``

2. Add authorized users to that security group: ``sudo usermod -aG neuron {username-to-add-here}``, repeat for each user

3. Configure udev rules. Create a udev rule to automatically set correct ownership and permissions when Trainium (neuron) devices are detected.

   Create the file ``/etc/udev/rules.d/neuron-udev.rules`` with the following content:
    
   .. code-block:: shell

      # Neuron device access control
      # Only members of the 'neuron' group can access 'neuron' devices.

      SUBSYSTEM=="neuron*", KERNEL=="neuron*", GROUP="neuron", MODE="0660"

4. Apply the configuration:

   ``sudo udevadm control —-reload``
   ``sudo udevadm trigger —-subsystem-match=neuron``

5. Verify the configuration:

    ``ls -l /dev/neuron*``

    Expected output:

    ``crw-rw---- 1 root neuron 239, 0 Jan 9 15:58 /dev/neuron0``

