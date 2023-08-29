.. _setup-trn1-multi-node-execution:

How to prepare trn1.32xlarge for multi-node execution
=====================================================

EFA is a low latency transport that is used for inter-node communication.  Multi-node jobs, such as distributed training, requires EFA to be enabled on every participating trn1/trn1n 32xlarge instance. Please note that EFA is currently not available on the smaller instances sizes and they cannot be used for running multi-node jobs.

trn1.32xlarge has 8 EFA devices, trn1n.32xlarge has 16 EFA devices.  The rest of the document will refer to trn1.32xlarge but everything in the document also applies to trn1n.32xlarge except for the different number of EFA devices.


Launching an instance
^^^^^^^^^^^^^^^^^^^^^

Before launching trn1 you need to create a security group that allows EFA traffic between the instances.  Follow Step1 here: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start.html#efa-start-security and note the newly created security group ID.  It will be used on the next step.

Determine the region, the AMI, the key and the subnet that will be used to launch trn1.

At the moment launching Trn1 instances with EFA support from the console is not recommended. The instances must be launched using AWS CLI.  To launch trn1.32xlarge instance:


.. code-block:: bash

    export AMI=<ami>
    export SUBNET=<subnet id>
    export SG=<security group created on the previous step>
    export REG=<AWS region>
    export KEY=<the key>

    aws ec2 run-instances --region ${REG} \
    --image-id ${AMI} --instance-type trn1.32xlarge \
    --key-name ${KEY} \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=\"friendly name\"}]" \
    --network-interfaces \
    "NetworkCardIndex=0,DeviceIndex=0,Groups=${SG},SubnetId=${SUBNET},InterfaceType=efa" \
    "NetworkCardIndex=1,DeviceIndex=1,Groups=${SG},SubnetId=${SUBNET},InterfaceType=efa" \
    "NetworkCardIndex=2,DeviceIndex=1,Groups=${SG},SubnetId=${SUBNET},InterfaceType=efa" \
    "NetworkCardIndex=3,DeviceIndex=1,Groups=${SG},SubnetId=${SUBNET},InterfaceType=efa" \
    "NetworkCardIndex=4,DeviceIndex=1,Groups=${SG},SubnetId=${SUBNET},InterfaceType=efa" \
    "NetworkCardIndex=5,DeviceIndex=1,Groups=${SG},SubnetId=${SUBNET},InterfaceType=efa" \
    "NetworkCardIndex=6,DeviceIndex=1,Groups=${SG},SubnetId=${SUBNET},InterfaceType=efa" \
    "NetworkCardIndex=7,DeviceIndex=1,Groups=${SG},SubnetId=${SUBNET},InterfaceType=efa" 



Note that one of the cards is assigned DeviceIndex 0 and the rest are assigned DeviceIndex 1.  Cloud-init will configure instance routing to route outgoing traffic prioritized by the device index field.  I.e the outbound traffic will always egress from the interface with DeviceIndex 0.  That avoids network connectivity problems when multiple interfaces are attached to the same subnet.

To launch trn1n.32xlarge instance:

.. code-block:: bash

    export AMI=<ami>
    export SUBNET=<subnet id>
    export SG=<security group created on the previous step>
    export REG=<AWS region>
    export KEY=<the key>
    
    aws ec2 run-instances --region ${REG} \
    --image-id ${AMI} --instance-type trn1.32xlarge \
    --key-name ${KEY} \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=\"friendly name\"}]" \
    --network-interfaces \
        NetworkCardIndex=0,DeviceIndex=0,Groups=$SG,SubnetId=$SUBNET,InterfaceType=efa \
        NetworkCardIndex=1,DeviceIndex=1,Groups=$SG,SubnetId=$SUBNET,InterfaceType=efa \
        NetworkCardIndex=2,DeviceIndex=2,Groups=$SG,SubnetId=$SUBNET,InterfaceType=efa \
        NetworkCardIndex=3,DeviceIndex=3,Groups=$SG,SubnetId=$SUBNET,InterfaceType=efa \
        NetworkCardIndex=4,DeviceIndex=1,Groups=$SG,SubnetId=$SUBNET,InterfaceType=efa \
        NetworkCardIndex=5,DeviceIndex=1,Groups=$SG,SubnetId=$SUBNET,InterfaceType=efa \
        NetworkCardIndex=6,DeviceIndex=1,Groups=$SG,SubnetId=$SUBNET,InterfaceType=efa \
        NetworkCardIndex=7,DeviceIndex=1,Groups=$SG,SubnetId=$SUBNET,InterfaceType=efa \
        NetworkCardIndex=8,DeviceIndex=1,Groups=$SG,SubnetId=$SUBNET,InterfaceType=efa \
        NetworkCardIndex=9,DeviceIndex=1,Groups=$SG,SubnetId=$SUBNET,InterfaceType=efa \
        NetworkCardIndex=10,DeviceIndex=1,Groups=$SG,SubnetId=$SUBNET,InterfaceType=efa \
        NetworkCardIndex=11,DeviceIndex=1,Groups=$SG,SubnetId=$SUBNET,InterfaceType=efa \
        NetworkCardIndex=12,DeviceIndex=1,Groups=$SG,SubnetId=$SUBNET,InterfaceType=efa \
        NetworkCardIndex=13,DeviceIndex=1,Groups=$SG,SubnetId=$SUBNET,InterfaceType=efa \
        NetworkCardIndex=14,DeviceIndex=1,Groups=$SG,SubnetId=$SUBNET,InterfaceType=efa \
        NetworkCardIndex=15,DeviceIndex=1,Groups=$SG,SubnetId=$SUBNET,InterfaceType=efa

Assigning public IP address
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Multi-interface instances are not assigned public IP automatically.  If you require access to the newly launched trn1 from the Internet you need to assign Elastic IP to the interface with DeviceIndex = 0.  To find the right interface either parse the output of the instance launch command or use describe-instances command:


.. code-block:: bash

    $ aws ec2 describe-instances --instance-ids i-01b17afa1e6021d6c
    {
        "Reservations": [
            {
                "Groups": [],
                "Instances": [
                    {
                        "AmiLaunchIndex": 0,
                        "ImageId": "ami-01257e71ecb2f431c",
                        "InstanceId": "i-01b17afa1e6021d6c",
                        "InstanceType": "trn1.32xlarge",
                        .........
                        "NetworkInterfaces": [
                            {
                                "Attachment": {
                                    "AttachTime": "2023-05-19T17:37:26.000Z",
                                    "AttachmentId": "eni-attach-03730388baedd4b96",
                                    "DeleteOnTermination": true,
                                    "DeviceIndex": 0,
                                    "Status": "attached",
                                    "NetworkCardIndex": 4
                                },
                                "Description": "",
                                .........
                                "InterfaceType": "efa"
                            },
                            {
                                "Attachment": {
                                    "AttachTime": "2023-05-19T17:37:26.000Z",
                                    "AttachmentId": "eni-attach-0e1242371cd2532df",
                                    "DeleteOnTermination": true,
                                    "DeviceIndex": 0,
                                    "Status": "attached",
                                    "NetworkCardIndex": 3
                                },
                                "Description": "",
                                ................
            
            }
        ]
    }



The second entry in “NetworkInterfaces” in this example has “DeviceIndex” 0 and should be used to attach EIP.


Software installation
^^^^^^^^^^^^^^^^^^^^^

The software required for EFA operation is distributed via aws-efa-installer package.  The package is preinstalled on Neuron DLAMI.  If you’d like to install the latest or if you are using your own AMI follow these steps:

.. code-block:: bash

    curl -O https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz 
    wget https://efa-installer.amazonaws.com/aws-efa-installer.key && gpg --import aws-efa-installer.key 
    cat aws-efa-installer.key | gpg --fingerprint 
    wget https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz.sig && gpg --verify ./aws-efa-installer-latest.tar.gz.sig 
    tar -xvf aws-efa-installer-latest.tar.gz 
    cd aws-efa-installer && sudo bash efa_installer.sh --yes 
    cd 
    sudo rm -rf aws-efa-installer-latest.tar.gz aws-efa-installer


Containers
^^^^^^^^^^

aws-efa-installer package must be installed on the instance.  That installs both the efa kernel module and the libraries.  The libraries must be accessible to an application running inside a container.  This can be accomplished by either installing aws-efa-installer package inside the container or by making on the instance library installation path available inside a container.

If installing aws-efa-installer package inside a container pass the flag that disables the kernel module installation:

.. code-block:: bash

    sudo bash efa_installer.sh --yes --skip-kmod


The location of the libraries is distribution specific:

.. code-block:: bash

    /opt/amazon/efa/lib   # Ubuntu
    /opt/amazon/efa/lib64 # AL2


Application execution environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When running an application make sure the following environment variables are set:

.. code-block:: bash

    FI_PROVIDER=efa
    FI_EFA_USE_DEVICE_RDMA=1
    FI_EFA_FORK_SAFE=1  # only required when running on AL2


Appendix - trn1 instance launch example script
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    #!/bin/bash
     
    set -e
 
    # AWS CLI v2 Installation instructions for Linux:
    # curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    # unzip awscliv2.zip
    # sudo ./aws/install
    # $ aws --version
    # aws-cli/2.11.20 Python/3.11.3 Linux/5.15.0-1034-aws exe/x86_64.ubuntu.20 prompt/off
    # Someone with AWS console admin privileges can create an access key ID and secret for this:
    # Configure credentials: aws configure
 
    # Search the AWS AMIs for the most recent "Deep Learning Base Neuron AMI (Ubuntu 20.04) <Latest_Date>"
    # This one is 2023-05-17 - ami-01257e71ecb2f431c
    AMI= ... # the ami
    KEYNAME= ... # your key
    SG= ... # the security group 
    SUBNET= ... # the subnet
    REGION=us-west-2
    
    # Launch instances
    echo "Starting instances..."
    output=$(aws ec2 --region $REGION run-instances \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=_Trainium-Big}]' \
    --count 1 \
    --image-id $AMI \
    --instance-type trn1.32xlarge \
    --key-name $KEYNAME \
    --network-interfaces "NetworkCardIndex=0,DeviceIndex=0,Groups=$SG,SubnetId=$SUBNET,InterfaceType=efa" \
    "NetworkCardIndex=1,DeviceIndex=1,Groups=$SG,SubnetId=$SUBNET,InterfaceType=efa" \
    "NetworkCardIndex=2,DeviceIndex=1,Groups=$SG,SubnetId=$SUBNET,InterfaceType=efa" \
    "NetworkCardIndex=3,DeviceIndex=1,Groups=$SG,SubnetId=$SUBNET,InterfaceType=efa" \
    "NetworkCardIndex=4,DeviceIndex=1,Groups=$SG,SubnetId=$SUBNET,InterfaceType=efa" \
    "NetworkCardIndex=5,DeviceIndex=1,Groups=$SG,SubnetId=$SUBNET,InterfaceType=efa" \
    "NetworkCardIndex=6,DeviceIndex=1,Groups=$SG,SubnetId=$SUBNET,InterfaceType=efa" \
    "NetworkCardIndex=7,DeviceIndex=1,Groups=$SG,SubnetId=$SUBNET,InterfaceType=efa")
    
    
    # Parse the output to get the instance IDs
    instance_ids=$(echo $output | jq -r .Instances[].InstanceId)
    echo "Got created instance IDs: $instance_ids"
 
    # Loop through each instance ID
    public_ips=""
    for instance_id in $instance_ids; do
      echo "Waiting for instance $instance_id to be running..."
      aws ec2 wait instance-running --instance-ids $instance_id --region $REGION
 
      echo "Creating SSH public IP newtork inteface for instance $instance_id..."
      interface_id=""
      INSTANCE_INFO=$(aws ec2 describe-instances --region $REGION --instance-ids $instance_id)
      OUTPUT=$(echo "$INSTANCE_INFO" | jq -r '.Reservations[0].Instances[0].NetworkInterfaces[] | "\(.Attachment.DeviceIndex),\(.NetworkInterfaceId)"')
      echo $OUTPUT
      for pair in $OUTPUT; do
          IFS="," read -r device_idx ni_id <<< $pair
          if [ "$device_idx" == "0" ]; then
              interface_id=$ni_id
              break
          fi
      done
      if [ "$interface_id" == "" ]; then
          exit -1
      fi
      echo $interface_id
 
      echo "Checking for unassociated Elastic IPs..."
      unassociated_eips=$(aws ec2 describe-addresses --region $REGION | jq -r '.Addresses[] | select(.AssociationId == null) | .AllocationId')
      if [[ -z "$unassociated_eips" ]]; then
          echo "No unassociated Elastic IPs found. Allocating new Elastic IP..."
          eip_output=$(aws ec2 allocate-address --domain vpc --region $REGION)
          eip_id=$(echo $eip_output | jq -r .AllocationId)
          echo "Allocated Elastic IP ID: $eip_id"
          eip_public_ip=$(echo $eip_output | jq -r .PublicIp)
          echo "Allocated Elastic IP Public IP: $eip_public_ip"
          echo "Note that this newly allocated Elasic IP will persist even after the instance termination"
          echo "If the Elastic IP is not going to be reused do not forget to delete it"
      else
          # use the first unassociated Elastic IP found
          eip_id=$(echo "$unassociated_eips" | head -n 1)
          echo "Found unassociated Elastic IP ID: $eip_id"
          eip_public_ip=$(aws ec2 describe-addresses --allocation-ids $eip_id --region $REGION | jq -r .Addresses[0].PublicIp)
          echo "Elastic IP Public IP: $eip_public_ip"
      fi
      public_ips+="${eip_public_ip} "
 
      echo "Associating Elastic IP with network interface $interface_id..."
      aws ec2 associate-address --allocation-id $eip_id --network-interface-id $interface_id --region $REGION
      echo "Associated Elastic IP with network interface."
    done
 
    echo "The instance has been launched.\nYou can now SSH into $public_ips with key $KEYNAME.\n"

.. note:: if you face connectivity issues after launching trn1\\trn1n 32xlarge instance on Ubuntu, please follow the troubleshooting instructions mentioned :ref:`here. <trn1_ubuntu_troubleshooting>`

