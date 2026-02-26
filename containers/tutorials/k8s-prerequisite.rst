.. _k8s-prerequisite:

.. meta::
   :description: Learn how to create an Amazon EKS cluster with AWS Trainium instances (Trn1, Trn2) for machine learning workloads using AWS Neuron SDK. Step-by-step guide with eksctl and CloudFormation templates.
   :keywords: EKS, Kubernetes, Trainium, Trn1, Trn2, Neuron, AWS, machine learning, distributed training, eksctl, CloudFormation, EFA, node group

Before setting up Neuron components on your EKS cluster, you must create an EKS cluster and add Neuron-enabled nodes. This section guides you through creating an Amazon Elastic Kubernetes Service (EKS) cluster with AWS Trainium-enabled nodes (Trn1 or Trn2 instances) using CloudFormation templates and the eksctl command-line tool. You'll configure optimized networking with Elastic Fabric Adapter (EFA) support and pre-configured Neuron components for distributed training and inference workloads.

For detailed information, refer to:

* `EKS Cluster Creation Guide <https://docs.aws.amazon.com/eks/latest/userguide/create-cluster.html>`_
* `EKS Compute Resources Guide <https://docs.aws.amazon.com/eks/latest/userguide/eks-compute.html>`_
* `eksctl Getting Started <https://eksctl.io/getting-started/>`_

**Step 1: Download Node Group Template**

Download the node group CloudFormation template for your instance type.

.. tab-set::

   .. tab-item:: Trn1

      .. code-block:: bash

         wget https://raw.githubusercontent.com/aws-neuron/aws-neuron-eks-samples/master/dp_bert_hf_pretrain/cfn/eks_trn1_ng_stack.yaml

   .. tab-item:: Trn2

      .. code-block:: bash

         wget https://raw.githubusercontent.com/aws-neuron/aws-neuron-eks-samples/master/dp_bert_hf_pretrain/cfn/eks_trn2_ng_stack_al2023.yaml

**Important template configuration information**

* **Placement Group:** Optimizes network speed between nodes
* **EFA Driver:** Installed automatically (ensure ``libfabric`` version matches between AMI and workload containers)
* **AMI:** Uses `EKS optimized accelerated AMI <https://docs.aws.amazon.com/eks/latest/userguide/eks-optimized-ami.html#gpu-ami>`_ with Neuron components pre-installed
* **Instance Type:** Configured for trn1.32xlarge or trn2.48xlarge (update to your desired instance type)
* **Kubernetes Version:** Trn1 templates use Kubernetes 1.25+, Trn2 templates use Kubernetes 1.34+ (update as needed)

Trn2 LNC configuration (Optional):

Trn2 instances use a default Logical NeuronCore Configuration (LNC) of ``2``. To change it to ``1``, update the ``UserData`` section of the launch template:

.. code-block:: bash

    --==BOUNDARY==
    Content-Type: text/x-shellscript; charset="us-ascii"

    #!/bin/bash
    set -ex
    config_dir=/opt/aws/neuron
    config_file=${config_dir}/logical_nc_config
    [ -d "$config_dir" ] || mkdir -p "$config_dir"
    [ -f "$config_file" ] || touch "$config_file"
    if ! grep -q "^NEURON_LOGICAL_NC_CONFIG=1$" "$config_file" 2>/dev/null; then
        printf "NEURON_LOGICAL_NC_CONFIG=1" >> "$config_file"
    fi
    --==BOUNDARY==--

**Step 2: Create Cluster Parameter Script**

Create a bash script to capture the parameters needed for the node template:

.. tab-set::

   .. tab-item:: Trn1

      .. code-block:: bash

        #!/bin/bash

        CLUSTER_NAME=$1
        CLUSTER_SG=$(eksctl get cluster $CLUSTER_NAME -o json | jq -r ".[0].ResourcesVpcConfig.ClusterSecurityGroupId")
        VPC_ID=$(eksctl get cluster $CLUSTER_NAME -o json | jq -r ".[0].ResourcesVpcConfig.VpcId")

        cat <<EOF > cfn_params.json
        [
            {
                "ParameterKey": "ClusterName",
                "ParameterValue": "$CLUSTER_NAME"
            },
            {
                "ParameterKey": "ClusterControlPlaneSecurityGroup",
                "ParameterValue": "$CLUSTER_SG"
            },
            {
                "ParameterKey": "VpcId",
                "ParameterValue": "$VPC_ID"
            }
        ]
        EOF

   .. tab-item:: Trn2

      .. code-block:: bash

          #!/bin/bash

          CLUSTER_NAME=$1
          CLUSTER_SG=$(eksctl get cluster $CLUSTER_NAME -o json | jq -r ".[0].ResourcesVpcConfig.ClusterSecurityGroupId")
          VPC_ID=$(eksctl get cluster $CLUSTER_NAME -o json | jq -r ".[0].ResourcesVpcConfig.VpcId")
          CLUSTER_ENDPOINT=$(eksctl get cluster $CLUSTER_NAME -o json | jq -r ".[0].Endpoint")
          CLUSTER_SERVICE_CIDR=$(eksctl get cluster $CLUSTER_NAME -o json | jq -r ".[0].KubernetesNetworkConfig.ServiceIpv4Cidr")
          CLUSTER_CA=$(eksctl get cluster $CLUSTER_NAME -o json | jq -r ".[0].CertificateAuthority.Data")

          cat <<EOF > cfn_params.json
          [
              {
                  "ParameterKey": "ClusterName",
                  "ParameterValue": "$CLUSTER_NAME"
              },
              {
                  "ParameterKey": "ClusterControlPlaneSecurityGroup",
                  "ParameterValue": "$CLUSTER_SG"
              },
              {
                  "ParameterKey": "VpcId",
                  "ParameterValue": "$VPC_ID"
              },
              {
                  "ParameterKey": "ClusterEndpoint",
                  "ParameterValue": "$CLUSTER_ENDPOINT"
              },
              {
                  "ParameterKey": "ClusterServiceCidr",
                  "ParameterValue": "$CLUSTER_SERVICE_CIDR"
              },
              {
                  "ParameterKey": "ClusterCertificateAuthority",
                  "ParameterValue": "$CLUSTER_CA"
              }
          ]
          EOF


This script captures the cluster name, security group for control plane connectivity, and VPC ID.

**Step 3: Create CloudFormation Stack**

Create the CloudFormation stack for the node group.

.. tab-set::

   .. tab-item:: Trn1

      .. code-block:: bash

         aws cloudformation create-stack \
             --stack-name eks-trn1-ng-stack \
             --template-body file://eks_trn1_ng_stack.yaml \
             --parameters file://cfn_params.json \
             --capabilities CAPABILITY_IAM

   .. tab-item:: Trn2

      .. code-block:: bash

         aws cloudformation create-stack \
             --stack-name eks-trn2-ng-stack \
             --template-body file://eks_trn2_ng_stack_al2023.yaml \
             --parameters file://cfn_params.json \
             --capabilities CAPABILITY_IAM

Wait for the stack creation to complete before proceeding. You can monitor the progress in the AWS CloudFormation console.

**Step 4: Determine Availability Zones**

Identify the availability zones for your cluster:

.. code-block:: bash

    aws ec2 describe-availability-zones \
        --region $REGION_CODE \
        --query "AvailabilityZones[]" \
        --filters "Name=zone-id,Values=$1" \
        --query "AvailabilityZones[].ZoneName" \
        --output text

**Step 5: Generate Node Group Configuration**

Create a script named ``create_ng_yaml.sh`` to generate the node group YAML configuration. The script requires: region, availability zones, cluster name, and CloudFormation stack name.

.. tab-set::

   .. tab-item:: Trn1

      .. code-block:: bash

         #!/bin/bash

         REGION_CODE=$1
         EKSAZ1=$2
         EKSAZ2=$3
         CLUSTER_NAME=$4
         STACKNAME=$5

         LT_ID_TRN1=$(aws cloudformation describe-stacks --stack-name $STACKNAME \
                 --query "Stacks[0].Outputs[?OutputKey=='LaunchTemplateIdTrn1'].OutputValue" \
                 --output text)

         cat <<EOF > trn1_nodegroup.yaml
         apiVersion: eksctl.io/v1alpha5
         kind: ClusterConfig

         metadata:
           name: $CLUSTER_NAME
           region: $REGION_CODE
           version: "1.28"

         iam:
           withOIDC: true

         availabilityZones: ["$EKSAZ1","$EKSAZ2"]

         managedNodeGroups:
           - name: trn1-32xl-ng1
             launchTemplate:
               id: $LT_ID_TRN1
             minSize: 1
             desiredCapacity: 1
             maxSize: 1
             availabilityZones: ["$EKSAZ1"]
             privateNetworking: true
             efaEnabled: true
         EOF

   .. tab-item:: Trn2

      .. code-block:: bash

         #!/bin/bash

         REGION_CODE=$1
         EKSAZ1=$2
         EKSAZ2=$3
         CLUSTER_NAME=$4
         STACKNAME=$5

         LT_ID_TRN2=$(aws cloudformation describe-stacks --stack-name $STACKNAME \
                 --query "Stacks[0].Outputs[?OutputKey=='LaunchTemplateIdTrn2'].OutputValue" \
                 --output text)

         cat <<EOF > trn2_nodegroup.yaml
         apiVersion: eksctl.io/v1alpha5
         kind: ClusterConfig

         metadata:
           name: $CLUSTER_NAME
           region: $REGION_CODE
           version: "1.34"

         iam:
           withOIDC: true

         availabilityZones: ["$EKSAZ1","$EKSAZ2"]

         managedNodeGroups:
           - name: trn2-48xl-ng1
             launchTemplate:
               id: $LT_ID_TRN2
             minSize: 1
             desiredCapacity: 1
             maxSize: 1
             availabilityZones: ["$EKSAZ1"]
             privateNetworking: true
             efaEnabled: true
         EOF

Run the script to generate the configuration file. Update the Kubernetes version as needed for your environment.

Example output:

.. tab-set::

   .. tab-item:: Trn1

      .. code-block:: yaml

         apiVersion: eksctl.io/v1alpha5
         kind: ClusterConfig

         metadata:
           name: nemo2
           region: us-west-2
           version: "1.28"

         iam:
           withOIDC: true

         availabilityZones: ["us-west-2d","us-west-2c"]

         managedNodeGroups:
           - name: trn1-32xl-ng1
             launchTemplate:
               id: lt-093c222b35ea89009
             minSize: 1
             desiredCapacity: 1
             maxSize: 1
             availabilityZones: ["us-west-2d"]
             privateNetworking: true
             efaEnabled: true

   .. tab-item:: Trn2

      .. code-block:: yaml

         apiVersion: eksctl.io/v1alpha5
         kind: ClusterConfig

         metadata:
           name: nemo2
           region: us-west-2
           version: "1.34"

         iam:
           withOIDC: true

         availabilityZones: ["us-west-2d","us-west-2c"]

         managedNodeGroups:
           - name: trn2-48xl-ng1
             launchTemplate:
               id: lt-093c222b35ea89010
             minSize: 1
             desiredCapacity: 1
             maxSize: 1
             availabilityZones: ["us-west-2d"]
             privateNetworking: true
             efaEnabled: true

**Step 6: Create Node Group**

Create the node group using the generated configuration.

.. tab-set::

   .. tab-item:: Trn1

      .. code-block:: bash

         eksctl create nodegroup -f trn1_nodegroup.yaml

   .. tab-item:: Trn2

      .. code-block:: bash

         eksctl create nodegroup -f trn2_nodegroup.yaml

Wait for the nodes to reach the ``Ready`` state. Verify using:

.. code-block:: bash

    kubectl get nodes

**Step 7: Install EFA Device Plugin (Optional)**

If you plan to run distributed training or inference jobs, install the EFA device plugin following the instructions at the `EFA device plugin repository <https://github.com/aws-samples/aws-efa-eks>`_.
