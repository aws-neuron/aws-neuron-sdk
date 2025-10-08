Please refer to `EKS instructions <https://docs.aws.amazon.com/eks/latest/userguide/create-cluster.html>`_ to create a cluster. Once the cluster is ACTIVE, please add nodes to the cluster. We recommend using node template for neuron nodes. Following example demonstrates how to add neuron nodes using node template. The example adds managed nodes using `eksctl tool <https://eksctl.io/getting-started/>`__. For more details, please refer to `EKS User Guide <https://docs.aws.amazon.com/eks/latest/userguide/eks-compute.html>`_.


As first step, please create a script to capture the parameters for the node template:

.. code-block:: bash

    #!/bin/bash

    CLUSTER_NAME=$1
    CLUSTER_SG=$(eksctl get cluster $CLUSTER_NAME -o json|jq -r ".[0].ResourcesVpcConfig.ClusterSecurityGroupId")
    VPC_ID=$(eksctl get cluster $CLUSTER_NAME -o json|jq -r ".[0].ResourcesVpcConfig.VpcId")

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

These parameters include the name of the cluster, the security group the nodes can use to connect to the control plane and the vpcid.
Next, get the node group template from tutorial below -

.. code-block:: bash

    wget https://raw.githubusercontent.com/aws-neuron/aws-neuron-eks-samples/master/dp_bert_hf_pretrain/cfn/eks_trn1_ng_stack.yaml


This template file has a few important config settings -

* It places the node in a placement group. This optimizes the network speed between the nodes.
* The template installs the EFA driver. Please note that the libfabric version should match between the AMI and the workload containers.
* It uses the `EKS optimized accelerated AMI <https://docs.aws.amazon.com/eks/latest/userguide/eks-optimized-ami.html#gpu-ami>`__ which  has the necessary neuron components installed. The template uses AMI for Kubernetes version 1.25. Please update to appropriate version.
* The template adds trn1.32xlarge nodes to the cluster. Please update to the desired instance type.
* Trn2 instance types use a default LNC (Logical NeuronCore Configuration) setting of `2`, if you want to change it to `1`, update the UserData section of the launch template to a new LNC setting as shown below, and deploy the new/updated version of launch template.

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

Finally, run the following command to create cloud formation stack:

.. code-block:: bash

    aws cloudformation create-stack \
    --stack-name eks-trn1-ng-stack \
    --template-body file://eks_trn1_ng_stack.yaml \
    --parameters file://cfn_params.json \
    --capabilities CAPABILITY_IAM


The above command will create a stack named eks-trn1-ng-stack, which will be visible in cloudformation.
Please wait for that stack creation to complete before proceeding to next step.

Now we are ready to add the nodes. The example will demonstrate creating node groups using eksctl tool.

Please run following command to determine the AZs:

.. code-block:: bash

    aws ec2 describe-availability-zones \
    --region $REGION_CODE \
    --query "AvailabilityZones[]" \
    --filters "Name=zone-id,Values=$1" \
    --query "AvailabilityZones[].ZoneName" \
    --output text

Next, create a script named create_ng_yaml.sh to generate node group yaml. The arguments to the script include the region, AZs, cluster name and name of the cloudformation stack created earlier (eks-trn1-ng-stack in case of this example):

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

Run the above script. It should produce a yaml similar to -

.. code-block:: bash

    apiVersion: eksctl.io/v1alpha5
    kind: ClusterConfig

    metadata:
      name: nemo2
      region: us-west-2
      version: "1.25"

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

The example shows kubernetes version 1.25. Please update the version as needed. This yaml can now be used with eksctl.

.. code-block:: bash

    eksctl create nodegroup -f trn1_nodegroup.yaml


This will add the nodes to the cluster. Please wait for the nodes to be 'Ready'. This can be verified using the get node command.

.. code-block:: bash
  
    kubectl get node

If you are running a distributed training or inference job, you will need EFA resources. Please install the EFA device plugin using instructions at `EFA device plugin repository <https://github.com/aws-samples/aws-efa-eks>`_.

Next, we will install the Neuron Device Plugin.
