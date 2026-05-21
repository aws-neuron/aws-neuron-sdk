.. _k8s-neuron-problem-detector-and-recovery-irsa:

Permissions for Neuron Node Problem Detector and Recovery
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Neuron Node Problem Detector and Recovery requires IAM roles for service accounts (IRSA) for authorization. For more information, see `IAM roles for service accounts <https://docs.aws.amazon.com/eks/latest/userguide/iam-roles-for-service-accounts.html>`__ in the Amazon EKS User Guide.

This section shows how to configure an IAM role for service accounts using the ``eksctl`` command-line tool.

**Step 1: Install eksctl**

Install the ``eksctl`` CLI using the instructions at https://eksctl.io/installation/.

**Step 2: Create IAM Policy**

Create an IAM policy that grants the necessary permissions for the Neuron Node Problem Detector.

.. code:: json

    {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Action": [
                    "autoscaling:SetInstanceHealth",
                    "autoscaling:DescribeAutoScalingInstances"
                ],
                "Effect": "Allow",
                "Resource": "<arn of the Auto Scaling group corresponding to the Neuron nodes for the cluster>"
            },
            {
                "Action": [
                    "ec2:DescribeInstances"
                ],
                "Effect": "Allow",
                "Resource": "*",
                "Condition": {
                    "ForAllValues:StringEquals": {
                        "ec2:ResourceTag/aws:autoscaling:groupName": "<name of the Auto Scaling group corresponding to the Neuron nodes for the cluster>"
                    }
                }
            },
            {
                "Action": [
                    "cloudwatch:PutMetricData"
                ],
                "Effect": "Allow",
                "Resource": "*",
                "Condition": {
                    "StringEquals": {
                        "cloudwatch:Namespace": "NeuronHealthCheck"
                    }
                }
            }
        ]
    }

Save the policy template above to a file named ``npd-policy.json`` (replacing the placeholder values), then run:

.. code:: bash

    aws iam create-policy \
        --policy-name NeuronProblemDetectorPolicy \
        --policy-document file://npd-policy.json

**Step 3: Create Namespace and Service Account**

Create a dedicated namespace for the Neuron Node Problem Detector:

.. code:: bash

    kubectl create ns neuron-healthcheck-system

**Step 4: Associate IAM Role with Service Account**

Use the following script to create the service account and associate it with the IAM role:

.. code:: bash

    #!/bin/bash
    CLUSTER_NAME=<eks cluster name>
    REGION_CODE=$(aws configure get region)
    POLICY_ARN=<policy arn for NeuronProblemDetectorPolicy>

    eksctl create iamserviceaccount \
        --name node-problem-detector \
        --namespace neuron-healthcheck-system \
        --cluster $CLUSTER_NAME \
        --attach-policy-arn $POLICY_ARN \
        --approve \
        --role-name neuron-problem-detector-role-$CLUSTER_NAME \
        --region $REGION_CODE \
        --override-existing-serviceaccounts

**Step 5: Verify Service Account Configuration**

Verify that the service account is annotated correctly with the IAM role:

.. code:: bash

    kubectl describe sa node-problem-detector -n neuron-healthcheck-system

Expected output:

.. code:: bash

    Name:                node-problem-detector
    Namespace:           neuron-healthcheck-system
    Labels:              app.kubernetes.io/managed-by=eksctl
    Annotations:         eks.amazonaws.com/role-arn: arn:aws:iam::111111111111:role/neuron-problem-detector-role-cluster1
    Image pull secrets:  <none>
    Mountable secrets:   <none>
    Tokens:              <none>
    Events:              <none>

**Cleanup**

To remove the service account and associated IAM role, use the following command:

.. code:: bash

    #!/bin/bash
    CLUSTER_NAME=<eks cluster name>
    REGION_CODE=$(aws configure get region)

    eksctl delete iamserviceaccount \
        --name node-problem-detector \
        --namespace neuron-healthcheck-system \
        --cluster $CLUSTER_NAME \
        --approve \
        --region $REGION_CODE
