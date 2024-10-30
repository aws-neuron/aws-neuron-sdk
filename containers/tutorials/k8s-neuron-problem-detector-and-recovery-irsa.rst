.. _k8s-neuron-problem-detector-and-recovery-irsa:

Neuron node problem detection and recovery is authorized via IAM roles for service accounts. For more information, see `IAM roles for service accounts <https://docs.aws.amazon.com/eks/latest/userguide/iam-roles-for-service-accounts.html>`__ in the Amazon EKS User Guide. This documentation shows how to configure an IAM role for service accounts using the command line tool eksctl. Follow the instructions below to configure IAM authorization for service accounts:

* Install the eksctl CLI using instructions listed at https://eksctl.io/installation/.
* Create a policy as shown below:

    Policy template

    .. code:: bash

        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Action": [
                        "autoscaling:SetInstanceHealth",
                        "autoscaling:DescribeAutoScalingInstances"
                    ],
                    "Effect": "Allow",
                    "Resource": <arn of the Auto Scaling group corresponding to the Neuron nodes for the cluster>
                },
                {
                    "Action": [
                        "ec2:DescribeInstances"
                    ],
                    "Effect": "Allow",
                    "Resource": "*",
                    "Condition": {
                        "ForAllValues:StringEquals": {
                            "ec2:ResourceTag/aws:autoscaling:groupName": <name of the Auto Scaling group corresponding to the Neuron nodes for the cluster>
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

    To create the policy, the AWS CLI can be used as shown below, where npd-policy-trimmed.json is the JSON policy constructed from the template above.

    .. code:: bash

        aws iam create-policy   \
          --policy-name NeuronProblemDetectorPolicy \
          --policy-document file://npd-policy-trimmed.json

* Create a namespace for the Neuron Node Problem Detector and its service account:

    .. code:: bash

        kubectl create ns neuron-healthcheck-system

* Associate the authorization with the service account using the following script:

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

* Verify that the service account is annotated correctly. An example is shown below:

    .. code:: bash

        kubectl describe sa node-problem-detector -n neuron-healthcheck-system
        Name:                node-problem-detector
        Namespace:           neuron-healthcheck-system
        Labels:              app.kubernetes.io/managed-by=eksctl
        Annotations:         eks.amazonaws.com/role-arn: arn:aws:iam::111111111111:role/neuron-problem-detector-role-cluster1
        Image pull secrets:  <none>
        Mountable secrets:   <none>
        Tokens:              <none>
        Events:              <none>

* To cleanup, deletion of the service account can be done using the following command:

    .. code:: bash

        #!/bin/bash
        CLUSTER_NAME=<eks cluster name>
        REGION_CODE=$(aws configure get region)

        eksctl delete iamserviceaccount \
            --name node-problem-detector \
            --namespace neuron-healthcheck-system \
            --cluster $CLUSTER_NAME \
            --approve \
            --region $REGION_CODE \
