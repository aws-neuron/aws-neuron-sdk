# Neuron K8 Release Notes

This document lists the current release notes for AWS Neuron Kubernetes (k8) components.  Neuron K8 components include a device plugin and a scheduler extension to assist with deployment and management of inf1 nodes within Kubernetes clusters.  Both components are offered as pre-built containers in ECR ready for deployment.  

* **Device Plugin:** 790709498068.dkr.ecr.us-east-1.amazonaws.com/neuron-device-plugin:latest  
* **Neuron Scheduler:** 790709498068.dkr.ecr.us-east-1.amazonaws.com/neuron-scheduler:latest  
  
It's recommended to pin the version of the components used and to never use the "latest" tag.  To get the list of image tags, please refer to these notes or pull from the ECR directy like so:
```
aws ecr list-images --registry-id 790709498068 --repository-name  neuron-device-plugin --region us-west-2
aws ecr list-images --registry-id 790709498068 --repository-name  neuron-scheduler --region us-west-2
```

To Pull the Images from ECR:
```
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 790709498068.dkr.ecr.us-west-2.amazonaws.com
docker pull  790709498068.dkr.ecr.us-west-2.amazonaws.com/neuron-device-plugin
docker pull  790709498068.dkr.ecr.us-west-2.amazonaws.com/neuron-scheduler
```
# [1.2.0.0]

Date: 11/17/2020

## Summary
Minor internal enhancements.


# [1.1.23.0]

Date: 10/22/2020

## Summary
Support added for use with Neuron Runtime 1.1.  More details in the Neuron Runtime release notes [here](./neuron-runtime.md).



# [1.1.17.0]

Date: 09/22/2020

## Summary
Minor internal enhancements.


# [1.0.11000.0]

Date: 08/08/2020

## Summary
First release of the Neuron K8 Scheduler extension.

## Major New Features
* New scheduler extension is provided to ensure that kubelet is scheduling pods on inf1 with contiguous device ids.  Additional details about the new scheduler are provided [here](../docs/neuron-container-tools/k8s-neuron-scheduler.md), including instructions on how to apply it.  
  * NOTE: The scheduler is only required when using inf1.6xlarge and/or inf1.24xlarge
* With this release the device plugin now requires RBAC permission changes to get/patch NODE/POD objects.  Please apply the [k8s-neuron-device-plugin-rbac.yml](../docs/neuron-container-tools/k8s-neuron-device-plugin-rbac.yml) before using the new device plugin.

## Resolved Issues
* Scheduler is intended to address https://github.com/aws/aws-neuron-sdk/issues/110
