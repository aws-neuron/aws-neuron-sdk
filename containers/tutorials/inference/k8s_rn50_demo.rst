.. _example-deploy-rn50-as-k8s-service:

Deploy a TensorFlow Resnet50 model as a Kubernetes service
----------------------------------------------------------

This tutorial uses Resnet50 model as a teaching example on how to deploy an
inference application using Kubernetes on the Inf1 instances.

Prerequisite:
^^^^^^^^^^^^^

-  Please follow instructions at :ref:`tutorial-k8s-env-setup-for-neuron` to setup k8s support on your cluster.
-  Inf1 instances as worker nodes with attached roles allowing:

   -  ECR read access policy to retrieve container images from ECR:
      **arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly**
   -  S3 access to retrieve saved_model from within tensorflow serving
      container.

Deploy a TensorFlow Serving application image
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A trained model must be compiled to an Inferentia target before it can be deployed on Inferentia instances\.
To continue, you will need a Neuron-optimized TensorFlow model saved in Amazon S3\.
If you don’t already have a SavedModel, please follow the tutorial for `creating a Neuron compatible ResNet50 model <https://docs.aws.amazon.com/dlami/latest/devguide/tutorial-inferentia-tf-neuron.html>`_
and upload the resulting SavedModel to S3\.

ResNet-50 is a popular machine learning model used for image
classification tasks\. For more information about compiling Neuron models, see
`The AWS Inferentia Chip With DLAMI <https://docs.aws.amazon.com/dlami/latest/devguide/tutorial-inferentia.html>`_
in the AWS Deep Learning AMI Developer Guide\.

The sample deployment manifest manages a pre-built inference serving container for TensorFlow provided by
AWS Deep Learning Containers. Inside the container is the AWS Neuron Runtime and the TensorFlow Serving application.
A complete list of pre-built Deep Learning Containers optimized for Neuron is maintained on GitHub under
`Available Images <https://github.com/aws/deep-learning-containers/blob/master/available_images.md#user-content-neuron-containers>`_.
At start\-up, the DLC will fetch your model from Amazon S3, launch Neuron TensorFlow Serving with the saved model,
and wait for prediction requests\.

The number of Neuron devices allocated to your serving application can be adjusted by changing the
`aws.amazon.com/neuron` resource in the deployment yaml\. Please note that communication between TensorFlow Serving
and the Neuron runtime happens over GRPC, which requires passing the `IPC_LOCK` capability to the container.

1. Create a file named `rn50_deployment.yaml` with the contents below\. Update the region\-code and model path to match your desired settings. The model name is for identification purposes when a client makes a request to the TensorFlow server\. This example uses a model name to match a sample ResNet50 client script that will be used in a later step for sending prediction requests\.

.. note::
   1. Replace the s3 bucket name in model_base_path arg in the file with the location of the where the saved model was stored in s3.
   2. In the image:  add the appropriate location of the DLC tensorflow image


::

   kind: Deployment
   apiVersion: apps/v1
   metadata:
     name: k8s-neuron-test
     labels:
       app: k8s-neuron-test
       role: master
   spec:
     replicas: 2
     selector:
       matchLabels:
         app: k8s-neuron-test
         role: master
     template:
       metadata:
         labels:
           app: k8s-neuron-test
           role: master
       spec:
         containers:
           - name: k8s-neuron-test
             image: 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference-neuron:1.15.4-neuron-py37-ubuntu18.04
             command:
               - /usr/local/bin/entrypoint.sh
             args:
               - --port=8500
               - --rest_api_port=9000
               - --model_name=resnet50_neuron
               - --model_base_path=s3://${your-bucket-of-models}/resnet50_neuron/
             ports:
               - containerPort: 8500
               - containerPort: 9000
             imagePullPolicy: IfNotPresent
             env:
               - name: AWS_REGION
                 value: "us-east-1"
               - name: S3_USE_HTTPS
                 value: "1"
               - name: S3_VERIFY_SSL
                 value: "0"
               - name: S3_ENDPOINT
                 value: s3.us-east-1.amazonaws.com
               - name: AWS_LOG_LEVEL
                 value: "3"
             resources:
               limits:
                 cpu: 4
                 memory: 4Gi
                 aws.amazon.com/neuron: 1
               requests:
                 cpu: "1"
                 memory: 1Gi
             securityContext:
               capabilities:
                 add:
                   - IPC_LOCK

2. Deploy the model\.

::

   kubectl apply -f rn50_deployment.yaml

3. Create a file named `rn50_service.yaml` with the following contents\. The HTTP and gRPC ports are opened for accepting prediction requests\.

::

   kind: Service
   apiVersion: v1
   metadata:
     name: k8s-neuron-test
     labels:
       app: k8s-neuron-test
   spec:
     type: ClusterIP
     ports:
       - name: http-tf-serving
         port: 8500
         targetPort: 8500
       - name: grpc-tf-serving
         port: 9000
         targetPort: 9000
     selector:
       app: k8s-neuron-test
       role: master


4. Create a Kubernetes service for your TensorFlow model Serving application\.

::

   kubectl apply -f rn50_service.yaml

Make predictions against your TensorFlow Serving service
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. To test locally, forward the gRPC port to the `k8s-neuron-test` service\.

::

   kubectl port-forward service/k8s-neuron-test 8500:8500 &

2. Create a Python script called `tensorflow-model-server-infer.py` with the following content. This script runs inference via gRPC, which is service framework.

::

   import numpy as np
   import grpc
   import tensorflow as tf
   from tensorflow.keras.preprocessing import image
   from tensorflow.keras.applications.resnet50 import preprocess_input
   from tensorflow_serving.apis import predict_pb2
   from tensorflow_serving.apis import prediction_service_pb2_grpc
   from tensorflow.keras.applications.resnet50 import decode_predictions

   if __name__ == '__main__':
       channel = grpc.insecure_channel('localhost:8500')
       stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
       img_file = tf.keras.utils.get_file(
           "./kitten_small.jpg",
           "https://raw.githubusercontent.com/awslabs/mxnet-model-server/master/docs/images/kitten_small.jpg")
       img = image.load_img(img_file, target_size=(224, 224))
       img_array = preprocess_input(image.img_to_array(img)[None, ...])
       request = predict_pb2.PredictRequest()
       request.model_spec.name = 'resnet50_inf1'
       request.inputs['input'].CopyFrom(
           tf.make_tensor_proto(img_array, shape=img_array.shape))
       result = stub.Predict(request)
       prediction = tf.make_ndarray(result.outputs['output'])
       print(decode_predictions(prediction))

3. Run the script to submit predictions to your service\.
::

   python3 tensorflow-model-server-infer.py

   Your output should look like the following:

::

   [[(u'n02123045', u'tabby', 0.68817204), (u'n02127052', u'lynx', 0.12701613), (u'n02123159', u'tiger_cat', 0.08736559), (u'n02124075', u'Egyptian_cat', 0.063844085), (u'n02128757', u'snow_leopard', 0.009240591)]]
