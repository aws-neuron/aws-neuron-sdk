.. _ec2-instance:

EC2 Instance
============
Introduction
------------
Use of Neuron in containers on EC2 can be simple to achieve by following these steps
    - :ref:`tutorial-docker-env-setup-for-neuron`
    - More details on EC2 setup `can be found at <https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/deep-learning-containers-ec2-setup.html>`_
DLC Images
----------
    - The location for DLC images for Neuron can be obtained from `here <https://github.com/aws/deep-learning-containers/blob/master/available_images.md>`_
    - To get the list of images for neuron, the following commands can be used.

      ``aws ecr list-images --registry-id 763104351884 --repository-name tensorflow-inference-neuron``

      ``aws ecr list-images --registry-id 763104351884 --repository-name pytorch-inference-neuron``

Setup recommendations
---------------------
    - The EC2 Inf1 instance needs to have the aws-neuron-runtime-base and aws-neruon-dkms package installed.
    - The DLC inference container runs the framework server (like tensorflow-model-server or TorchServe) and also the neuron runtime that interacts with the neuron driver running in the host.
    - For more details on setting up the container, check the `tensorflow <https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/deep-learning-containers-ec2-tutorials-inference.html#deep-learning-containers-ec2-tutorials-inference-tf>`_ or `pytorch <https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/deep-learning-containers-ec2-tutorials-inference.html#deep-learning-containers-ec2-tutorials-inference-pytorch>`_. Make sure the appropriate framework container image is used.

Debug Hints
-----------
    - Use the docker log command to get the neuron rtd logs in the container.

       ``docker logs <container-name>``
    - Look for errors like the following
        - If we see *nrtd[8]: [TDRV:tdrv_init_mla_phase1] Could not open the device index:0*, it either means that some other container is using that device or the host is running the neuron-rtd process.
        - Check to see that host is not running neuron-rtd

           ``sudo systemctl status neuron-rtd``
