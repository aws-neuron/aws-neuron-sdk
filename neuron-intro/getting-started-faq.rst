Getting started with Neuron FAQs
================================

.. contents::
   :local:
   :depth: 1


How can I get started?
~~~~~~~~~~~~~~~~~~~~~~~~~

You can start your workflow by training your model in one of the popular
ML frameworks using EC2 GPU instances, or alternatively download a pre-trained model.
Once the model is trained to your required accuracy, you can use the ML frameworks' API to invoke
Neuron, to re-target(i.e. compile) the model for execution on Inferentia. The compilation is done once and its artifacts can then be deployed at scale. Once compiled, the binary can be loaded into one or more chips to start service inference calls.

In order to get started quickly, you can use `AWS Deep Learning
AMIs <https://aws.amazon.com/machine-learning/amis/>`__ that come
pre-installed with ML frameworks and the Neuron SDK. For a fully managed
experience, you can use Amazon SageMaker to seamlessly deploy and accelerate your production models on `ml.inf1 instances <https://github.com/aws/amazon-sagemaker-examples/blob/master/sagemaker_neo_compilation_jobs/deploy_tensorflow_model_on_Inf1_instance/tensorflow_distributed_mnist_neo_inf1.ipynb>`__.

For customers who use popular frameworks like TensorFlow, Apache MXNet (Incubating) and
PyTorch, a guide to help you get started with frameworks is available
at:

-  :ref:`tensorflow-neuron`
-  :ref:`neuron-pytorch`
-  :ref:`neuron-mxnet`

You can also visit :ref:`neuron-gettingstarted`.

How do I select which Inf1 instance size to use?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The decision as to which Inf1 instance size to use is based upon the
application and its performance/cost targets. To assist, the Neuron Plugin
for TensorBoard will show actual results when executed on a given instance.
A guide to this process is available here: :ref:`neuron-plugin-tensorboard`.

As a rule of thumb, we encourage you to start with inf1.xlarge and test your model. For example, many computer vision models require pre/post processing that consume CPU resources and such models will get higher throughput on the inf1.2xlarge that provides higher ratio of vCPU/Chip.

We encourages you try out all the Inf1 instance
sizes with your specific models, until you find the best size for your application needs.
