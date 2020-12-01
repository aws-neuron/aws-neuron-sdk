General Neuron FAQs
===================

Getting started with Neuron FAQs
--------------------------------

Q: How can I get started?
~~~~~~~~~~~~~~~~~~~~~~~~~

You can start your workflow by training your model in one of the popular
ML frameworks using EC2 GPU instances, or alternatively download a pre-trained model. 
Once the model is trained to your required accuracy, you can use the ML frameworks' API to invoke
Neuron, to re-target(i.e. compile) the model for execution on Inferentia. The compilation is done once and its artifacts can then be deployed at scale. Once compiled, the binary can be loaded into one or more chips to start service inference calls. 

In order to get started quickly, you can use `AWS Deep Learning
AMIs <https://aws.amazon.com/machine-learning/amis/>`__ that come
pre-installed with ML frameworks and the Neuron SDK. For a fully managed
experience, you can use Amazon SageMaker to seamlessly deploy and accelerate your production models on `ml.inf1 instances <https://github.com/aws/amazon-sagemaker-examples/blob/master/sagemaker_neo_compilation_jobs/deploy_tensorflow_model_on_Inf1_instance/tensorflow_distributed_mnist_neo_inf1.ipynb>`__.

For customers who use popular frameworks like TensorFlow, MXNet and
PyTorch, a guide to help you get started with frameworks is available
at:

-  :ref:`neuron-tensorflow`
-  :ref:`neuron-pytorch`
-  :ref:`neuron-mxnet`

You can also visit :ref:`neuron-gettingstarted`.

Q: How do I select which Inf1 instance size to use?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The decision as to which Inf1 instance size to use is based upon the
application and its performance/cost targets. To assist, TensorBoard profiling
will show actual results when executed on a given instance. A guide to
this process is available here: :ref:`tensorboard-neuron`.

As a rule of thumb, we encourage you to start with inf1.xlarge and test your model. For example, many computer vision models require pre/post processing that consume CPU resources and such models will get higher throughput on the inf1.2xlarge that provides higher ratio of vCPU/Chip. 

We encourages you try out all the Inf1 instance
sizes with your specific models, until you find the best size for your application needs.

General
-------

Q: What ML models types and operators are supported by AWS Neuron?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AWS Neuron includes a compiler that converts your trained machine
learning models to a binary object for execution (aka Neuron Executable File Format or a NEFF file in short). The Neuron
compiler supports many commonly used machine learning operators used in computer vision, natural language processing, recommender engines and more. A list of supported ML operators and supported inputs are in :ref:`neuron-supported-operators` .

It's important to mention that to get good performance doens't require all of the model operators to run on the chip. In many cases, some of the operators will continue to run on the instance CPUs, like the case of embeddings or image pre-processing, and will still provide a compelling end to end performance. We call this approach auto-partitioning, where the Neuron compiler optimizes the model execution based on operators that are most suitable to run on the CPU or the chip.

We constantly add more operators based on customers' feedback.

Q: Why is a compiler needed, and how do I use it?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Neuron compiler converts from a framework level Neural Network
graph, with operators like convolution and pooling, into a
hardware-specific instruction set, builds the schedule for
execution of these instructions, and converts the model parameters into
format that the chip can consume. The supported input formats include
TensorFlow, PyTorch, and MXNet. The output from the
compiler is a Neuron Executable File Format (NEFF) artifact. The NEFF
contains a combination of binary code, the model parameters, and
additional meta-data needed by the Neuron runtime and profiler.

Q: I am using a ML framework today – what will change for me to use this?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To use Inferentia within the Inf1 instances, the developer need to perform one-time compilation
of the pre-trained model to generate a NEFF, and use this as the inference
model in fleet of Inf1 instances.

-  :ref:`neuron-tensorflow`
-  :ref:`neuron-pytorch`
-  :ref:`neuron-mxnet`

Q: What is a NeuronCore Pipeline ? and How do I take advantage of it?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A NeuronCore Pipeline is a unique technique to shard a specific Neural
Network across multiple NeuronCores, to take advantage of the large
on-chip cache instead of moving data in and out of external memory. The result is an increased throughput and reduce latency
typically important for real-time inference applications. All Inf1 instances support it, and the Inf1
instances with multiple Inferentia accelerators, such as inf1.6xlarge or
inf1.24xlarge support it thanks to the fast chip-to-chip interconnect.

Developers can choose to use NeuronCore Pipeline mode during compile
stage, with an opt-in flag. :ref:`neuron-cc` provides further details.

Q: NeuronCores, NeuronCore Groups and NeuronCore Pipelines: What do they do?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each Inferentia chip has four compute engines called NeuronCores. A
NeuronCore Group is a way to aggregate NeuronCores to increase hardware
utilization and assign models with the right compute sizing for a
specific application. If you want to run mutliple models in parallel,
you can assign different models to separate NeuronCore Groups. A model
compiled to use multiple NeuronCores in a NeuronCore Pipeline can be
assigned to a NeuronCore Group with enough NeuronCores to load into.
Finally- it is also possible for sets of Inferentia devices to be mapped
to separate Neuron Runtimes. :ref:`neuron-fundamentals` section has more
information and examples.

Q: Can I use TensorFlow networks from tfhub.dev as-is ? if not, what should I do?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes. Models format can be imported into TensorFlow, either as a standard
model-server, in which case it appears as a simple command line utility,
or via the Python based TensorFlow environment. The primary additional
step needed is to compile the model into Inferentia NEFF format.

Troubleshooting FAQs
--------------------

Q: Performance is not what I expect it to be, what's the next step?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Please check our :ref:`performance-optimization` section on performance
tuning and other notes on how to use pipelining and batching to improve
performance!

Q: Do I need to worry about size of model and size of inferentia memory? what problems can I expect to have?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Errors like this will be logged and can be found as shown
:ref:`neuron_gatherinfo`.

Q: How can I debug / profile my inference request?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See :ref:`tensorboard-neuron`

Contributing Guidelines FAQs
----------------------------

Whether it's
a bug report, new feature, correction, or additional documentation, we
greatly value feedback and contributions from our community.

Please read through this document before submitting any issues or pull
requests to ensure we have all the necessary information to effectively
respond to your bug report or contribution.

Q: How to reporting Bugs/Feature Requests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We welcome you to use the GitHub issue tracker to report bugs or suggest
features.

When filing an issue, please check existing open, or recently closed,
issues to make sure somebody else hasn't already reported the issue.
Please try to include as much information as you can. Details like these
are incredibly useful:

-  A reproducible test case or series of steps
-  The version of our code being used
-  Any modifications you've made relevant to the bug
-  Anything unusual about your environment or deployment

Q: Contributing via Pull Requests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Contributions via pull requests are much appreciated. Before sending us
a pull request, please ensure that:

1. You are working against the latest source on the *master* branch.
2. You check existing open, and recently merged, pull requests to make
   sure someone else hasn't addressed the problem already.
3. You open an issue to discuss any significant work - we would hate for
   your time to be wasted.

To send us a pull request, please:

1. Fork the repository.
2. Modify the source; please focus on the specific change you are
   contributing. If you also reformat all the code, it will be hard for
   us to focus on your change.
3. Ensure local tests pass.
4. Commit to your fork using clear commit messages.
5. Send us a pull request, answering any default questions in the pull
   request interface.
6. Pay attention to any automated CI failures reported in the pull
   request, and stay involved in the conversation.

GitHub provides additional document on `forking a
repository <https://help.github.com/articles/fork-a-repo/>`__ and
`creating a pull
request <https://help.github.com/articles/creating-a-pull-request/>`__.

Q: How to find contributions to work on
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Looking at the existing issues is a great way to find something to
contribute on. As our projects, by default, use the default GitHub issue
labels (enhancement/bug/duplicate/help wanted/invalid/question/wontfix),
looking at any 'help wanted' issues is a great place to start.

Q: What is the code of conduct
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This project has adopted the `Amazon Open Source Code of
Conduct <https://aws.github.io/code-of-conduct>`__. For more information
see the `Code of Conduct
FAQ <https://aws.github.io/code-of-conduct-faq>`__ or contact
opensource-codeofconduct@amazon.com with any additional questions or
comments.

Q: How to notify for a security issue
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you discover a potential security issue in this project we ask that
you notify AWS/Amazon Security via our `vulnerability reporting
page <http://aws.amazon.com/security/vulnerability-reporting/>`__.
Please do **not** create a public github issue.

Q: What is the licensing
~~~~~~~~~~~~~~~~~~~~~~~~

See the :ref:`license-dicumentation` and :ref:`license-summary-docs-samples` files
for our project's licensing. We will ask you to confirm the licensing of
your contribution.

We may ask you to sign a `Contributor License Agreement
(CLA) <http://en.wikipedia.org/wiki/Contributor_License_Agreement>`__
for larger changes.
