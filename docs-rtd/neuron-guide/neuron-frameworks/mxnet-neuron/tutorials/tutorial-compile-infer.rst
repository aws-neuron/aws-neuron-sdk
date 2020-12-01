.. _mxnet-resnet50:

Getting Started with MXNet (ResNet50)
=====================================

Neuron supports Python module, Symbol APIs and the C predict API. The
following quick start example uses the Symbol API.

Steps Overview:
---------------

1. Launch an EC2 instance for compilation and inference
2. Install MXNet-Neuron and Neuron Compiler on a compilation instance
3. Compile on compilation server
4. Install MXNet-Neuron and Neuron Runtime on inference instance
5. Execute inference on Inf1

Step 1: Launch EC2 Instances
----------------------------

A typical workflow with the Neuron SDK will be to compile trained ML
models on a compilation server and then distribute the artifacts to a
fleet of Inf1 instances for execution. Neuron enables MXNet to be used
for all of these steps.

1.1. Select an AMI of your choice. Refer to the
:ref:`neuron-install-guide` for details.

1.2. Select and launch an EC2 instance of your choice to compile. Launch
an instance by following `EC2
instructions <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html#ec2-launch-instance>`__.

-  It is recommended to use c5.4xlarge or larger. For this example we
   will use a c5.4xlarge.
-  If you would like to compile and infer on the same machine, please
   select inf1.6xlarge.

1.3. Select and launch an Inf1 instance of your choice to run the
compiled model. Launch an instance by following `EC2
instructions <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html#ec2-launch-instance>`__.

Step 2: Install MXNet-Neuron and Neuron Compiler On Compilation Instance
------------------------------------------------------------------------

If using Conda DLAMI version 26 and up, activate pre-installed
MXNet-Neuron environment (using ``source activate aws_neuron_mxnet_p36``
command). Please update Neuron by following update steps in
:ref:`dlami-neuron-rn`.

To install in your own AMI, please see :ref:`neuron-install-guide` to
setup virtual environment and install MXNet-Neuron (mxnet-neuron) and
Neuron Compiler (neuron-cc) packages.

Step 3: Compile on Compilation Server
-------------------------------------

A trained model must be compiled to Inferentia target before it can run
on Inferentia. In this step we compile a pre-trained ResNet50 and export
it as a compiled MXNet checkpoint.

3.1. Create a file ``compile_resnet50.py`` with the content below and
run it using ``python compile_resnet50.py``. Compilation will take a few
minutes on c5.4xlarge. At the end of compilation, the files
``resnet-50_compiled-0000.params`` and
``resnet-50_compiled-symbol.json`` will be created in local directory.

.. code:: python

   import mxnet as mx
   import numpy as np

   path='http://data.mxnet.io/models/imagenet/'
   mx.test_utils.download(path+'resnet/50-layers/resnet-50-0000.params')
   mx.test_utils.download(path+'resnet/50-layers/resnet-50-symbol.json')
   sym, args, aux = mx.model.load_checkpoint('resnet-50', 0)

   # Compile for Inferentia using Neuron
   inputs = { "data" : mx.nd.ones([1,3,224,224], name='data', dtype='float32') }
   sym, args, aux = mx.contrib.neuron.compile(sym, args, aux, inputs)

   #save compiled model
   mx.model.save_checkpoint("resnet-50_compiled", 0, sym, args, aux)

3.2. If not compiling and inferring on the same instance, copy the
artifact to the inference server (use ec2-user as user for AML2):

.. code:: bash

   scp -i <PEM key file>  resnet-50_compiled-0000.params ubuntu@<instance DNS>:~/  # Ubuntu
   scp -i <PEM key file>  resnet-50_compiled-symbol.json ubuntu@<instance DNS>:~/  # Ubuntu

Step 4: Install MXNet-Neuron and Neuron Runtime on Inference Instance
---------------------------------------------------------------------

**If using DLAMI, activate pre-installed MXNet-Neuron environment (using
\``source activate aws_neuron_mxnet_p36`\` command) and skip this
step.**

On the instance you are going to use for inference, install
TensorFlow-Neuron and Neuron Runtime.

4.1. Follow Step 2 above to install MXNet-Neuron.

-  Install neuron-cc if compilation on inference instance is desired
   (see notes above on recommended Inf1 sizes for compilation)
-  Skip neuron-cc if compilation is not done on inference instance

4.2. To install Neuron Runtime, see :ref:`rtd-getting-started`.

Step 5: Execute inference on Inf1
---------------------------------

In this step we run inference on Inf1 using the model compiled in Step
3.

5.1. On the Inf1, create a inference Python script named
``infer_resnet50.py`` with the following content:

.. code:: python

   import mxnet as mx
   import numpy as np

   path='http://data.mxnet.io/models/imagenet/'
   mx.test_utils.download(path+'synset.txt')

   fname = mx.test_utils.download('https://raw.githubusercontent.com/awslabs/mxnet-model-server/master/docs/images/kitten_small.jpg?raw=true')
   img = mx.image.imread(fname)# convert into format (batch, RGB, width, height)
   img = mx.image.imresize(img, 224, 224) # resize
   img = img.transpose((2, 0, 1)) # Channel first
   img = img.expand_dims(axis=0) # batchify
   img = img.astype(dtype='float32')

   sym, args, aux = mx.model.load_checkpoint('resnet-50_compiled', 0)
   softmax = mx.nd.random_normal(shape=(1,))
   args['softmax_label'] = softmax
   args['data'] = img

   # Inferentia context
   ctx = mx.neuron()

   exe = sym.bind(ctx=ctx, args=args, aux_states=aux, grad_req='null')

   with open('synset.txt', 'r') as f:
        labels = [l.rstrip() for l in f]

   exe.forward(data=img)
   prob = exe.outputs[0].asnumpy()# print the top-5
   prob = np.squeeze(prob)
   a = np.argsort(prob)[::-1]
   for i in a[0:5]:
        print('probability=%f, class=%s' %(prob[i], labels[i]))

5.2. Run the script to see inference results:

.. code:: bash

   python infer_resnet50.py

.. code:: bash

   probability=0.642454, class=n02123045 tabby, tabby cat
   probability=0.189407, class=n02123159 tiger cat
   probability=0.100798, class=n02124075 Egyptian cat
   probability=0.030649, class=n02127052 lynx, catamount
   probability=0.016278, class=n02129604 tiger, Panthera tigris
