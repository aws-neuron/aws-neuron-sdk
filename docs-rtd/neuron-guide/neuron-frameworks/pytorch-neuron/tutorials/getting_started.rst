
.. _pytorch-getting-started:

Getting started with torch-neuron (resnet-50 tutorial)
======================================================

Introduction
------------

A typical workflow with the Neuron SDK will be to compile trained ML
models on a general compute instance (the compilation instance), and
then distribute the artifacts to a fleet of Inf1 instances (the
deployment instances) for inference execution. Neuron enables PyTorch
for both of these steps.

Steps Overview:
---------------

1. Launch an EC2 compilation instance (recommended instance: c5.4xlarge
   or larger)
2. Install Torch-Neuron and Neuron-Compiler on the compilation instance
3. Compile the compute-graph on the compilation-instance, and copy the
   artifacts into the deployment-instance
4. Install Torch-Neuron and Neuron-Runtime on an Inf1 instance
   (deployment Instance)
5. Run inference on the Inf1 instance

Additionally we'll cover the following topics as we progress:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. How do I analyze a model for use with AWS Neuron?
2. How do I make sure that my model is using all of the available
   neuron-cores?
3. How can I choose my input shapes to maximise the throughput of my
   model on neuron hardware?

Step 1: Launch EC2 compilation instance
---------------------------------------

1.1. Select an AMI of your choice.

Refer to the :ref:`neuron-install-guide`  for details.

1.2. Select and launch an EC2 instance

-  A c5.4xlarge or larger is recommended. For this example we will use a
   c5.4xlarge.
-  Users may choose to compile and deploy on the same instance, in which
   case an inf1.6xlarge instance or larger is recommended. If you choose
   “launch instance” and search for “neuron” in the AWS EC2 console you
   will see a short list of DLAMI images to select from. If you choose a
   DLAMI, make sure to check the Neuron version installed, and upgrade
   as needed.

1.3. Select and launch a deployment (Inf1) instance of your choice (if
not compiling and inferencing on the same instance). Launch an instance
by following EC2 instructions.

Step 2: Compilation instance installations
------------------------------------------

If using Conda DLAMI, ensure you use the latest version, you can
activate pre-installed PyTorch-Neuron environment (using
``source activate aws_neuron_pytorch_p36`` command).

To install in your own AMI, please see :ref:`neuron-install-guide` to setup virtual
environment and install the latest Torch-Neuron packages.

Step 3: Compile on compilation instance
---------------------------------------

A trained model must be compiled to Inferentia target before it can be
deployed on Inf1 instances. In this step we compile the torchvision
ResNet50 model and export it as a saved TorchScript module.

If you are familiar with running Jupyter notebooks these compile steps
are reproduced :pytorch-neuron-src:`here <getting_started_compile.ipynb>`

3.1. Install torchvision

.. code:: bash

   pip install torchvision==0.6.1

3.2. Create a python script named ``trace_resnet50.py`` with the
following content:

.. code:: python

   import torch
   import numpy as np
   import os
   import torch_neuron
   from torchvision import models
   import logging

   ## Enable logging so we can see any important warnings
   logger = logging.getLogger('Neuron')
   logger.setLevel(logging.INFO)

   image = torch.zeros([1, 3, 224, 224], dtype=torch.float32)

   ## Load a pretrained ResNet50 model
   model = models.resnet50(pretrained=True)

   ## Tell the model we are using it for evaluation (not training)
   model.eval()

   ## Analyze the model - this will show operator support and operator count
   torch.neuron.analyze_model( model, example_inputs=[image] )

   ## Now compile the model - with logging set to "info" we will see
   ## what compiles for Neuron, and if there are any fallbacks
   ## Note: The "-O2" setting is default in recent releases, but may be needed for DLAMI
   ##       and older installed environments
   model_neuron = torch.neuron.trace(model, example_inputs=[image], compiler_args="-O2")

   ## Export to saved model
   model_neuron.save("resnet50_neuron.pt")

3.3. Run the compilation script, which will take a few minutes on
c5.4xlarge. At the end of script execution, the compiled model is saved
as ``resnet50_neuron.pt`` in local directory:

.. code:: bash

   python trace_resnet50.py

You should see (indicative output only):

::

   INFO:Neuron:The following operations are currently supported in torch-neuron for this model:
   INFO:Neuron:aten::relu
   INFO:Neuron:aten::flatten
   INFO:Neuron:aten::t
   INFO:Neuron:aten::max_pool2d
   INFO:Neuron:aten::add
   INFO:Neuron:aten::addmm
   INFO:Neuron:aten::_convolution
   INFO:Neuron:aten::batch_norm
   INFO:Neuron:aten::adaptive_avg_pool2d
   INFO:Neuron:prim::ListConstruct
   INFO:Neuron:prim::Constant
   INFO:Neuron:100.00% of all operations (including primitives) (1645 of 1645) are supported
   INFO:Neuron:100.00% of arithmetic operations (176 of 176) are supported
   OrderedDict([('percent_supported', 100.0), ('percent_supported_arithmetic', 100.0), ('supported_count', 1645), ('total_count', 1645), ('supported_count_arithmetic', 176), ('total_count_arithmetic', 176), ('supported_operators', {'aten::relu', 'aten::flatten', 'aten::t', 'aten::max_pool2d', 'aten::add', 'aten::addmm', 'aten::_convolution', 'aten::batch_norm', 'aten::adaptive_avg_pool2d', 'prim::ListConstruct', 'prim::Constant'}), ('unsupported_operators', []), ('operators', ['aten::_convolution', 'aten::adaptive_avg_pool2d', 'aten::add', 'aten::addmm', 'aten::batch_norm', 'aten::flatten', 'aten::max_pool2d', 'aten::relu', 'aten::t', 'prim::Constant', 'prim::ListConstruct']), ('operator_count', OrderedDict([('aten::_convolution', 53), ('aten::adaptive_avg_pool2d', 1), ('aten::add', 16), ('aten::addmm', 1), ('aten::batch_norm', 53), ('aten::flatten', 1), ('aten::max_pool2d', 1), ('aten::relu', 49), ('aten::t', 1), ('prim::Constant', 1252), ('prim::ListConstruct', 217)]))])
   INFO:Neuron:Number of arithmetic operators (pre-compilation) before = 176, fused = 176, percent fused = 100.0%
   INFO:Neuron:compiling function _NeuronGraph$1108 with neuron-cc
   INFO:Neuron:Compiling with command line: '/home/ubuntu/test_beta_env/bin/neuron-cc compile /tmp/tmp2fisdcmu/graph_def.pb --framework TENSORFLOW --pipeline compile SaveTemps --output /tmp/tmp2fisdcmu/graph_def.neff --io-config {"inputs": {"0:0": [[1, 3, 224, 224], "float32"]}, "outputs": ["Add_69:0"]}''
   INFO:Neuron:Number of arithmetic operators (post-compilation) before = 176, compiled = 176, percent compiled = 100.0%

3.4. WARNING: If you run the inference script (in section 4 below) on
your CPU instance you will get output, but see the following warning.

::

   [E neuron_op_impl.cpp:53] Warning: Tensor output are *** NOT CALCULATED *** during CPU
   execution and only indicate tensor shape

The warning is also displayed during trace (where it is expected). This
is an artifact of the way we trace a model on your compile instance. Do
not perform inference with a neuron traced model on a non neuron
supported instance, results will not be calculated.

3.5. If not compiling and inferring on the same instance, copy the
compiled artifacts to the inference server:

::

   scp -i <PEM key file>  ./resnet50_neuron.pt ubuntu@<instance DNS>:~/ # if Ubuntu-based AMI
   scp -i <PEM key file>  ./resnet50_neuron.pt ec2-user@<instance DNS>:~/  # if using AML2-based AMI

Step 4: Deployment Instance Installations
-----------------------------------------

On the instance you are going to use for inference, install Torch-Neuron
and Neuron Runtime

4.1. Follow Step 2 above to install Torch-Neuron.

-  Install neuron-cc[tensorflow] if compilation on inference instance is
   desired (see notes above on recommended Inf1 sizes for compilation)
-  Skip neuron-cc if compilation is not done on inference instance

4.2. Install the Neuron Runtime using instructions from :ref:`rtd-getting-started`.

Step 5: Run inference
---------------------

In this step we run inference on an Inf1 instance using the model
compiled in Step 3. Initially we will just use one of the available
neuron cores.

5.1. On the Inf1, create a inference Python script named
``infer_resnet50.py`` with the following content:

.. code:: python

   import os
   import time
   import torch
   import torch_neuron
   import json
   import numpy as np
   from urllib import request
   from torchvision import models, transforms, datasets
   from time import time

   ## Create an image directory containing a small kitten
   os.makedirs("./torch_neuron_test/images", exist_ok=True)
   request.urlretrieve("https://raw.githubusercontent.com/awslabs/mxnet-model-server/master/docs/images/kitten_small.jpg",
    "./torch_neuron_test/images/kitten_small.jpg")

   ## Fetch labels to output the top classifications
   request.urlretrieve("https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json","imagenet_class_index.json")
   idx2label = []

   with open("imagenet_class_index.json", "r") as read_file:
    class_idx = json.load(read_file)
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]

   ## Import a sample image and normalize it into a tensor
   normalize = transforms.Normalize(
       mean=[0.485, 0.456, 0.406],
       std=[0.229, 0.224, 0.225])

   eval_dataset = datasets.ImageFolder(
       os.path.dirname("./torch_neuron_test/"),
       transforms.Compose([
           transforms.Resize([224, 224]),
           transforms.ToTensor(),
           normalize,
       ])
   )
   image, _ = eval_dataset[0]
   image = torch.tensor(image.numpy()[np.newaxis, ...])

   ## Load model
   model_neuron = torch.jit.load( 'resnet50_neuron.pt' )

   ## Since the first inference also load the model let's exclude it 
   ## from timing
   results = model_neuron( image )

   ## Predict for 100 loops
   start = time()

   loops = 100
   for _ in range(loops):
       results = model_neuron( image )
   elapsed_time = time() - start
   images_sec = loops / float(elapsed_time)

   # Get the top 5 results
   top5_idx = results[0].sort()[1][-5:]

   # Lookup and print the top 5 labels
   top5_labels = [idx2label[idx] for idx in top5_idx]

   print("Top 5 labels:\n {}".format(top5_labels) )
   print("Completed {} operations in {} seconds => {} images / second".format(loops, round(elapsed_time,2), round(images_sec,0) ) )

5.2. Run the inference

::

   Top 5 labels:
    ['tiger', 'lynx', 'tiger_cat', 'Egyptian_cat', 'tabby']
   Completed 100 operations in 0.37 seconds => 267.0 images / second

Step 6: Run on parallel neuron cores
------------------------------------

To fully leverage the inferentia hardware we want to use all the cores.
On an inf1.xlarge or inf1.2xlarge we have four available cores, with 16
cores on inf1.6xlarge and inf1.24xlarge instances. Here we use the
futures library to create a simple class that runs four parallel
inference threads

Using all of the available cores is important for achieving maximum
performance on Neuron hardware. The implementation below uses an
aggregated batch size.

-  It loads the model into four cores
-  At input it accepts a batch four times the size of the compiled model
-  It splits the data across the four cores, and once all cores are done
   collates the output into a result tensor

This is intended as a good starting implementation - but you may want to
vary it depending on your application

6.1 Create a data parallel class which handles larger tensor batches

.. code:: python

   from concurrent import futures
   import torch
   import torch.neuron
   import os

   class NeuronSimpleDataParallel():

       def __init__(self, model_file, num_neuron_cores, batch_size=1):
           # Construct a list of models
           self.num_neuron_cores = num_neuron_cores
           self.batch_size = batch_size

           class SimpleWrapper():

               def __init__(self, model):
                   self.model = model

               def eval(self):
                   self.model.eval()

               def train(self):
                   self.model.train()

               def __call__(self, *args):
                   results = self.model(*args)

                   # Make the output iterable - if it is not already a tuple or list
                   if not isinstance(results, tuple) or isinstance(results, list):
                       results = [results]

                   return results

           self.models = [SimpleWrapper(torch.jit.load(model_file))
                          for i in range(num_neuron_cores)]

           ## Important - please read:
           ##     https://github.com/aws/aws-neuron-sdk/blob/master/docs/tensorflow-neuron/tutorial-NeuronCore-Group.md
           ## For four cores we use 
           ##     os.environ['NEURONCORE_GROUP_SIZES'] = "1,1,1,1" 
           ## when launching four threads
           ## In this logic exists in worker processes, each process should use 
           ##     os.environ['NEURONCORE_GROUP_SIZES'] = "1"
           nc_env = ','.join(['1'] * num_neuron_cores)
           os.environ['NEURONCORE_GROUP_SIZES'] = nc_env

           self.executor = futures.ThreadPoolExecutor(
               max_workers=self.num_neuron_cores)

       def eval(self):
           for m in self.models:
               m.eval()

       def train(self):
           for m in self.models:
               m.train()

       def __call__(self, *args):
           assert all(isinstance(a, torch.Tensor)
                      for a in args), "Non tensor input - tensors are needed to generate batches"
           assert all(a.shape[0] % self.num_neuron_cores ==
                      0 for a in args), "Batch size must be even multiple of the number of parallel neuron cores"

           args_per_core = [[] for i in range(self.num_neuron_cores)]

           # Split args
           for a in args:
               # Based on batch size for arg
               step_size = a.shape[0] // self.num_neuron_cores
               for i in range(self.num_neuron_cores):
                   # Append a slice of a view
                   start = i * step_size
                   end = (i + 1) * step_size

                   # Slice
                   args_per_core[i].append(a[start:end])

           # Call each core with their split and wait to complete
           running = {self.executor.submit(
               self.models[idx], *args_per_core[idx]): idx for idx in range(self.num_neuron_cores)}

           results = []

           for future in futures.as_completed(running):
               running[future]

               # Expect a tuple of tensors - convert to a list of tensors
               results.append(future.result())

           # Remove zero dimensional tensors (unsqueeze)
           # Iterate results per core
           for ic in range(len(results)):
               # Iterate result tuples
               for ir in range(len(results[ic])):
                   # Unsqueeze if zero dimensional or does not look batched (i.e. first dim does not match batch)
                   if len(results[ic][ir].size()) == 0 or results[ic][ir].shape[0] != self.batch_size:
                       results[ic][ir] = torch.unsqueeze(
                           results[ic][ir], 0)

           # Concatenate
           output = results[0][0]

           for i in range(1, len(results)):
               for j in range(len(results[i])):
                   output = torch.cat([output, results[i][j]], 0)

           return output

Save the code above to “parallel.py”

6.2 Now we can update our inference code for four cores (additions are
shown below in orange):

.. code:: python

   import os
   from time import time
   import torch
   import torch_neuron
   import json
   import numpy as np
   from urllib import request
   from torchvision import models, transforms, datasets
   from parallel import NeuronSimpleDataParallel

   ## Assuming you are working on and inf1.xlarge or inf1.2xlarge
   num_neuron_cores = 4

   ## Create an image directory containing a small kitten
   os.makedirs("./torch_neuron_test/images", exist_ok=True)
   request.urlretrieve("https://raw.githubusercontent.com/awslabs/mxnet-model-server/master/docs/images/kitten_small.jpg",
                       "./torch_neuron_test/images/kitten_small.jpg")

   ## Fetch labels to output the top classifications
   request.urlretrieve("https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json","imagenet_class_index.json")
   idx2label = []

   with open("imagenet_class_index.json", "r") as read_file:
       class_idx = json.load(read_file)
       idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
       
   ## Import a sample image and normalize it into a tensor
   normalize = transforms.Normalize(
       mean=[0.485, 0.456, 0.406],
       std=[0.229, 0.224, 0.225])
       
   eval_dataset = datasets.ImageFolder(
       os.path.dirname("./torch_neuron_test/"),
       transforms.Compose([
       transforms.Resize([224, 224]),
       transforms.ToTensor(),
       normalize,
       ])
   )
   image, _ = eval_dataset[0]
   image = torch.tensor(image.numpy()[np.newaxis, ...])

   ## Load model
   model_neuron = NeuronSimpleDataParallel( 'resnet50_neuron.pt', num_neuron_cores )

   ## Create a "batch" image with enough images to go on each of the four cores
   batch_image = image

   for i in range(num_neuron_cores - 1):
       batch_image = torch.cat( [batch_image, image], 0 )

   print(batch_image.shape)

   ## Since the first inference also loads the model to the chip let's exclude it 
   ## from timing
   results = model_neuron( batch_image )

   ## Predict
   loops = 100
   start = time()
   for _ in range(loops):
       results = model_neuron( batch_image )
   elapsed_time = time() - start
   images_sec = loops * batch_image.size(0) / float(elapsed_time)

   # Get the top 5 results
   top5_idx = results[0].sort()[1][-5:]

   # Lookup and print the top 5 labels
   top5_labels = [idx2label[idx] for idx in top5_idx]
   print("Top 5 labels:\n {}".format(top5_labels) )
   print("Completed {} operations in {} seconds => {} images / second".format(loops * batch_image.size(0), round(elapsed_time,2), round(images_sec,0) ) )

6.3 Run the inference: Sample output

::

   Top 5 labels:
    ['tiger', 'lynx', 'tiger_cat', 'Egyptian_cat', 'tabby']
   Completed 400 operations in 0.86 seconds => 466.0 images / second

Step 7: Experiment with different batch sizes:
----------------------------------------------

Different models will show better and worse throughput with different
batch sizes. In general neuron models will work best with small batch
sizes when compared with GPU inference - even though overall a single
neuron instance may outperform a GPU instance on a given task.

As a general best practice we recommend starting with a small batch size
and working up to find peak throughput.

Now that we are using all four cores we can experiment with compiling
and running larger batch sizes on each of our four cores

7.1 Modify the training code Here we use a batch size of 5 - but you can
use any value, or test multiple. Changes in orange

.. code:: python

   import torch
   import numpy as np
   import os
   import torch_neuron
   from torchvision import models
   import logging

   ## Enable logging so we can see any important warnings
   logger = logging.getLogger('Neuron')
   logger.setLevel(logging.INFO)

   batch_size = 5

   image = torch.zeros([batch_size, 3, 224, 224], dtype=torch.float32)

   ## Load a pretrained ResNet50 model
   model = models.resnet50(pretrained=True)

   ## Tell the model we are using it for evaluation (not training)
   model.eval()

   ## Analyze the model - this will show operator support and operator count
   analyze_results = torch.neuron.analyze_model( model, example_inputs=[image] )

   print(analyze_results)

   ## Now compile the model
   ## Note: The "-O2" setting is default in recent releases, but may be needed for DLAMI
   ##       and older installed environments
   model_neuron = torch.neuron.trace(model, example_inputs=[image], compiler_args="-O2")

   ## Export to saved model
   model_neuron.save("resnet50_neuron_b{}.pt".format(batch_size))

7.2 Modify the inference code

.. code:: python

   import os
   from time import time
   import torch
   import torch_neuron
   import json
   import numpy as np
   from urllib import request
   from torchvision import models, transforms, datasets
   from parallel import NeuronSimpleDataParallel

   ## Assuming you are working on and inf1.xlarge or inf1.2xlarge
   num_neuron_cores = 4
   batch_size = 5

   ## Create an image directory containing a small kitten
   os.makedirs("./torch_neuron_test/images", exist_ok=True)
   request.urlretrieve("https://raw.githubusercontent.com/awslabs/mxnet-model-server/master/docs/images/kitten_small.jpg",
                       "./torch_neuron_test/images/kitten_small.jpg")

   ## Fetch labels to output the top classifications
   request.urlretrieve("https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json","imagenet_class_index.json")
   idx2label = []

   with open("imagenet_class_index.json", "r") as read_file:
       class_idx = json.load(read_file)
       idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
       
   ## Import a sample image and normalize it into a tensor
   normalize = transforms.Normalize(
       mean=[0.485, 0.456, 0.406],
       std=[0.229, 0.224, 0.225])
       
   eval_dataset = datasets.ImageFolder(
       os.path.dirname("./torch_neuron_test/"),
       transforms.Compose([
       transforms.Resize([224, 224]),
       transforms.ToTensor(),
       normalize,
       ])
   )
   image, _ = eval_dataset[0]
   image = torch.tensor(image.numpy()[np.newaxis, ...])

   ## Load model
   model_neuron = NeuronSimpleDataParallel( 'resnet50_neuron_b{}.pt'.format(batch_size), num_neuron_cores, batch_size=batch_size )

   ## Create a "batch" image with enough images to go on each of the four cores
   batch_image = image

   for i in range((num_neuron_cores * batch_size) - 1):
       batch_image = torch.cat( [batch_image, image], 0 )

   ## Since the first inference also loads the model to the chip let's exclude it 
   ## from timing
   results = model_neuron( batch_image )

   ## Predict
   start = time()
   loops = 100
   for _ in range(loops):
       results = model_neuron( batch_image )
   elapsed_time = time() - start
   images_sec = loops * batch_image.size(0) / elapsed_time

   # Get the top 5 results
   top5_idx = results[0].sort()[1][-5:]

   # Lookup and print the top 5 labels
   top5_labels = [idx2label[idx] for idx in top5_idx]
   print("Top 5 labels:\n {}".format(top5_labels) )
   print("Completed {} operations in {} seconds => {} images / second".format( 
       loops * batch_image.size(0), round(elapsed_time, 2), round(images_sec,0) ) )

7.2 Run the inference Sample output

::

   Top 5 labels:
    ['tiger', 'lynx', 'tiger_cat', 'Egyptian_cat', 'tabby']
   Completed 2000 operations in 3.19 seconds => 626.0 images / second

You can experiment with different batch size values to see what gives
the best overall throughput

Step 8: Terminate instances
---------------------------

Don’t forget to terminate your instances (compile and inference) from
the AWS console so that you don’t continue paying for them once you are
done
