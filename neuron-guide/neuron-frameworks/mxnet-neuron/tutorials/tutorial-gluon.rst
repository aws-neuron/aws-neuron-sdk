.. _mxnet-gluon-tutorial:

MXNet 1.8: Getting Started with Gluon Tutorial 
==============================================

.. contents:: Table of Contents
   :local:
   :depth: 2


Overview:
---------
In this tutorial you will compile and deploy resnet-50 using the newly supported MXNet 1.8 and Gluon API
on an Inf1 instance. This tutorial is only supported with MXNet 1.8.

Run The Tutorial
----------------

A trained model must be compiled to Inferentia target before it can run
on Inferentia. In this step we compile a pre-trained ResNet50 and export
it as a compiled MXNet checkpoint.

1. Create a file ``compile_resnet50.py`` with the content below and
   run it using ``python compile_resnet50.py``. Compilation will take a few
   minutes on c5.4xlarge. At the end of compilation, the files
   ``resnet-50_compiled-0000.params`` and
   ``resnet-50_compiled-symbol.json`` will be created in local directory.

.. code:: python

    import mxnet as mx
    import mx_neuron as neuron 
    import numpy as np

    path='http://data.mxnet.io/models/imagenet/'
    mx.test_utils.download(path+'resnet/50-layers/resnet-50-0000.params')
    mx.test_utils.download(path+'resnet/50-layers/resnet-50-symbol.json')
    block = mx.gluon.nn.SymbolBlock.imports('resnet-50-symbol.json',\
        ['data', 'softmax_label'], 'resnet-50-0000.params', ctx=mx.cpu())

    block.hybridize() 

    # Compile for Inferentia using Neuron
    inputs = { "data" : mx.nd.ones([1,3,224,224], name='data', dtype='float32'), 'softmax_label' : mx.nd.ones([1], name='data', dtype='float32') }
    block = neuron.compile(block, inputs=inputs)

    #save compiled model
    block.export("resnet-50_compiled", 0, block)


- To check the supported operations for the uncompiled model or information on Neuron subgraphs for the compiled model, please see :ref:`neuron_check_model`.

2. Create an inference Python script named ``infer_resnet50.py`` with the following content: 

.. code:: python 

    import numpy as np
    import mxnet as mx 
    import mx_neuron as neuron 

    path='http://data.mxnet.io/models/imagenet/'
    mx.test_utils.download(path+'synset.txt')

    fname = mx.test_utils.download('https://raw.githubusercontent.com/awslabs/mxnet-model-server/master/docs/images/kitten_small.jpg?raw=true')
    img = mx.image.imread(fname)# convert into format (batch, RGB, width, height)
    img = mx.image.imresize(img, 224, 224) # resize
    img = img.transpose((2, 0, 1)) # Channel first
    img = img.expand_dims(axis=0) # batchify
    img = img.astype(dtype='float32')

    block = mx.gluon.nn.SymbolBlock.imports('resnet-50_compiled-symbol.json',\
        ['data', 'softmax_label'], 'resnet-50_compiled-0000.params', ctx=mx.cpu())
    softmax = mx.nd.random_normal(shape=(1,))

    out = block(img, softmax).asnumpy()

    with open('synset.txt', 'r') as f:
        labels = [l.rstrip() for l in f]

    out = block(img, softmax).asnumpy()

    prob = np.squeeze(out)
    a = np.argsort(prob)[::-1]
    for i in a[0:5]:
        print('probability=%f, class=%s' %(prob[i], labels[i]))


3. Run the script to see the inference results:

.. code:: bash 

    python infer_resnet50.py 

.. code:: bash 

    probability=0.643591, class=n02123045 tabby, tabby cat
    probability=0.184392, class=n02123159 tiger cat
    probability=0.105063, class=n02124075 Egyptian cat
    probability=0.030101, class=n02127052 lynx, catamount
    probability=0.016112, class=n02129604 tiger, Panthera tigris

