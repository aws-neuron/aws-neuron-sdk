# Tutorial: MXNet Configurations for NeuronCore Groups

A NeuronCore Group is a set of NeuronCores that are used to load and run compiled models with one-to-one mapping; At any time, one model will be running in a NeuronCore Group. With NeuronCore groups a user may load independent models in parallel to execute. Additionally, within a NeuronCore Group, loaded models can be dynamically started and stopped, allowing for dynamic context switching from one model to another.

To explicitly specify the NeuronCore Groups, set environment variable `NEURONCORE_GROUP_SIZES` to a list of group sizes. The consecutive NeuronCore groups will be created by Neuron-RTD and be available to map the models to.

Note that in order to map a model to a group, the model must be compiled to fit within the group size. To limit the number of NeuronCores during compilation, use compiler_args dictionary with field “--num-neuroncores“ set to the group size:

```
compile_args = {'--num-neuroncores' : 2}
sym, args, auxs = neuron.compile(sym, args, auxs, inputs, **compile_args)
```

Before starting this example, please ensure that mxnet-neuron is installed along with the Neuron Compiler (see [MXNet Tutorial](./tutorial-compile-infer.md)) and Neuron runtime is running with default settings (see [Neuron runtime getting started](./../neuron-runtime/nrt_start.md) ).

## Compile Model

Model must be compiled to Inferentia target before it can be used on Inferentia.

Create compile_resnet50.py with `--num-neuroncores` set to 2 and run it. The files `resnet-50_compiled-0000.params` and `resnet-50_compiled-symbol.json` will be created in local directory:

```python
import mxnet as mx
import numpy as np

path='http://data.mxnet.io/models/imagenet/'
mx.test_utils.download(path+'resnet/50-layers/resnet-50-0000.params')
mx.test_utils.download(path+'resnet/50-layers/resnet-50-symbol.json')
sym, args, aux = mx.model.load_checkpoint('resnet-50', 0)

# Compile for Inferentia using Neuron, fit to NeuronCore group size of 2
inputs = { "data" : mx.nd.ones([1,3,224,224], name='data', dtype='float32') }
compile_args = {'--num-neuroncores' : 2}
sym, args, aux = mx.contrib.neuron.compile(sym, args, aux, inputs, **compile_args)

#save compiled model
mx.model.save_checkpoint("resnet-50_compiled", 0, sym, args, aux)

```

## Run Inference

During inference, to subdivide the pool of a single Inferentia chip into three groups of 1, 2, and 1 NeuronCores, specify `NEURONCORE_GROUP_SIZES` as follows:

```bash
NEURONCORE_GROUP_SIZES='[1,2,1]' <launch process>`
```

Within the framework, the model can be mapped to group using  `ctx=mx.neuron(N)` context where N is the group index within the `NEURONCORE_GROUP_SIZES` list.

Create infer_resnet50.py with the following content:

```python
import mxnet as mx
import numpy as np

path='http://data.mxnet.io/models/imagenet/'
mx.test_utils.download(path+'synset.txt')

fname = mx.test_utils.download('https://raw.githubusercontent.com/awslabs/mxnet-model-server/master/docs/images/kitten_small.jpg?raw=true')
img = mx.image.imread(fname) # convert into format (batch, RGB, width, height)
img = mx.image.imresize(img, 224, 224) # resize
img = img.transpose((2, 0, 1)) # Channel first
img = img.expand_dims(axis=0) # batchify
img = img.astype(dtype='float32')

sym, args, aux = mx.model.load_checkpoint('resnet-50_compiled', 0)
softmax = mx.nd.random_normal(shape=(1,))
args['softmax_label'] = softmax
args['data'] = img

# Inferentia context - group index 1 (size 2) in NEURONCORE_GROUP_SIZES=[1,2,1]
ctx = mx.neuron(1)

exe = sym.bind(ctx=ctx, args=args, aux_states=aux, grad_req='null')

with open('synset.txt', 'r') as f:
     labels = [l.rstrip() for l in f]

exe.forward(data=img)
prob = exe.outputs[0].asnumpy()# print the top-5
prob = np.squeeze(prob)
a = np.argsort(prob)[::-1]
for i in a[0:5]:
     print('probability=%f, class=%s' %(prob[i], labels[i]))
```

Run the script to see inference results using NeuronCore group 1:

```bash
NEURONCORE_GROUP_SIZES='[1,2,1]' python infer_resnet50.py
```

```bash
probability=0.646784, class=n02123045 tabby, tabby cat
probability=0.185307, class=n02123159 tiger cat
probability=0.099188, class=n02124075 Egyptian cat
probability=0.032201, class=n02127052 lynx, catamount
probability=0.016192, class=n02129604 tiger, Panthera tigris
```

If not enough NeuronCores are provided, an error message will be displayed:

```bash
NEURONCORE_GROUP_SIZES='[1,1,1]' python infer_resnet50.py
```

```bash
...
mxnet.base.MXNetError: [04:01:39] src/operator/subgraph/neuron/./neuron_util.h:541: Check failed: rsp.status().code() == 0: Failed load model with Neuron-RTD Error. Neuron-RTD Status Code: 9, details: ""
```
