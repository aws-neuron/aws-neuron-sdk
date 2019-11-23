# Tutorial: MXNet Configurations for NeuronCore Groups

To further subdivide the pool of NeuronCores controled by a Neuron-RTD, specify the NeuronCore Groups within that pool using the environment variable `NEURONCORE_GROUP_SIZES`  set to list of group sizes. The consecutive NeuronCore groups will be created by Neuron-RTD and be available for use to map the models. 

Note that to map a model to a group, the model must be compiled to fit within the group size. To limit the number of NeuronCores during compilation, use compiler_args dictionary with field “--num-neuroncores“ set to the group size:

```
compile_args = {'--num-neuroncores' : 2}
sym, args, auxs = neuron.compile(sym, args, auxs, inputs, **compile_args)
```

Following this example: [MXNet tutorial](./tutorial-mxnet-neuron-compile-infer.md), create compile_resnet50.py with `--num-neuroncores` set to 2 and run it:

```
import mxnet as mx
import numpy as np

path='http://data.mxnet.io/models/imagenet/‘
mx.test_utils.download(path+'resnet/50-layers/resnet-50-0000.params')
mx.test_utils.download(path+'resnet/50-layers/resnet-50-symbol.json')

sym, args, aux = mx.model.load_checkpoint('resnet-50', 0)

# Compile for Inferentia using Neuron, fit to NeuronCore group size of 2
inputs = { "data" : mx.nd.ones([1,3,224,224], name='data', dtype='float32') }
compile_args = {'--num-neuroncores' : 2}
sym, args, aux = mx.contrib.neuron.compile(sym, args, aux, inputs, **compile_args)

#save compiled model
mx.model.save_checkpoint("compiled_resnet50", 0, sym, args, aux)
```

During inference, to subdivide the pool of one Inferentia into groups of 1, 2, and 1 NeuronCores, specify `NEURONCORE_GROUP_SIZES` as follows:

`NEURONCORE_GROUP_SIZES``='[1,2,1]' <launch process>`

Within the framework, the model can be mapped to group using  `ctx=mx.neuron(N)` context where N is the group index within the `NEURONCORE_GROUP_SIZES` list. Following the example here,  [MXNet Tutorial](./[tutorial-mxnet-neuron-compile-infer.md](https://github.com/aws/aws-neuron-sdk/blob/master/docs/tutorial-mxnet-neuron-compile-infer.md)) , create infer_resnet50.py with the following content:

```
import mxnet as mx
import numpy as np

path='http://data.mxnet.io/models/imagenet/‘
mx.test_utils.download(path+'synset.txt')

fname = mx.test_utils.download('https://github.com/dmlc/web-data/blob/master/mxnet/doc/tutorials/python/predict_image/cat.jpg?raw=true')
img = mx.image.imread(fname)# convert into format (batch, RGB, width, height)
img = mx.image.imresize(img, 224, 224) # resize
img = img.transpose((2, 0, 1)) # Channel first
img = img.expand_dims(axis=0) # batchify
img = img.astype(dtype='float32')

sym, args, aux = mx.model.load_checkpoint('compiled_resnet50', 0)
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

```
`NEURONCORE_GROUP_SIZES``=``'[1,2,1]'`` `python infer_resnet50.py

probability=0.379626, class=n02119789 kit fox, Vulpes macrotis
probability=0.290867, class=n02119022 red fox, Vulpes vulpes
probability=0.034885, class=n02124075 Egyptian cat
probability=0.028950, class=n02085620 Chihuahua
probability=0.027466, class=n02120505 grey fox, gray fox, Urocyon cinereoargenteus
```



