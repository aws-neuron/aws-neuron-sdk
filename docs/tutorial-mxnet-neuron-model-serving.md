# Tutorial: MXNet-Neuron Model Serving

The Neuron MXNet Model Serving (MMS) example is adapted from the MXNet vision service example which uses pretrained squeezenet to perform image classification: https://github.com/awslabs/mxnet-model-server/tree/master/examples/mxnet_vision. Before starting this example, please ensure that Neuron-optimized MXNet version mxnet-neuron is installed [link to Tutorial MXNet] and Neuron RTD is running with default settings (see Getting Started guide[TODO ADD LINK]) .


1. First, install mxnet-model-server, download the example code, and download the pre-trained Squeezenet model:

```
cd ~/
pip install mxnet-model-server
git clone https://github.com/awslabs/mxnet-model-server
cd ~/mxnet-model-server/examples/mxnet_vision
curl -O https://s3.amazonaws.com/model-server/model_archive_1.0/examples/squeezenet_v1.1/squeezenet_v1.1-symbol.json
curl -O https://s3.amazonaws.com/model-server/model_archive_1.0/examples/squeezenet_v1.1/squeezenet_v1.1-0000.params
```

1. Compile model to Inferentia target by saving the following Python script to compile.py and run “`python compile.py`”

```
import mxnet as mx
from mxnet.contrib import neuron
import numpy as np

nn_name = "squeezenet_v1.1"

#Load a model
sym, args, auxs = mx.model.load_checkpoint(nn_name, 0)

#Define compilation parameters
#  - input shape and dtype
inputs = {}
inputs['data'] = mx.nd.zeros([1,3,224,224], dtype='float32')
inputs['prob_label'] = mx.nd.zeros([1,1000], dtype='float32')

# compile graph to inferentia target
compile_args = {'num-neuroncores' : 1}
compile_args['excl_node_names'] = ['drop9']
csym, cargs, cauxs = neuron.compile(sym, args, auxs, inputs, **compile_args)

# save compiled model
mx.model.save_checkpoint(nn_name + "_compiled", 0, csym, cargs, cauxs)
```

1. Prepare signature file `signature.json` to configure the input name and shape:

```
{
  "inputs": [
    {
      "data_name": "data",
      "data_shape": [
        1,
        3,
        224,
        224
      ]
    }
  ]
}
```

1. Prepare `synset.txt` which is a list of names for ImageNet prediction classes:

```
curl -O https://s3.amazonaws.com/model-server/model_archive_1.0/examples/squeezenet_v1.1/synset.txt
```

1. Create custom service class following template in model_server_template folder:

```
cp -r ../model_service_template/* .
```

Edit `mxnet_model_service.py` and replace mx.cpu() context with mx.neuron() context:

```
self.mxnet_ctx = mx.neuron()
```

Also, comment out unnecessary data copy for model_input in `mxnet_model_service.py`  as NDArray/Gluon API is not supported in MXNet-Neuron:

```
#model_input = [item.as_in_context(self.mxnet_ctx) for item in model_input]
```

1. Package the model with model-archiver

```
cd ~/mxnet-model-server/examples
model-archiver --force --model-name squeezenet_v1.1_compiled --model-path mxnet_vision --handler mxnet_vision_service:handle
```

1. Start MXNet Model Server (MMS) and load model using RESTful API. The number of workers should be less than or equal number of NeuronCores divided by the number of NeuronCores required by model (<link to API>). Please ensure that Neuron RTD is running with default settings (see Getting Started guide):

```
cd ~/mxnet-model-server/
`mxnet``-``model``-``server ``--``start ``--``model``-``store examples ``>`` ``/dev/``null`
# Pipe to log file if you want to keep a log of MMS
curl -v -X POST "http://localhost:8081/models?initial_workers=1&response_timeout=600&synchronous=true&url=squeezenet_v1.1_compiled.mar"
```

1. Test inference using an example image:

```
`curl ``-``O https``:``//s3.amazonaws.com/model-server/inputs/kitten.jpg`
curl -X POST http://127.0.0.1:8080/predictions/squeezenet_v1.1_compiled -T kitten.jpg
```

You will see the following output:

```
curl -X POST http://127.0.0.1:8080/predictions/squeezenet_v1.1_compiled -T kitten.jpg
[
  {
    "probability": 0.8636875748634338,
    "class": "n02124075 Egyptian cat"
  },
  {
    "probability": 0.0910319983959198,
    "class": "n02123045 tabby, tabby cat"
  },
  {
    "probability": 0.03348880261182785,
    "class": "n02123159 tiger cat"
  },
  {
    "probability": 0.005819481331855059,
    "class": "n02128385 leopard, Panthera pardus"
  },
  {
    "probability": 0.0027489282656461,
    "class": "n02127052 lynx, catamount"
  }
]
```

1. To cleanup after test, issue a delete command via RESTful API and stop the model server:

```
curl -X DELETE http://127.0.0.1:8081/models/squeezenet_v1.1_compiled

mxnet-model-server --stop
```
