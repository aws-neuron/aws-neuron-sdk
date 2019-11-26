# Tutorial: MXNet-Neuron Model Serving

This Neuron MXNet Model Serving (MMS) example is adapted from the MXNet vision service example which uses pretrained squeezenet to perform image classification: https://github.com/awslabs/mxnet-model-server/tree/master/examples/mxnet_vision.

Before starting this example, please ensure that Neuron-optimized MXNet version mxnet-neuron is installed (see [MXNet Tutorial](./tutorial-mxnet-neuron-compile-infer.md#step-2-install-neuron) ) and Neuron RTD is running with default settings (see [Neuron Runtime getting started](././neuron-runtime/readme.md) ).


1. First, install Java runtime and mxnet-model-server and download the example code:

```bash
cd ~/
# sudo yum -y install -q jre # for AML2
sudo apt-get install -y -q default-jre  # for Ubuntu
pip install mxnet-model-server
git clone https://github.com/awslabs/mxnet-model-server
cd ~/mxnet-model-server/examples/mxnet_vision
```

2. Compile ResNet50 model to Inferentia target by saving the following Python script to compile_resnet50.py and run “`python compile_resnet50.py`”

```python
import mxnet as mx
from mxnet.contrib import neuron
import numpy as np

path='http://data.mxnet.io/models/imagenet/'
mx.test_utils.download(path+'resnet/50-layers/resnet-50-0000.params')
mx.test_utils.download(path+'resnet/50-layers/resnet-50-symbol.json')
mx.test_utils.download(path+'synset.txt')

nn_name = "resnet-50"

#Load a model
sym, args, auxs = mx.model.load_checkpoint(nn_name, 0)

#Define compilation parameters
#  - input shape and dtype
inputs = {'data' : mx.nd.zeros([1,3,224,224], dtype='float32') }

# compile graph to inferentia target
csym, cargs, cauxs = neuron.compile(sym, args, auxs, inputs)

# save compiled model
mx.model.save_checkpoint(nn_name + "_compiled", 0, csym, cargs, cauxs)
```

3. Prepare signature file `signature.json` to configure the input name and shape:

```json
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

4. Prepare `synset.txt` which is a list of names for ImageNet prediction classes:

```bash
curl -O https://s3.amazonaws.com/model-server/model_archive_1.0/examples/squeezenet_v1.1/synset.txt
```

5. Create custom service class following template in model_server_template folder:

```bash
cp -r ../model_service_template/* .
```

Edit `mxnet_model_service.py` and replace mx.cpu() context with mx.neuron() context:

```bash
self.mxnet_ctx = mx.neuron()
```

Also, comment out unnecessary data copy for model_input in `mxnet_model_service.py`  as NDArray/Gluon API is not supported in MXNet-Neuron:

```bash
#model_input = [item.as_in_context(self.mxnet_ctx) for item in model_input]
```

6. Package the model with model-archiver

```bash
cd ~/mxnet-model-server/examples
model-archiver --force --model-name resnet-50_compiled --model-path mxnet_vision --handler mxnet_vision_service:handle
```

7. Start MXNet Model Server (MMS) and load model using RESTful API. The number of workers should be less than or equal number of NeuronCores divided by the number of NeuronCores required by model (<link to API>). Please ensure that Neuron RTD is running with default settings (see Getting Started guide):

```bash
cd ~/mxnet-model-server/
mxnet-model-server --start --model-store examples > /dev/null
# Pipe to log file if you want to keep a log of MMS
curl -v -X POST "http://localhost:8081/models?initial_workers=1&response_timeout=600&synchronous=true&url=resnet-50_compiled.mar"
sleep 10 # allow sufficient time to load model
```

8. Test inference using an example image:

```bash
curl -O https://raw.githubusercontent.com/awslabs/mxnet-model-server/master/docs/images/kitten_small.jpg
curl -X POST http://127.0.0.1:8080/predictions/resnet-50_compiled -T kitten_small.jpg
```

You will see the following output:

```bash
[
  {
    "probability": 0.6375716328620911,
    "class": "n02123045 tabby, tabby cat"
  },
  {
    "probability": 0.1692783385515213,
    "class": "n02123159 tiger cat"
  },
  {
    "probability": 0.12187337130308151,
    "class": "n02124075 Egyptian cat"
  },
  {
    "probability": 0.028840631246566772,
    "class": "n02127052 lynx, catamount"
  },
  {
    "probability": 0.019691042602062225,
    "class": "n02129604 tiger, Panthera tigris"
  }
]
```

9. To cleanup after test, issue a delete command via RESTful API and stop the model server:

```bash
curl -X DELETE http://127.0.0.1:8081/models/resnet-50_compiled

mxnet-model-server --stop

/opt/aws/neuron/bin/neuron-cli reset
```
