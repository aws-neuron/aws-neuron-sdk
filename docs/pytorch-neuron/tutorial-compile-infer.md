# Tutorial: Getting started with torch-neuron (resnet-50 tutorial)

## Steps Overview:

1. Launch an EC2 compilation instance (recommended instance: c5.4xlarge or larger)
2. Install Torch-Neuron and Neuron-Compiler on the compilation instance
3. Compile the compute-graph on the compilation-instance, and copy the artifacts into the deployment-instance
4. Install Torch-Neuron and Neuron-Runtime on an Inf1 instance (deployment Instance)
5. Run inference on the Inf1 instance

## Step 1: Launch EC2 compilation instance

A typical workflow with the Neuron SDK will be to compile trained ML models on a general compute instance (the compilation instance), and then distribute the artifacts to a fleet of Inf1 instances (the deployment instances) for inference execution. Neuron enables PyTorch for both of these steps.

1.1. Select an AMI of your choice. Refer to the [Neuron installation guide](../neuron-install-guide.md) for details.

1.2. Select and launch an EC2 instance

* A c5.4xlarge or larger is recommended. For this example we will use a c5.4xlarge.
* Users may choose to compile and deploy on the same instance, in which case an inf1.6xlarge instance or larger is recommended.  If you choose “launch instance” and search for “neuron” in the AWS EC2 console you will see a short list of  DLAMI images to select from. If you choose a DLAMI, make sure to check the Neuron version installed, and upgrade as needed.

1.3. Select and launch a deployment (Inf1) instance of your choice (if not compiling and inferencing on the same instance). Launch an instance by following EC2 instructions.

## Step 2: Compilation instance installations

If using Conda DLAMI version 27 and up, activate pre-installed PyTorch-Neuron environment (using `source activate aws_neuron_pytorch_p36`  command). Please update Neuron by following update steps in [DLAMI release notes](../../release-notes/dlami-release-notes.md).

To install in your own AMI, please see [Neuron Install Guide](../neuron-install-guide.md) to setup virtual environment and install Torch-Neuron (torch-neuron) and Neuron Compiler (neuron-cc) packages. Also, please install pillow and torchvision for the pretrained resnet50 model (we use no-deps for torchvision because we already have Neuron version of torch installed through torch-neuron)

```bash
pip install pillow==6.2.2

# We use the --no-deps here to prevent torchvision installing standard torch
pip install torchvision==0.4.2 --no-deps
```

## Step 3: Compile on compilation instance

A trained model must be compiled to Inferentia target before it can be deployed on Inf1 instances. In this step we compile the torchvision ResNet50 model and export it as a SavedModel which is in the torchscript format for PyTorch models.

3.1. Create a python script named `trace_resnet50.py` with the following content:

```python
import torch
import numpy as np
import os
import torch_neuron
from torchvision import models

image = torch.zeros([1, 3, 224, 224], dtype=torch.float32)

## Load a pretrained ResNet50 model
model = models.resnet50(pretrained=True)

## Tell the model we are using it for evaluation (not training)
model.eval()
model_neuron = torch.neuron.trace(model, example_inputs=[image])

## Export to saved model
model_neuron.save("resnet50_neuron.pt")
```


3.2. Run the compilation script, which will take a few minutes on c5.4xlarge. At the end of script execution, the compiled model is saved as `resnet50_neuron.pt`  in local directory:

```bash
python trace_resnet50.py
```

You should see:

```bash
INFO:Neuron:compiling module ResNet with neuron-cc
```

3.3 **WARNING**:  If you run the inference script (in section 4 below) on your CPU instance you will get output, but see the following warning.  

```bash
[E neuron_op_impl.cpp:53] Warning: Tensor output are *** NOT CALCULATED *** during CPU
execution and only indicate tensor shape
```
The warning is also displayed during trace where it is expected.

This is an artifact of the way we trace a model on your compile instance.  **Do not perform inference with a neuron traced model on a non neuron supported instance, results will not be calculated.**

3.4. If not compiling and inferring on the same instance, copy the compiled artifacts to the inference server:

```bash
scp -i <PEM key file>  ./resnet50_neuron.pt ubuntu@<instance DNS>:~/ # if Ubuntu-based AMI
scp -i <PEM key file>  ./resnet50_neuron.pt ec2-user@<instance DNS>:~/  # if using AML2-based AMI
```

## Step 4: Deployment Instance Installations

On the instance you are going to use for inference, install Torch-Neuron and Neuron Runtime

4.1. Follow Step 2 above to install Torch-Neuron.

* Install neuron-cc[tensorflow] if compilation on inference instance is desired (see notes above on recommended Inf1 sizes for compilation)
* Skip neuron-cc if compilation is not done on inference instance

4.2. Install the Neuron Runtime using instructions from [Getting started: Installing and Configuring Neuron-RTD](https://github.com/aws/aws-neuron-sdk/blob/master/docs/neuron-runtime/nrt_start.md).


## Step 5: Run inference

In this step we run inference on Inf1 instances using the model compiled in Step 3.

5.1. On the Inf1, create a inference Python script named `infer_resnet50.py` with the following content:


```bash
import os
import time
import torch
import torch_neuron
import json
import numpy as np

from urllib import request

from torchvision import models, transforms, datasets

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

## Predict
results = model_neuron( image )

# Get the top 5 results
top5_idx = results[0].sort()[1][-5:]

# Lookup and print the top 5 labels
top5_labels = [idx2label[idx] for idx in top5_idx]

print("Top 5 labels:\n {}".format(top5_labels) )
```


5.2. Run the inference:

```bash
['tiger', 'lynx', 'tiger_cat', 'Egyptian_cat', 'tabby']
```

## Step 6: Terminate instances

Don’t forget to terminate your instances (compile and inference) from the AWS console so that you don’t continue paying for them once you are done
