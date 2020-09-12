# Running SSD300 with AWS Neuron

This demo shows a Neuron compatible SSD300 implementation that is functionally equivalent to open source SSD300 model. This demo uses TensorFlow-Neuron, PyTorch SSD300 model and checkpoint (https://pytorch.org/hub/nvidia_deeplearningexamples_ssd/) and also shows the performance achieved by the Inf1 instance. 

## Table of Contents

1. Launch EC2 instance and update AWS Neuron SDK software
2. Generating Neuron compatible SSD300 TensorFlow SavedModel
   * Convert open source PyTorch SSD300 model and checkpoint into Neuron compatible SSD300 TensorFlow SavedModel
3. Evaluate the generated SSD300 TensorFlow SavedModel for both accuracy and performance
   * Running threaded inference through the COCO 2017 validation dataset

## Launch EC2 instances and update tensorflow-neuron and neuron-cc

For this demo, launch one inf1.xlarge EC2 instance. We recommend using the latest Ubuntu 18 Deep Learning AMI (DLAMI).

Please configure your ubuntu16/ubuntu18/yum repo following the steps in the [Neuron installation guide](../../../../docs/neuron-install-guide.md) in order to install `tensorflow-model-server-neuron`.

## Generating Neuron compatible SSD300 TensorFlow SavedModel
First connect to your inf1.xlarge instance

### Compile open source PyTorch SSD300 model and checkpoint into Neuron compatible SSD300 TensorFlow SavedModel

In the same directory ssd300_demo, run the following:

1. Create venv and install dependencies

```bash
sudo apt update
sudo apt install g++ python3-dev python3-venv unzip
sudo apt install tensorflow-model-server-neuron
python3 -m venv env
source ./env/bin/activate
pip install pip setuptools --upgrade
pip install -r ./requirements.txt --extra-index-url=https://pip.repos.neuron.amazonaws.com
```

2. Clone NVIDIA's DeepLearningExamples repo that contains PyTorch SSD300.
```bash
git clone https://github.com/NVIDIA/DeepLearningExamples.git
cd DeepLearningExamples
git checkout a644350589f9abc91b203f73e686a50f5d6f3e96
cd ..
```

3. Download PyTorch SSD300 checkpoint file.
```bash
curl -LO https://api.ngc.nvidia.com/v2/models/nvidia/ssdpyt_fp32/versions/1/files/nvidia_ssdpyt_fp32_20190225.pt
```

4. Download COCO 2017 validation set and annotations.
```bash
curl -LO http://images.cocodataset.org/zips/val2017.zip
unzip ./val2017.zip
curl -LO http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip ./annotations_trainval2017.zip
```

5. Convert PyTorch SSD300 model and checkpoint into a Neuron-compatible TensorFlow SavedModel.
```bash
python ssd300_model.py --torch_checkpoint=./nvidia_ssdpyt_fp32_20190225.pt --output_saved_model=./ssd300_tf_neuron/1
```

This converts PyTorch SSD300 model and checkpoint to a Neuron-compatible TensorFlow SavedModel using tensorflow-neuron and neuron-cc. The compilation output is stored in `./ssd300_tf_neuron`.

6. Launch the `tensorflow-model-server-neuron` gRPC server at default port 8500 in the background.
```bash
tensorflow_model_server_neuron --model_base_path=$(pwd)/ssd300_tf_neuron &
```

7. In client, evaluate the Neuron-compatible TensorFlow SavedModel for both accuracy and performance. Note that this client by default assumes a `tensorflow-model-server-neuron` listening at `localhost:8500`. On inf1.xlarge, the expected throughput is 83 images/second once the server is fully warmed up, and the expected mean average precision (mAP) is 0.253.

```bash
python ssd300_evaluation_client.py --val2017=./val2017 --instances_val2017_json=./annotations/instances_val2017.json
```

8. After running the demo, please cleanup resources allocated in Neuron runtime by gracefully killing the `tensorflow_model_server_neuron` process, e. g.,
```bash
killall tensorflow_model_server_neuron
```
