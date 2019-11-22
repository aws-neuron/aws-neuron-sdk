# Tutorial: Using Data Parallel on 4 Neuron Cores with TensorFlow-Neuron and the Neuron Compiler with Resnet50

## Steps Overview:

1. Launch an EC2 instance for compilation  and/or Inference
2. Install Neuron for Compiler and Runtime execution
3. Run example

## Step 1-3
These steps are the same as [link](./tutorial-tensorflow-neuron-compile-infer.md). The final step to create the inference script - step 3.4 is replaced with this:

3.4. On the Inf1, create a inference Python script named `infer_resnet50.py` with the following content:
```python
 import os
 from concurrent.futures import ThreadPoolExecutor
 import numpy as np
 import tensorflow as tf
 from tensorflow.keras.preprocessing import image
 from tensorflow.keras.applications import resnet50

 # Create input from image
 img_sgl = image.load_img('kitten_small.jpg', target_size=(224, 224))
 img_arr = image.img_to_array(img_sgl)
 img_arr2 = np.expand_dims(img_arr, axis=0)
 img_arr3 = resnet50.preprocess_input(img_arr2)

 # Load model
 COMPILED_MODEL_DIR = './resnet50_neuron/'
 num_parallel = 4
 predictor_list = [tf.contrib.predictor.from_saved_model(COMPILED_MODEL_DIR) for _ in range(num_parallel)]

 # Run inference
 model_feed_dict={'input': img_arr3}
 with ThreadPoolExecutor(max_workers=num_parallel) as executor:
     future_list = [executor.submit(pred, {'input': img_arr3}) for pred in predictor_list]
     infa_rslts_list = [future.result() for future in future_list]

 # Display results
 for infa_rslts in infa_rslts_list:
     print(resnet50.decode_predictions(infa_rslts["output"], top=5)[0])
```

5. Unzip the mode, download the example image and run the inference:
```bash
 unzip resnet50_neuron.zip
 curl -O https://raw.githubusercontent.com/awslabs/mxnet-model-server/master/docs/images/kitten_small.jpg
 pip install pillow # Necessary for loading images
 python infer_resnet50.py

 [('n02123045', 'tabby', 0.6956522), ('n02127052', 'lynx', 0.120923914), ('n02123159', 'tiger_cat', 0.08831522), ('n02124075', 'Egyptian_cat', 0.06453805), ('n02128757', 'snow_leopard', 0.0087466035)]
```
