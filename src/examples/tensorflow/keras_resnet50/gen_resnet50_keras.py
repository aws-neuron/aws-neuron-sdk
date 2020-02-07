""" Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
    SPDX-License-Identifier: MIT-0
"""

import re
import argparse
import tensorflow as tf
import numpy as np

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

from google.protobuf import text_format
import tensorflow.python.saved_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp16", action='store_true', help="use float16 parameters and operations")
    args = parser.parse_args()

    # set Keras global configurations
    tf.keras.backend.set_learning_phase(0)
    tf.keras.backend.set_image_data_format('channels_last')
    if (args.fp16):
        float_type = 'float16'
        float_type2 = 'fp16'
    else:
        float_type = 'float32'
        float_type2 = 'fp32'
    tf.keras.backend.set_floatx(float_type)

    # load pre-trained model using Keras
    model_name = 'resnet50_%s_keras'%float_type2
    model = ResNet50(weights='imagenet')

    # various save files
    frozen_file = model_name + '.pb'
    opt_file = model_name + '_opt.pb'

    # obtain parameters
    model_input = model.input.name.replace(':0', '')
    model_output = model.output.name.replace(':0', '')
    batch, height, width, channels = model.input.shape

    print ("model, frozen file, optimized file, input size, input node, output node,")
    print ("%s, %s, %s, %dx%dx%d, %s, %s" %(model_name, frozen_file, opt_file, width, height, channels, model_input, model_output) ) 

    # obtain the TF session
    sess = tf.compat.v1.keras.backend.get_session()

    # save checkpoint files for freeze_graph
    ckpt_file = '/tmp/' + model_name + '/' + model_name + '.ckpt'
    graph_file = '/tmp/' + model_name + '/' + model_name + '.pb'
    tf.compat.v1.train.Saver().save(sess, ckpt_file)
    tf.io.write_graph(sess.graph.as_graph_def(), logdir='.', name=graph_file, as_text=False)

    print(model_output)
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
          saver = tf.compat.v1.train.import_meta_graph(ckpt_file + '.meta')
          saver.restore(sess, ckpt_file)
          output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
              sess, tf.compat.v1.get_default_graph().as_graph_def(), [model_output])
          output_graph_def = tf.compat.v1.graph_util.remove_training_nodes(
              output_graph_def, protected_nodes=[model_output])
          with open(frozen_file, 'wb') as f:
              f.write(output_graph_def.SerializeToString())

