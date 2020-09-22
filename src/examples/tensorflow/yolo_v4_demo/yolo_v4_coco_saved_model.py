import argparse
import os
import io
from functools import partial
import requests
import numpy as np
import torch
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def download_file_from_google_drive(id):
    URL = 'https://docs.google.com/uc?export=download'
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    return save_response_content(response)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response):
    CHUNK_SIZE = 32768
    f = io.BytesIO()
    for chunk in response.iter_content(CHUNK_SIZE):
        if chunk:  # filter out keep-alive new chunks
            f.write(chunk)
    f.seek(0)
    return f


def rename_weights(checkpoint):
    name_mapping = {
        'down1.conv1.conv.0.weight': 'models.0.conv1.weight',
        'down1.conv1.conv.1.weight': 'models.0.bn1.weight',
        'down1.conv1.conv.1.bias': 'models.0.bn1.bias',
        'down1.conv1.conv.1.running_mean': 'models.0.bn1.running_mean',
        'down1.conv1.conv.1.running_var': 'models.0.bn1.running_var',
        'down1.conv1.conv.1.num_batches_tracked': 'models.0.bn1.num_batches_tracked',
        'down1.conv2.conv.0.weight': 'models.1.conv2.weight',
        'down1.conv2.conv.1.weight': 'models.1.bn2.weight',
        'down1.conv2.conv.1.bias': 'models.1.bn2.bias',
        'down1.conv2.conv.1.running_mean': 'models.1.bn2.running_mean',
        'down1.conv2.conv.1.running_var': 'models.1.bn2.running_var',
        'down1.conv2.conv.1.num_batches_tracked': 'models.1.bn2.num_batches_tracked',
        'down1.conv3.conv.0.weight': 'models.2.conv3.weight',
        'down1.conv3.conv.1.weight': 'models.2.bn3.weight',
        'down1.conv3.conv.1.bias': 'models.2.bn3.bias',
        'down1.conv3.conv.1.running_mean': 'models.2.bn3.running_mean',
        'down1.conv3.conv.1.running_var': 'models.2.bn3.running_var',
        'down1.conv3.conv.1.num_batches_tracked': 'models.2.bn3.num_batches_tracked',
        'down1.conv4.conv.0.weight': 'models.4.conv4.weight',
        'down1.conv4.conv.1.weight': 'models.4.bn4.weight',
        'down1.conv4.conv.1.bias': 'models.4.bn4.bias',
        'down1.conv4.conv.1.running_mean': 'models.4.bn4.running_mean',
        'down1.conv4.conv.1.running_var': 'models.4.bn4.running_var',
        'down1.conv4.conv.1.num_batches_tracked': 'models.4.bn4.num_batches_tracked',
        'down1.conv5.conv.0.weight': 'models.5.conv5.weight',
        'down1.conv5.conv.1.weight': 'models.5.bn5.weight',
        'down1.conv5.conv.1.bias': 'models.5.bn5.bias',
        'down1.conv5.conv.1.running_mean': 'models.5.bn5.running_mean',
        'down1.conv5.conv.1.running_var': 'models.5.bn5.running_var',
        'down1.conv5.conv.1.num_batches_tracked': 'models.5.bn5.num_batches_tracked',
        'down1.conv6.conv.0.weight': 'models.6.conv6.weight',
        'down1.conv6.conv.1.weight': 'models.6.bn6.weight',
        'down1.conv6.conv.1.bias': 'models.6.bn6.bias',
        'down1.conv6.conv.1.running_mean': 'models.6.bn6.running_mean',
        'down1.conv6.conv.1.running_var': 'models.6.bn6.running_var',
        'down1.conv6.conv.1.num_batches_tracked': 'models.6.bn6.num_batches_tracked',
        'down1.conv7.conv.0.weight': 'models.8.conv7.weight',
        'down1.conv7.conv.1.weight': 'models.8.bn7.weight',
        'down1.conv7.conv.1.bias': 'models.8.bn7.bias',
        'down1.conv7.conv.1.running_mean': 'models.8.bn7.running_mean',
        'down1.conv7.conv.1.running_var': 'models.8.bn7.running_var',
        'down1.conv7.conv.1.num_batches_tracked': 'models.8.bn7.num_batches_tracked',
        'down1.conv8.conv.0.weight': 'models.10.conv8.weight',
        'down1.conv8.conv.1.weight': 'models.10.bn8.weight',
        'down1.conv8.conv.1.bias': 'models.10.bn8.bias',
        'down1.conv8.conv.1.running_mean': 'models.10.bn8.running_mean',
        'down1.conv8.conv.1.running_var': 'models.10.bn8.running_var',
        'down1.conv8.conv.1.num_batches_tracked': 'models.10.bn8.num_batches_tracked',
        'down2.conv1.conv.0.weight': 'models.11.conv9.weight',
        'down2.conv1.conv.1.weight': 'models.11.bn9.weight',
        'down2.conv1.conv.1.bias': 'models.11.bn9.bias',
        'down2.conv1.conv.1.running_mean': 'models.11.bn9.running_mean',
        'down2.conv1.conv.1.running_var': 'models.11.bn9.running_var',
        'down2.conv1.conv.1.num_batches_tracked': 'models.11.bn9.num_batches_tracked',
        'down2.conv2.conv.0.weight': 'models.12.conv10.weight',
        'down2.conv2.conv.1.weight': 'models.12.bn10.weight',
        'down2.conv2.conv.1.bias': 'models.12.bn10.bias',
        'down2.conv2.conv.1.running_mean': 'models.12.bn10.running_mean',
        'down2.conv2.conv.1.running_var': 'models.12.bn10.running_var',
        'down2.conv2.conv.1.num_batches_tracked': 'models.12.bn10.num_batches_tracked',
        'down2.conv3.conv.0.weight': 'models.14.conv11.weight',
        'down2.conv3.conv.1.weight': 'models.14.bn11.weight',
        'down2.conv3.conv.1.bias': 'models.14.bn11.bias',
        'down2.conv3.conv.1.running_mean': 'models.14.bn11.running_mean',
        'down2.conv3.conv.1.running_var': 'models.14.bn11.running_var',
        'down2.conv3.conv.1.num_batches_tracked': 'models.14.bn11.num_batches_tracked',
        'down2.resblock.module_list.0.0.conv.0.weight': 'models.15.conv12.weight',
        'down2.resblock.module_list.0.0.conv.1.weight': 'models.15.bn12.weight',
        'down2.resblock.module_list.0.0.conv.1.bias': 'models.15.bn12.bias',
        'down2.resblock.module_list.0.0.conv.1.running_mean': 'models.15.bn12.running_mean',
        'down2.resblock.module_list.0.0.conv.1.running_var': 'models.15.bn12.running_var',
        'down2.resblock.module_list.0.0.conv.1.num_batches_tracked': 'models.15.bn12.num_batches_tracked',
        'down2.resblock.module_list.0.1.conv.0.weight': 'models.16.conv13.weight',
        'down2.resblock.module_list.0.1.conv.1.weight': 'models.16.bn13.weight',
        'down2.resblock.module_list.0.1.conv.1.bias': 'models.16.bn13.bias',
        'down2.resblock.module_list.0.1.conv.1.running_mean': 'models.16.bn13.running_mean',
        'down2.resblock.module_list.0.1.conv.1.running_var': 'models.16.bn13.running_var',
        'down2.resblock.module_list.0.1.conv.1.num_batches_tracked': 'models.16.bn13.num_batches_tracked',
        'down2.resblock.module_list.1.0.conv.0.weight': 'models.18.conv14.weight',
        'down2.resblock.module_list.1.0.conv.1.weight': 'models.18.bn14.weight',
        'down2.resblock.module_list.1.0.conv.1.bias': 'models.18.bn14.bias',
        'down2.resblock.module_list.1.0.conv.1.running_mean': 'models.18.bn14.running_mean',
        'down2.resblock.module_list.1.0.conv.1.running_var': 'models.18.bn14.running_var',
        'down2.resblock.module_list.1.0.conv.1.num_batches_tracked': 'models.18.bn14.num_batches_tracked',
        'down2.resblock.module_list.1.1.conv.0.weight': 'models.19.conv15.weight',
        'down2.resblock.module_list.1.1.conv.1.weight': 'models.19.bn15.weight',
        'down2.resblock.module_list.1.1.conv.1.bias': 'models.19.bn15.bias',
        'down2.resblock.module_list.1.1.conv.1.running_mean': 'models.19.bn15.running_mean',
        'down2.resblock.module_list.1.1.conv.1.running_var': 'models.19.bn15.running_var',
        'down2.resblock.module_list.1.1.conv.1.num_batches_tracked': 'models.19.bn15.num_batches_tracked',
        'down2.conv4.conv.0.weight': 'models.21.conv16.weight',
        'down2.conv4.conv.1.weight': 'models.21.bn16.weight',
        'down2.conv4.conv.1.bias': 'models.21.bn16.bias',
        'down2.conv4.conv.1.running_mean': 'models.21.bn16.running_mean',
        'down2.conv4.conv.1.running_var': 'models.21.bn16.running_var',
        'down2.conv4.conv.1.num_batches_tracked': 'models.21.bn16.num_batches_tracked',
        'down2.conv5.conv.0.weight': 'models.23.conv17.weight',
        'down2.conv5.conv.1.weight': 'models.23.bn17.weight',
        'down2.conv5.conv.1.bias': 'models.23.bn17.bias',
        'down2.conv5.conv.1.running_mean': 'models.23.bn17.running_mean',
        'down2.conv5.conv.1.running_var': 'models.23.bn17.running_var',
        'down2.conv5.conv.1.num_batches_tracked': 'models.23.bn17.num_batches_tracked',
        'down3.conv1.conv.0.weight': 'models.24.conv18.weight',
        'down3.conv1.conv.1.weight': 'models.24.bn18.weight',
        'down3.conv1.conv.1.bias': 'models.24.bn18.bias',
        'down3.conv1.conv.1.running_mean': 'models.24.bn18.running_mean',
        'down3.conv1.conv.1.running_var': 'models.24.bn18.running_var',
        'down3.conv1.conv.1.num_batches_tracked': 'models.24.bn18.num_batches_tracked',
        'down3.conv2.conv.0.weight': 'models.25.conv19.weight',
        'down3.conv2.conv.1.weight': 'models.25.bn19.weight',
        'down3.conv2.conv.1.bias': 'models.25.bn19.bias',
        'down3.conv2.conv.1.running_mean': 'models.25.bn19.running_mean',
        'down3.conv2.conv.1.running_var': 'models.25.bn19.running_var',
        'down3.conv2.conv.1.num_batches_tracked': 'models.25.bn19.num_batches_tracked',
        'down3.conv3.conv.0.weight': 'models.27.conv20.weight',
        'down3.conv3.conv.1.weight': 'models.27.bn20.weight',
        'down3.conv3.conv.1.bias': 'models.27.bn20.bias',
        'down3.conv3.conv.1.running_mean': 'models.27.bn20.running_mean',
        'down3.conv3.conv.1.running_var': 'models.27.bn20.running_var',
        'down3.conv3.conv.1.num_batches_tracked': 'models.27.bn20.num_batches_tracked',
        'down3.resblock.module_list.0.0.conv.0.weight': 'models.28.conv21.weight',
        'down3.resblock.module_list.0.0.conv.1.weight': 'models.28.bn21.weight',
        'down3.resblock.module_list.0.0.conv.1.bias': 'models.28.bn21.bias',
        'down3.resblock.module_list.0.0.conv.1.running_mean': 'models.28.bn21.running_mean',
        'down3.resblock.module_list.0.0.conv.1.running_var': 'models.28.bn21.running_var',
        'down3.resblock.module_list.0.0.conv.1.num_batches_tracked': 'models.28.bn21.num_batches_tracked',
        'down3.resblock.module_list.0.1.conv.0.weight': 'models.29.conv22.weight',
        'down3.resblock.module_list.0.1.conv.1.weight': 'models.29.bn22.weight',
        'down3.resblock.module_list.0.1.conv.1.bias': 'models.29.bn22.bias',
        'down3.resblock.module_list.0.1.conv.1.running_mean': 'models.29.bn22.running_mean',
        'down3.resblock.module_list.0.1.conv.1.running_var': 'models.29.bn22.running_var',
        'down3.resblock.module_list.0.1.conv.1.num_batches_tracked': 'models.29.bn22.num_batches_tracked',
        'down3.resblock.module_list.1.0.conv.0.weight': 'models.31.conv23.weight',
        'down3.resblock.module_list.1.0.conv.1.weight': 'models.31.bn23.weight',
        'down3.resblock.module_list.1.0.conv.1.bias': 'models.31.bn23.bias',
        'down3.resblock.module_list.1.0.conv.1.running_mean': 'models.31.bn23.running_mean',
        'down3.resblock.module_list.1.0.conv.1.running_var': 'models.31.bn23.running_var',
        'down3.resblock.module_list.1.0.conv.1.num_batches_tracked': 'models.31.bn23.num_batches_tracked',
        'down3.resblock.module_list.1.1.conv.0.weight': 'models.32.conv24.weight',
        'down3.resblock.module_list.1.1.conv.1.weight': 'models.32.bn24.weight',
        'down3.resblock.module_list.1.1.conv.1.bias': 'models.32.bn24.bias',
        'down3.resblock.module_list.1.1.conv.1.running_mean': 'models.32.bn24.running_mean',
        'down3.resblock.module_list.1.1.conv.1.running_var': 'models.32.bn24.running_var',
        'down3.resblock.module_list.1.1.conv.1.num_batches_tracked': 'models.32.bn24.num_batches_tracked',
        'down3.resblock.module_list.2.0.conv.0.weight': 'models.34.conv25.weight',
        'down3.resblock.module_list.2.0.conv.1.weight': 'models.34.bn25.weight',
        'down3.resblock.module_list.2.0.conv.1.bias': 'models.34.bn25.bias',
        'down3.resblock.module_list.2.0.conv.1.running_mean': 'models.34.bn25.running_mean',
        'down3.resblock.module_list.2.0.conv.1.running_var': 'models.34.bn25.running_var',
        'down3.resblock.module_list.2.0.conv.1.num_batches_tracked': 'models.34.bn25.num_batches_tracked',
        'down3.resblock.module_list.2.1.conv.0.weight': 'models.35.conv26.weight',
        'down3.resblock.module_list.2.1.conv.1.weight': 'models.35.bn26.weight',
        'down3.resblock.module_list.2.1.conv.1.bias': 'models.35.bn26.bias',
        'down3.resblock.module_list.2.1.conv.1.running_mean': 'models.35.bn26.running_mean',
        'down3.resblock.module_list.2.1.conv.1.running_var': 'models.35.bn26.running_var',
        'down3.resblock.module_list.2.1.conv.1.num_batches_tracked': 'models.35.bn26.num_batches_tracked',
        'down3.resblock.module_list.3.0.conv.0.weight': 'models.37.conv27.weight',
        'down3.resblock.module_list.3.0.conv.1.weight': 'models.37.bn27.weight',
        'down3.resblock.module_list.3.0.conv.1.bias': 'models.37.bn27.bias',
        'down3.resblock.module_list.3.0.conv.1.running_mean': 'models.37.bn27.running_mean',
        'down3.resblock.module_list.3.0.conv.1.running_var': 'models.37.bn27.running_var',
        'down3.resblock.module_list.3.0.conv.1.num_batches_tracked': 'models.37.bn27.num_batches_tracked',
        'down3.resblock.module_list.3.1.conv.0.weight': 'models.38.conv28.weight',
        'down3.resblock.module_list.3.1.conv.1.weight': 'models.38.bn28.weight',
        'down3.resblock.module_list.3.1.conv.1.bias': 'models.38.bn28.bias',
        'down3.resblock.module_list.3.1.conv.1.running_mean': 'models.38.bn28.running_mean',
        'down3.resblock.module_list.3.1.conv.1.running_var': 'models.38.bn28.running_var',
        'down3.resblock.module_list.3.1.conv.1.num_batches_tracked': 'models.38.bn28.num_batches_tracked',
        'down3.resblock.module_list.4.0.conv.0.weight': 'models.40.conv29.weight',
        'down3.resblock.module_list.4.0.conv.1.weight': 'models.40.bn29.weight',
        'down3.resblock.module_list.4.0.conv.1.bias': 'models.40.bn29.bias',
        'down3.resblock.module_list.4.0.conv.1.running_mean': 'models.40.bn29.running_mean',
        'down3.resblock.module_list.4.0.conv.1.running_var': 'models.40.bn29.running_var',
        'down3.resblock.module_list.4.0.conv.1.num_batches_tracked': 'models.40.bn29.num_batches_tracked',
        'down3.resblock.module_list.4.1.conv.0.weight': 'models.41.conv30.weight',
        'down3.resblock.module_list.4.1.conv.1.weight': 'models.41.bn30.weight',
        'down3.resblock.module_list.4.1.conv.1.bias': 'models.41.bn30.bias',
        'down3.resblock.module_list.4.1.conv.1.running_mean': 'models.41.bn30.running_mean',
        'down3.resblock.module_list.4.1.conv.1.running_var': 'models.41.bn30.running_var',
        'down3.resblock.module_list.4.1.conv.1.num_batches_tracked': 'models.41.bn30.num_batches_tracked',
        'down3.resblock.module_list.5.0.conv.0.weight': 'models.43.conv31.weight',
        'down3.resblock.module_list.5.0.conv.1.weight': 'models.43.bn31.weight',
        'down3.resblock.module_list.5.0.conv.1.bias': 'models.43.bn31.bias',
        'down3.resblock.module_list.5.0.conv.1.running_mean': 'models.43.bn31.running_mean',
        'down3.resblock.module_list.5.0.conv.1.running_var': 'models.43.bn31.running_var',
        'down3.resblock.module_list.5.0.conv.1.num_batches_tracked': 'models.43.bn31.num_batches_tracked',
        'down3.resblock.module_list.5.1.conv.0.weight': 'models.44.conv32.weight',
        'down3.resblock.module_list.5.1.conv.1.weight': 'models.44.bn32.weight',
        'down3.resblock.module_list.5.1.conv.1.bias': 'models.44.bn32.bias',
        'down3.resblock.module_list.5.1.conv.1.running_mean': 'models.44.bn32.running_mean',
        'down3.resblock.module_list.5.1.conv.1.running_var': 'models.44.bn32.running_var',
        'down3.resblock.module_list.5.1.conv.1.num_batches_tracked': 'models.44.bn32.num_batches_tracked',
        'down3.resblock.module_list.6.0.conv.0.weight': 'models.46.conv33.weight',
        'down3.resblock.module_list.6.0.conv.1.weight': 'models.46.bn33.weight',
        'down3.resblock.module_list.6.0.conv.1.bias': 'models.46.bn33.bias',
        'down3.resblock.module_list.6.0.conv.1.running_mean': 'models.46.bn33.running_mean',
        'down3.resblock.module_list.6.0.conv.1.running_var': 'models.46.bn33.running_var',
        'down3.resblock.module_list.6.0.conv.1.num_batches_tracked': 'models.46.bn33.num_batches_tracked',
        'down3.resblock.module_list.6.1.conv.0.weight': 'models.47.conv34.weight',
        'down3.resblock.module_list.6.1.conv.1.weight': 'models.47.bn34.weight',
        'down3.resblock.module_list.6.1.conv.1.bias': 'models.47.bn34.bias',
        'down3.resblock.module_list.6.1.conv.1.running_mean': 'models.47.bn34.running_mean',
        'down3.resblock.module_list.6.1.conv.1.running_var': 'models.47.bn34.running_var',
        'down3.resblock.module_list.6.1.conv.1.num_batches_tracked': 'models.47.bn34.num_batches_tracked',
        'down3.resblock.module_list.7.0.conv.0.weight': 'models.49.conv35.weight',
        'down3.resblock.module_list.7.0.conv.1.weight': 'models.49.bn35.weight',
        'down3.resblock.module_list.7.0.conv.1.bias': 'models.49.bn35.bias',
        'down3.resblock.module_list.7.0.conv.1.running_mean': 'models.49.bn35.running_mean',
        'down3.resblock.module_list.7.0.conv.1.running_var': 'models.49.bn35.running_var',
        'down3.resblock.module_list.7.0.conv.1.num_batches_tracked': 'models.49.bn35.num_batches_tracked',
        'down3.resblock.module_list.7.1.conv.0.weight': 'models.50.conv36.weight',
        'down3.resblock.module_list.7.1.conv.1.weight': 'models.50.bn36.weight',
        'down3.resblock.module_list.7.1.conv.1.bias': 'models.50.bn36.bias',
        'down3.resblock.module_list.7.1.conv.1.running_mean': 'models.50.bn36.running_mean',
        'down3.resblock.module_list.7.1.conv.1.running_var': 'models.50.bn36.running_var',
        'down3.resblock.module_list.7.1.conv.1.num_batches_tracked': 'models.50.bn36.num_batches_tracked',
        'down3.conv4.conv.0.weight': 'models.52.conv37.weight',
        'down3.conv4.conv.1.weight': 'models.52.bn37.weight',
        'down3.conv4.conv.1.bias': 'models.52.bn37.bias',
        'down3.conv4.conv.1.running_mean': 'models.52.bn37.running_mean',
        'down3.conv4.conv.1.running_var': 'models.52.bn37.running_var',
        'down3.conv4.conv.1.num_batches_tracked': 'models.52.bn37.num_batches_tracked',
        'down3.conv5.conv.0.weight': 'models.54.conv38.weight',
        'down3.conv5.conv.1.weight': 'models.54.bn38.weight',
        'down3.conv5.conv.1.bias': 'models.54.bn38.bias',
        'down3.conv5.conv.1.running_mean': 'models.54.bn38.running_mean',
        'down3.conv5.conv.1.running_var': 'models.54.bn38.running_var',
        'down3.conv5.conv.1.num_batches_tracked': 'models.54.bn38.num_batches_tracked',
        'down4.conv1.conv.0.weight': 'models.55.conv39.weight',
        'down4.conv1.conv.1.weight': 'models.55.bn39.weight',
        'down4.conv1.conv.1.bias': 'models.55.bn39.bias',
        'down4.conv1.conv.1.running_mean': 'models.55.bn39.running_mean',
        'down4.conv1.conv.1.running_var': 'models.55.bn39.running_var',
        'down4.conv1.conv.1.num_batches_tracked': 'models.55.bn39.num_batches_tracked',
        'down4.conv2.conv.0.weight': 'models.56.conv40.weight',
        'down4.conv2.conv.1.weight': 'models.56.bn40.weight',
        'down4.conv2.conv.1.bias': 'models.56.bn40.bias',
        'down4.conv2.conv.1.running_mean': 'models.56.bn40.running_mean',
        'down4.conv2.conv.1.running_var': 'models.56.bn40.running_var',
        'down4.conv2.conv.1.num_batches_tracked': 'models.56.bn40.num_batches_tracked',
        'down4.conv3.conv.0.weight': 'models.58.conv41.weight',
        'down4.conv3.conv.1.weight': 'models.58.bn41.weight',
        'down4.conv3.conv.1.bias': 'models.58.bn41.bias',
        'down4.conv3.conv.1.running_mean': 'models.58.bn41.running_mean',
        'down4.conv3.conv.1.running_var': 'models.58.bn41.running_var',
        'down4.conv3.conv.1.num_batches_tracked': 'models.58.bn41.num_batches_tracked',
        'down4.resblock.module_list.0.0.conv.0.weight': 'models.59.conv42.weight',
        'down4.resblock.module_list.0.0.conv.1.weight': 'models.59.bn42.weight',
        'down4.resblock.module_list.0.0.conv.1.bias': 'models.59.bn42.bias',
        'down4.resblock.module_list.0.0.conv.1.running_mean': 'models.59.bn42.running_mean',
        'down4.resblock.module_list.0.0.conv.1.running_var': 'models.59.bn42.running_var',
        'down4.resblock.module_list.0.0.conv.1.num_batches_tracked': 'models.59.bn42.num_batches_tracked',
        'down4.resblock.module_list.0.1.conv.0.weight': 'models.60.conv43.weight',
        'down4.resblock.module_list.0.1.conv.1.weight': 'models.60.bn43.weight',
        'down4.resblock.module_list.0.1.conv.1.bias': 'models.60.bn43.bias',
        'down4.resblock.module_list.0.1.conv.1.running_mean': 'models.60.bn43.running_mean',
        'down4.resblock.module_list.0.1.conv.1.running_var': 'models.60.bn43.running_var',
        'down4.resblock.module_list.0.1.conv.1.num_batches_tracked': 'models.60.bn43.num_batches_tracked',
        'down4.resblock.module_list.1.0.conv.0.weight': 'models.62.conv44.weight',
        'down4.resblock.module_list.1.0.conv.1.weight': 'models.62.bn44.weight',
        'down4.resblock.module_list.1.0.conv.1.bias': 'models.62.bn44.bias',
        'down4.resblock.module_list.1.0.conv.1.running_mean': 'models.62.bn44.running_mean',
        'down4.resblock.module_list.1.0.conv.1.running_var': 'models.62.bn44.running_var',
        'down4.resblock.module_list.1.0.conv.1.num_batches_tracked': 'models.62.bn44.num_batches_tracked',
        'down4.resblock.module_list.1.1.conv.0.weight': 'models.63.conv45.weight',
        'down4.resblock.module_list.1.1.conv.1.weight': 'models.63.bn45.weight',
        'down4.resblock.module_list.1.1.conv.1.bias': 'models.63.bn45.bias',
        'down4.resblock.module_list.1.1.conv.1.running_mean': 'models.63.bn45.running_mean',
        'down4.resblock.module_list.1.1.conv.1.running_var': 'models.63.bn45.running_var',
        'down4.resblock.module_list.1.1.conv.1.num_batches_tracked': 'models.63.bn45.num_batches_tracked',
        'down4.resblock.module_list.2.0.conv.0.weight': 'models.65.conv46.weight',
        'down4.resblock.module_list.2.0.conv.1.weight': 'models.65.bn46.weight',
        'down4.resblock.module_list.2.0.conv.1.bias': 'models.65.bn46.bias',
        'down4.resblock.module_list.2.0.conv.1.running_mean': 'models.65.bn46.running_mean',
        'down4.resblock.module_list.2.0.conv.1.running_var': 'models.65.bn46.running_var',
        'down4.resblock.module_list.2.0.conv.1.num_batches_tracked': 'models.65.bn46.num_batches_tracked',
        'down4.resblock.module_list.2.1.conv.0.weight': 'models.66.conv47.weight',
        'down4.resblock.module_list.2.1.conv.1.weight': 'models.66.bn47.weight',
        'down4.resblock.module_list.2.1.conv.1.bias': 'models.66.bn47.bias',
        'down4.resblock.module_list.2.1.conv.1.running_mean': 'models.66.bn47.running_mean',
        'down4.resblock.module_list.2.1.conv.1.running_var': 'models.66.bn47.running_var',
        'down4.resblock.module_list.2.1.conv.1.num_batches_tracked': 'models.66.bn47.num_batches_tracked',
        'down4.resblock.module_list.3.0.conv.0.weight': 'models.68.conv48.weight',
        'down4.resblock.module_list.3.0.conv.1.weight': 'models.68.bn48.weight',
        'down4.resblock.module_list.3.0.conv.1.bias': 'models.68.bn48.bias',
        'down4.resblock.module_list.3.0.conv.1.running_mean': 'models.68.bn48.running_mean',
        'down4.resblock.module_list.3.0.conv.1.running_var': 'models.68.bn48.running_var',
        'down4.resblock.module_list.3.0.conv.1.num_batches_tracked': 'models.68.bn48.num_batches_tracked',
        'down4.resblock.module_list.3.1.conv.0.weight': 'models.69.conv49.weight',
        'down4.resblock.module_list.3.1.conv.1.weight': 'models.69.bn49.weight',
        'down4.resblock.module_list.3.1.conv.1.bias': 'models.69.bn49.bias',
        'down4.resblock.module_list.3.1.conv.1.running_mean': 'models.69.bn49.running_mean',
        'down4.resblock.module_list.3.1.conv.1.running_var': 'models.69.bn49.running_var',
        'down4.resblock.module_list.3.1.conv.1.num_batches_tracked': 'models.69.bn49.num_batches_tracked',
        'down4.resblock.module_list.4.0.conv.0.weight': 'models.71.conv50.weight',
        'down4.resblock.module_list.4.0.conv.1.weight': 'models.71.bn50.weight',
        'down4.resblock.module_list.4.0.conv.1.bias': 'models.71.bn50.bias',
        'down4.resblock.module_list.4.0.conv.1.running_mean': 'models.71.bn50.running_mean',
        'down4.resblock.module_list.4.0.conv.1.running_var': 'models.71.bn50.running_var',
        'down4.resblock.module_list.4.0.conv.1.num_batches_tracked': 'models.71.bn50.num_batches_tracked',
        'down4.resblock.module_list.4.1.conv.0.weight': 'models.72.conv51.weight',
        'down4.resblock.module_list.4.1.conv.1.weight': 'models.72.bn51.weight',
        'down4.resblock.module_list.4.1.conv.1.bias': 'models.72.bn51.bias',
        'down4.resblock.module_list.4.1.conv.1.running_mean': 'models.72.bn51.running_mean',
        'down4.resblock.module_list.4.1.conv.1.running_var': 'models.72.bn51.running_var',
        'down4.resblock.module_list.4.1.conv.1.num_batches_tracked': 'models.72.bn51.num_batches_tracked',
        'down4.resblock.module_list.5.0.conv.0.weight': 'models.74.conv52.weight',
        'down4.resblock.module_list.5.0.conv.1.weight': 'models.74.bn52.weight',
        'down4.resblock.module_list.5.0.conv.1.bias': 'models.74.bn52.bias',
        'down4.resblock.module_list.5.0.conv.1.running_mean': 'models.74.bn52.running_mean',
        'down4.resblock.module_list.5.0.conv.1.running_var': 'models.74.bn52.running_var',
        'down4.resblock.module_list.5.0.conv.1.num_batches_tracked': 'models.74.bn52.num_batches_tracked',
        'down4.resblock.module_list.5.1.conv.0.weight': 'models.75.conv53.weight',
        'down4.resblock.module_list.5.1.conv.1.weight': 'models.75.bn53.weight',
        'down4.resblock.module_list.5.1.conv.1.bias': 'models.75.bn53.bias',
        'down4.resblock.module_list.5.1.conv.1.running_mean': 'models.75.bn53.running_mean',
        'down4.resblock.module_list.5.1.conv.1.running_var': 'models.75.bn53.running_var',
        'down4.resblock.module_list.5.1.conv.1.num_batches_tracked': 'models.75.bn53.num_batches_tracked',
        'down4.resblock.module_list.6.0.conv.0.weight': 'models.77.conv54.weight',
        'down4.resblock.module_list.6.0.conv.1.weight': 'models.77.bn54.weight',
        'down4.resblock.module_list.6.0.conv.1.bias': 'models.77.bn54.bias',
        'down4.resblock.module_list.6.0.conv.1.running_mean': 'models.77.bn54.running_mean',
        'down4.resblock.module_list.6.0.conv.1.running_var': 'models.77.bn54.running_var',
        'down4.resblock.module_list.6.0.conv.1.num_batches_tracked': 'models.77.bn54.num_batches_tracked',
        'down4.resblock.module_list.6.1.conv.0.weight': 'models.78.conv55.weight',
        'down4.resblock.module_list.6.1.conv.1.weight': 'models.78.bn55.weight',
        'down4.resblock.module_list.6.1.conv.1.bias': 'models.78.bn55.bias',
        'down4.resblock.module_list.6.1.conv.1.running_mean': 'models.78.bn55.running_mean',
        'down4.resblock.module_list.6.1.conv.1.running_var': 'models.78.bn55.running_var',
        'down4.resblock.module_list.6.1.conv.1.num_batches_tracked': 'models.78.bn55.num_batches_tracked',
        'down4.resblock.module_list.7.0.conv.0.weight': 'models.80.conv56.weight',
        'down4.resblock.module_list.7.0.conv.1.weight': 'models.80.bn56.weight',
        'down4.resblock.module_list.7.0.conv.1.bias': 'models.80.bn56.bias',
        'down4.resblock.module_list.7.0.conv.1.running_mean': 'models.80.bn56.running_mean',
        'down4.resblock.module_list.7.0.conv.1.running_var': 'models.80.bn56.running_var',
        'down4.resblock.module_list.7.0.conv.1.num_batches_tracked': 'models.80.bn56.num_batches_tracked',
        'down4.resblock.module_list.7.1.conv.0.weight': 'models.81.conv57.weight',
        'down4.resblock.module_list.7.1.conv.1.weight': 'models.81.bn57.weight',
        'down4.resblock.module_list.7.1.conv.1.bias': 'models.81.bn57.bias',
        'down4.resblock.module_list.7.1.conv.1.running_mean': 'models.81.bn57.running_mean',
        'down4.resblock.module_list.7.1.conv.1.running_var': 'models.81.bn57.running_var',
        'down4.resblock.module_list.7.1.conv.1.num_batches_tracked': 'models.81.bn57.num_batches_tracked',
        'down4.conv4.conv.0.weight': 'models.83.conv58.weight',
        'down4.conv4.conv.1.weight': 'models.83.bn58.weight',
        'down4.conv4.conv.1.bias': 'models.83.bn58.bias',
        'down4.conv4.conv.1.running_mean': 'models.83.bn58.running_mean',
        'down4.conv4.conv.1.running_var': 'models.83.bn58.running_var',
        'down4.conv4.conv.1.num_batches_tracked': 'models.83.bn58.num_batches_tracked',
        'down4.conv5.conv.0.weight': 'models.85.conv59.weight',
        'down4.conv5.conv.1.weight': 'models.85.bn59.weight',
        'down4.conv5.conv.1.bias': 'models.85.bn59.bias',
        'down4.conv5.conv.1.running_mean': 'models.85.bn59.running_mean',
        'down4.conv5.conv.1.running_var': 'models.85.bn59.running_var',
        'down4.conv5.conv.1.num_batches_tracked': 'models.85.bn59.num_batches_tracked',
        'down5.conv1.conv.0.weight': 'models.86.conv60.weight',
        'down5.conv1.conv.1.weight': 'models.86.bn60.weight',
        'down5.conv1.conv.1.bias': 'models.86.bn60.bias',
        'down5.conv1.conv.1.running_mean': 'models.86.bn60.running_mean',
        'down5.conv1.conv.1.running_var': 'models.86.bn60.running_var',
        'down5.conv1.conv.1.num_batches_tracked': 'models.86.bn60.num_batches_tracked',
        'down5.conv2.conv.0.weight': 'models.87.conv61.weight',
        'down5.conv2.conv.1.weight': 'models.87.bn61.weight',
        'down5.conv2.conv.1.bias': 'models.87.bn61.bias',
        'down5.conv2.conv.1.running_mean': 'models.87.bn61.running_mean',
        'down5.conv2.conv.1.running_var': 'models.87.bn61.running_var',
        'down5.conv2.conv.1.num_batches_tracked': 'models.87.bn61.num_batches_tracked',
        'down5.conv3.conv.0.weight': 'models.89.conv62.weight',
        'down5.conv3.conv.1.weight': 'models.89.bn62.weight',
        'down5.conv3.conv.1.bias': 'models.89.bn62.bias',
        'down5.conv3.conv.1.running_mean': 'models.89.bn62.running_mean',
        'down5.conv3.conv.1.running_var': 'models.89.bn62.running_var',
        'down5.conv3.conv.1.num_batches_tracked': 'models.89.bn62.num_batches_tracked',
        'down5.resblock.module_list.0.0.conv.0.weight': 'models.90.conv63.weight',
        'down5.resblock.module_list.0.0.conv.1.weight': 'models.90.bn63.weight',
        'down5.resblock.module_list.0.0.conv.1.bias': 'models.90.bn63.bias',
        'down5.resblock.module_list.0.0.conv.1.running_mean': 'models.90.bn63.running_mean',
        'down5.resblock.module_list.0.0.conv.1.running_var': 'models.90.bn63.running_var',
        'down5.resblock.module_list.0.0.conv.1.num_batches_tracked': 'models.90.bn63.num_batches_tracked',
        'down5.resblock.module_list.0.1.conv.0.weight': 'models.91.conv64.weight',
        'down5.resblock.module_list.0.1.conv.1.weight': 'models.91.bn64.weight',
        'down5.resblock.module_list.0.1.conv.1.bias': 'models.91.bn64.bias',
        'down5.resblock.module_list.0.1.conv.1.running_mean': 'models.91.bn64.running_mean',
        'down5.resblock.module_list.0.1.conv.1.running_var': 'models.91.bn64.running_var',
        'down5.resblock.module_list.0.1.conv.1.num_batches_tracked': 'models.91.bn64.num_batches_tracked',
        'down5.resblock.module_list.1.0.conv.0.weight': 'models.93.conv65.weight',
        'down5.resblock.module_list.1.0.conv.1.weight': 'models.93.bn65.weight',
        'down5.resblock.module_list.1.0.conv.1.bias': 'models.93.bn65.bias',
        'down5.resblock.module_list.1.0.conv.1.running_mean': 'models.93.bn65.running_mean',
        'down5.resblock.module_list.1.0.conv.1.running_var': 'models.93.bn65.running_var',
        'down5.resblock.module_list.1.0.conv.1.num_batches_tracked': 'models.93.bn65.num_batches_tracked',
        'down5.resblock.module_list.1.1.conv.0.weight': 'models.94.conv66.weight',
        'down5.resblock.module_list.1.1.conv.1.weight': 'models.94.bn66.weight',
        'down5.resblock.module_list.1.1.conv.1.bias': 'models.94.bn66.bias',
        'down5.resblock.module_list.1.1.conv.1.running_mean': 'models.94.bn66.running_mean',
        'down5.resblock.module_list.1.1.conv.1.running_var': 'models.94.bn66.running_var',
        'down5.resblock.module_list.1.1.conv.1.num_batches_tracked': 'models.94.bn66.num_batches_tracked',
        'down5.resblock.module_list.2.0.conv.0.weight': 'models.96.conv67.weight',
        'down5.resblock.module_list.2.0.conv.1.weight': 'models.96.bn67.weight',
        'down5.resblock.module_list.2.0.conv.1.bias': 'models.96.bn67.bias',
        'down5.resblock.module_list.2.0.conv.1.running_mean': 'models.96.bn67.running_mean',
        'down5.resblock.module_list.2.0.conv.1.running_var': 'models.96.bn67.running_var',
        'down5.resblock.module_list.2.0.conv.1.num_batches_tracked': 'models.96.bn67.num_batches_tracked',
        'down5.resblock.module_list.2.1.conv.0.weight': 'models.97.conv68.weight',
        'down5.resblock.module_list.2.1.conv.1.weight': 'models.97.bn68.weight',
        'down5.resblock.module_list.2.1.conv.1.bias': 'models.97.bn68.bias',
        'down5.resblock.module_list.2.1.conv.1.running_mean': 'models.97.bn68.running_mean',
        'down5.resblock.module_list.2.1.conv.1.running_var': 'models.97.bn68.running_var',
        'down5.resblock.module_list.2.1.conv.1.num_batches_tracked': 'models.97.bn68.num_batches_tracked',
        'down5.resblock.module_list.3.0.conv.0.weight': 'models.99.conv69.weight',
        'down5.resblock.module_list.3.0.conv.1.weight': 'models.99.bn69.weight',
        'down5.resblock.module_list.3.0.conv.1.bias': 'models.99.bn69.bias',
        'down5.resblock.module_list.3.0.conv.1.running_mean': 'models.99.bn69.running_mean',
        'down5.resblock.module_list.3.0.conv.1.running_var': 'models.99.bn69.running_var',
        'down5.resblock.module_list.3.0.conv.1.num_batches_tracked': 'models.99.bn69.num_batches_tracked',
        'down5.resblock.module_list.3.1.conv.0.weight': 'models.100.conv70.weight',
        'down5.resblock.module_list.3.1.conv.1.weight': 'models.100.bn70.weight',
        'down5.resblock.module_list.3.1.conv.1.bias': 'models.100.bn70.bias',
        'down5.resblock.module_list.3.1.conv.1.running_mean': 'models.100.bn70.running_mean',
        'down5.resblock.module_list.3.1.conv.1.running_var': 'models.100.bn70.running_var',
        'down5.resblock.module_list.3.1.conv.1.num_batches_tracked': 'models.100.bn70.num_batches_tracked',
        'down5.conv4.conv.0.weight': 'models.102.conv71.weight',
        'down5.conv4.conv.1.weight': 'models.102.bn71.weight',
        'down5.conv4.conv.1.bias': 'models.102.bn71.bias',
        'down5.conv4.conv.1.running_mean': 'models.102.bn71.running_mean',
        'down5.conv4.conv.1.running_var': 'models.102.bn71.running_var',
        'down5.conv4.conv.1.num_batches_tracked': 'models.102.bn71.num_batches_tracked',
        'down5.conv5.conv.0.weight': 'models.104.conv72.weight',
        'down5.conv5.conv.1.weight': 'models.104.bn72.weight',
        'down5.conv5.conv.1.bias': 'models.104.bn72.bias',
        'down5.conv5.conv.1.running_mean': 'models.104.bn72.running_mean',
        'down5.conv5.conv.1.running_var': 'models.104.bn72.running_var',
        'down5.conv5.conv.1.num_batches_tracked': 'models.104.bn72.num_batches_tracked',
        'neek.conv1.conv.0.weight': 'models.105.conv73.weight',
        'neek.conv1.conv.1.weight': 'models.105.bn73.weight',
        'neek.conv1.conv.1.bias': 'models.105.bn73.bias',
        'neek.conv1.conv.1.running_mean': 'models.105.bn73.running_mean',
        'neek.conv1.conv.1.running_var': 'models.105.bn73.running_var',
        'neek.conv1.conv.1.num_batches_tracked': 'models.105.bn73.num_batches_tracked',
        'neek.conv2.conv.0.weight': 'models.106.conv74.weight',
        'neek.conv2.conv.1.weight': 'models.106.bn74.weight',
        'neek.conv2.conv.1.bias': 'models.106.bn74.bias',
        'neek.conv2.conv.1.running_mean': 'models.106.bn74.running_mean',
        'neek.conv2.conv.1.running_var': 'models.106.bn74.running_var',
        'neek.conv2.conv.1.num_batches_tracked': 'models.106.bn74.num_batches_tracked',
        'neek.conv3.conv.0.weight': 'models.107.conv75.weight',
        'neek.conv3.conv.1.weight': 'models.107.bn75.weight',
        'neek.conv3.conv.1.bias': 'models.107.bn75.bias',
        'neek.conv3.conv.1.running_mean': 'models.107.bn75.running_mean',
        'neek.conv3.conv.1.running_var': 'models.107.bn75.running_var',
        'neek.conv3.conv.1.num_batches_tracked': 'models.107.bn75.num_batches_tracked',
        'neek.conv4.conv.0.weight': 'models.114.conv76.weight',
        'neek.conv4.conv.1.weight': 'models.114.bn76.weight',
        'neek.conv4.conv.1.bias': 'models.114.bn76.bias',
        'neek.conv4.conv.1.running_mean': 'models.114.bn76.running_mean',
        'neek.conv4.conv.1.running_var': 'models.114.bn76.running_var',
        'neek.conv4.conv.1.num_batches_tracked': 'models.114.bn76.num_batches_tracked',
        'neek.conv5.conv.0.weight': 'models.115.conv77.weight',
        'neek.conv5.conv.1.weight': 'models.115.bn77.weight',
        'neek.conv5.conv.1.bias': 'models.115.bn77.bias',
        'neek.conv5.conv.1.running_mean': 'models.115.bn77.running_mean',
        'neek.conv5.conv.1.running_var': 'models.115.bn77.running_var',
        'neek.conv5.conv.1.num_batches_tracked': 'models.115.bn77.num_batches_tracked',
        'neek.conv6.conv.0.weight': 'models.116.conv78.weight',
        'neek.conv6.conv.1.weight': 'models.116.bn78.weight',
        'neek.conv6.conv.1.bias': 'models.116.bn78.bias',
        'neek.conv6.conv.1.running_mean': 'models.116.bn78.running_mean',
        'neek.conv6.conv.1.running_var': 'models.116.bn78.running_var',
        'neek.conv6.conv.1.num_batches_tracked': 'models.116.bn78.num_batches_tracked',
        'neek.conv7.conv.0.weight': 'models.117.conv79.weight',
        'neek.conv7.conv.1.weight': 'models.117.bn79.weight',
        'neek.conv7.conv.1.bias': 'models.117.bn79.bias',
        'neek.conv7.conv.1.running_mean': 'models.117.bn79.running_mean',
        'neek.conv7.conv.1.running_var': 'models.117.bn79.running_var',
        'neek.conv7.conv.1.num_batches_tracked': 'models.117.bn79.num_batches_tracked',
        'neek.conv8.conv.0.weight': 'models.120.conv80.weight',
        'neek.conv8.conv.1.weight': 'models.120.bn80.weight',
        'neek.conv8.conv.1.bias': 'models.120.bn80.bias',
        'neek.conv8.conv.1.running_mean': 'models.120.bn80.running_mean',
        'neek.conv8.conv.1.running_var': 'models.120.bn80.running_var',
        'neek.conv8.conv.1.num_batches_tracked': 'models.120.bn80.num_batches_tracked',
        'neek.conv9.conv.0.weight': 'models.122.conv81.weight',
        'neek.conv9.conv.1.weight': 'models.122.bn81.weight',
        'neek.conv9.conv.1.bias': 'models.122.bn81.bias',
        'neek.conv9.conv.1.running_mean': 'models.122.bn81.running_mean',
        'neek.conv9.conv.1.running_var': 'models.122.bn81.running_var',
        'neek.conv9.conv.1.num_batches_tracked': 'models.122.bn81.num_batches_tracked',
        'neek.conv10.conv.0.weight': 'models.123.conv82.weight',
        'neek.conv10.conv.1.weight': 'models.123.bn82.weight',
        'neek.conv10.conv.1.bias': 'models.123.bn82.bias',
        'neek.conv10.conv.1.running_mean': 'models.123.bn82.running_mean',
        'neek.conv10.conv.1.running_var': 'models.123.bn82.running_var',
        'neek.conv10.conv.1.num_batches_tracked': 'models.123.bn82.num_batches_tracked',
        'neek.conv11.conv.0.weight': 'models.124.conv83.weight',
        'neek.conv11.conv.1.weight': 'models.124.bn83.weight',
        'neek.conv11.conv.1.bias': 'models.124.bn83.bias',
        'neek.conv11.conv.1.running_mean': 'models.124.bn83.running_mean',
        'neek.conv11.conv.1.running_var': 'models.124.bn83.running_var',
        'neek.conv11.conv.1.num_batches_tracked': 'models.124.bn83.num_batches_tracked',
        'neek.conv12.conv.0.weight': 'models.125.conv84.weight',
        'neek.conv12.conv.1.weight': 'models.125.bn84.weight',
        'neek.conv12.conv.1.bias': 'models.125.bn84.bias',
        'neek.conv12.conv.1.running_mean': 'models.125.bn84.running_mean',
        'neek.conv12.conv.1.running_var': 'models.125.bn84.running_var',
        'neek.conv12.conv.1.num_batches_tracked': 'models.125.bn84.num_batches_tracked',
        'neek.conv13.conv.0.weight': 'models.126.conv85.weight',
        'neek.conv13.conv.1.weight': 'models.126.bn85.weight',
        'neek.conv13.conv.1.bias': 'models.126.bn85.bias',
        'neek.conv13.conv.1.running_mean': 'models.126.bn85.running_mean',
        'neek.conv13.conv.1.running_var': 'models.126.bn85.running_var',
        'neek.conv13.conv.1.num_batches_tracked': 'models.126.bn85.num_batches_tracked',
        'neek.conv14.conv.0.weight': 'models.127.conv86.weight',
        'neek.conv14.conv.1.weight': 'models.127.bn86.weight',
        'neek.conv14.conv.1.bias': 'models.127.bn86.bias',
        'neek.conv14.conv.1.running_mean': 'models.127.bn86.running_mean',
        'neek.conv14.conv.1.running_var': 'models.127.bn86.running_var',
        'neek.conv14.conv.1.num_batches_tracked': 'models.127.bn86.num_batches_tracked',
        'neek.conv15.conv.0.weight': 'models.130.conv87.weight',
        'neek.conv15.conv.1.weight': 'models.130.bn87.weight',
        'neek.conv15.conv.1.bias': 'models.130.bn87.bias',
        'neek.conv15.conv.1.running_mean': 'models.130.bn87.running_mean',
        'neek.conv15.conv.1.running_var': 'models.130.bn87.running_var',
        'neek.conv15.conv.1.num_batches_tracked': 'models.130.bn87.num_batches_tracked',
        'neek.conv16.conv.0.weight': 'models.132.conv88.weight',
        'neek.conv16.conv.1.weight': 'models.132.bn88.weight',
        'neek.conv16.conv.1.bias': 'models.132.bn88.bias',
        'neek.conv16.conv.1.running_mean': 'models.132.bn88.running_mean',
        'neek.conv16.conv.1.running_var': 'models.132.bn88.running_var',
        'neek.conv16.conv.1.num_batches_tracked': 'models.132.bn88.num_batches_tracked',
        'neek.conv17.conv.0.weight': 'models.133.conv89.weight',
        'neek.conv17.conv.1.weight': 'models.133.bn89.weight',
        'neek.conv17.conv.1.bias': 'models.133.bn89.bias',
        'neek.conv17.conv.1.running_mean': 'models.133.bn89.running_mean',
        'neek.conv17.conv.1.running_var': 'models.133.bn89.running_var',
        'neek.conv17.conv.1.num_batches_tracked': 'models.133.bn89.num_batches_tracked',
        'neek.conv18.conv.0.weight': 'models.134.conv90.weight',
        'neek.conv18.conv.1.weight': 'models.134.bn90.weight',
        'neek.conv18.conv.1.bias': 'models.134.bn90.bias',
        'neek.conv18.conv.1.running_mean': 'models.134.bn90.running_mean',
        'neek.conv18.conv.1.running_var': 'models.134.bn90.running_var',
        'neek.conv18.conv.1.num_batches_tracked': 'models.134.bn90.num_batches_tracked',
        'neek.conv19.conv.0.weight': 'models.135.conv91.weight',
        'neek.conv19.conv.1.weight': 'models.135.bn91.weight',
        'neek.conv19.conv.1.bias': 'models.135.bn91.bias',
        'neek.conv19.conv.1.running_mean': 'models.135.bn91.running_mean',
        'neek.conv19.conv.1.running_var': 'models.135.bn91.running_var',
        'neek.conv19.conv.1.num_batches_tracked': 'models.135.bn91.num_batches_tracked',
        'neek.conv20.conv.0.weight': 'models.136.conv92.weight',
        'neek.conv20.conv.1.weight': 'models.136.bn92.weight',
        'neek.conv20.conv.1.bias': 'models.136.bn92.bias',
        'neek.conv20.conv.1.running_mean': 'models.136.bn92.running_mean',
        'neek.conv20.conv.1.running_var': 'models.136.bn92.running_var',
        'neek.conv20.conv.1.num_batches_tracked': 'models.136.bn92.num_batches_tracked',
        'head.conv1.conv.0.weight': 'models.137.conv93.weight',
        'head.conv1.conv.1.weight': 'models.137.bn93.weight',
        'head.conv1.conv.1.bias': 'models.137.bn93.bias',
        'head.conv1.conv.1.running_mean': 'models.137.bn93.running_mean',
        'head.conv1.conv.1.running_var': 'models.137.bn93.running_var',
        'head.conv1.conv.1.num_batches_tracked': 'models.137.bn93.num_batches_tracked',
        'head.conv2.conv.0.weight': 'models.138.conv94.weight',
        'head.conv2.conv.0.bias': 'models.138.conv94.bias',
        'head.conv3.conv.0.weight': 'models.141.conv95.weight',
        'head.conv3.conv.1.weight': 'models.141.bn95.weight',
        'head.conv3.conv.1.bias': 'models.141.bn95.bias',
        'head.conv3.conv.1.running_mean': 'models.141.bn95.running_mean',
        'head.conv3.conv.1.running_var': 'models.141.bn95.running_var',
        'head.conv3.conv.1.num_batches_tracked': 'models.141.bn95.num_batches_tracked',
        'head.conv4.conv.0.weight': 'models.143.conv96.weight',
        'head.conv4.conv.1.weight': 'models.143.bn96.weight',
        'head.conv4.conv.1.bias': 'models.143.bn96.bias',
        'head.conv4.conv.1.running_mean': 'models.143.bn96.running_mean',
        'head.conv4.conv.1.running_var': 'models.143.bn96.running_var',
        'head.conv4.conv.1.num_batches_tracked': 'models.143.bn96.num_batches_tracked',
        'head.conv5.conv.0.weight': 'models.144.conv97.weight',
        'head.conv5.conv.1.weight': 'models.144.bn97.weight',
        'head.conv5.conv.1.bias': 'models.144.bn97.bias',
        'head.conv5.conv.1.running_mean': 'models.144.bn97.running_mean',
        'head.conv5.conv.1.running_var': 'models.144.bn97.running_var',
        'head.conv5.conv.1.num_batches_tracked': 'models.144.bn97.num_batches_tracked',
        'head.conv6.conv.0.weight': 'models.145.conv98.weight',
        'head.conv6.conv.1.weight': 'models.145.bn98.weight',
        'head.conv6.conv.1.bias': 'models.145.bn98.bias',
        'head.conv6.conv.1.running_mean': 'models.145.bn98.running_mean',
        'head.conv6.conv.1.running_var': 'models.145.bn98.running_var',
        'head.conv6.conv.1.num_batches_tracked': 'models.145.bn98.num_batches_tracked',
        'head.conv7.conv.0.weight': 'models.146.conv99.weight',
        'head.conv7.conv.1.weight': 'models.146.bn99.weight',
        'head.conv7.conv.1.bias': 'models.146.bn99.bias',
        'head.conv7.conv.1.running_mean': 'models.146.bn99.running_mean',
        'head.conv7.conv.1.running_var': 'models.146.bn99.running_var',
        'head.conv7.conv.1.num_batches_tracked': 'models.146.bn99.num_batches_tracked',
        'head.conv8.conv.0.weight': 'models.147.conv100.weight',
        'head.conv8.conv.1.weight': 'models.147.bn100.weight',
        'head.conv8.conv.1.bias': 'models.147.bn100.bias',
        'head.conv8.conv.1.running_mean': 'models.147.bn100.running_mean',
        'head.conv8.conv.1.running_var': 'models.147.bn100.running_var',
        'head.conv8.conv.1.num_batches_tracked': 'models.147.bn100.num_batches_tracked',
        'head.conv9.conv.0.weight': 'models.148.conv101.weight',
        'head.conv9.conv.1.weight': 'models.148.bn101.weight',
        'head.conv9.conv.1.bias': 'models.148.bn101.bias',
        'head.conv9.conv.1.running_mean': 'models.148.bn101.running_mean',
        'head.conv9.conv.1.running_var': 'models.148.bn101.running_var',
        'head.conv9.conv.1.num_batches_tracked': 'models.148.bn101.num_batches_tracked',
        'head.conv10.conv.0.weight': 'models.149.conv102.weight',
        'head.conv10.conv.0.bias': 'models.149.conv102.bias',
        'head.conv11.conv.0.weight': 'models.152.conv103.weight',
        'head.conv11.conv.1.weight': 'models.152.bn103.weight',
        'head.conv11.conv.1.bias': 'models.152.bn103.bias',
        'head.conv11.conv.1.running_mean': 'models.152.bn103.running_mean',
        'head.conv11.conv.1.running_var': 'models.152.bn103.running_var',
        'head.conv11.conv.1.num_batches_tracked': 'models.152.bn103.num_batches_tracked',
        'head.conv12.conv.0.weight': 'models.154.conv104.weight',
        'head.conv12.conv.1.weight': 'models.154.bn104.weight',
        'head.conv12.conv.1.bias': 'models.154.bn104.bias',
        'head.conv12.conv.1.running_mean': 'models.154.bn104.running_mean',
        'head.conv12.conv.1.running_var': 'models.154.bn104.running_var',
        'head.conv12.conv.1.num_batches_tracked': 'models.154.bn104.num_batches_tracked',
        'head.conv13.conv.0.weight': 'models.155.conv105.weight',
        'head.conv13.conv.1.weight': 'models.155.bn105.weight',
        'head.conv13.conv.1.bias': 'models.155.bn105.bias',
        'head.conv13.conv.1.running_mean': 'models.155.bn105.running_mean',
        'head.conv13.conv.1.running_var': 'models.155.bn105.running_var',
        'head.conv13.conv.1.num_batches_tracked': 'models.155.bn105.num_batches_tracked',
        'head.conv14.conv.0.weight': 'models.156.conv106.weight',
        'head.conv14.conv.1.weight': 'models.156.bn106.weight',
        'head.conv14.conv.1.bias': 'models.156.bn106.bias',
        'head.conv14.conv.1.running_mean': 'models.156.bn106.running_mean',
        'head.conv14.conv.1.running_var': 'models.156.bn106.running_var',
        'head.conv14.conv.1.num_batches_tracked': 'models.156.bn106.num_batches_tracked',
        'head.conv15.conv.0.weight': 'models.157.conv107.weight',
        'head.conv15.conv.1.weight': 'models.157.bn107.weight',
        'head.conv15.conv.1.bias': 'models.157.bn107.bias',
        'head.conv15.conv.1.running_mean': 'models.157.bn107.running_mean',
        'head.conv15.conv.1.running_var': 'models.157.bn107.running_var',
        'head.conv15.conv.1.num_batches_tracked': 'models.157.bn107.num_batches_tracked',
        'head.conv16.conv.0.weight': 'models.158.conv108.weight',
        'head.conv16.conv.1.weight': 'models.158.bn108.weight',
        'head.conv16.conv.1.bias': 'models.158.bn108.bias',
        'head.conv16.conv.1.running_mean': 'models.158.bn108.running_mean',
        'head.conv16.conv.1.running_var': 'models.158.bn108.running_var',
        'head.conv16.conv.1.num_batches_tracked': 'models.158.bn108.num_batches_tracked',
        'head.conv17.conv.0.weight': 'models.159.conv109.weight',
        'head.conv17.conv.1.weight': 'models.159.bn109.weight',
        'head.conv17.conv.1.bias': 'models.159.bn109.bias',
        'head.conv17.conv.1.running_mean': 'models.159.bn109.running_mean',
        'head.conv17.conv.1.running_var': 'models.159.bn109.running_var',
        'head.conv17.conv.1.num_batches_tracked': 'models.159.bn109.num_batches_tracked',
        'head.conv18.conv.0.weight': 'models.160.conv110.weight',
        'head.conv18.conv.0.bias': 'models.160.conv110.bias',
    }
    pth_weights = torch.load(checkpoint)
    pt_weights = type(pth_weights)()
    for name, new_name in name_mapping.items():
        pt_weights[new_name] = pth_weights[name]
    return pt_weights


def convert_pt_checkpoint_to_keras_h5(state_dict):
    print('============================================================')

    def copy1(conv, bn, idx):
        keyword1 = 'conv%d.weight' % idx
        keyword2 = 'bn%d.weight' % idx
        keyword3 = 'bn%d.bias' % idx
        keyword4 = 'bn%d.running_mean' % idx
        keyword5 = 'bn%d.running_var' % idx
        for key in state_dict:
            value = state_dict[key].numpy()
            if keyword1 in key:
                w = value
            elif keyword2 in key:
                y = value
            elif keyword3 in key:
                b = value
            elif keyword4 in key:
                m = value
            elif keyword5 in key:
                v = value
        w = w.transpose(2, 3, 1, 0)
        conv.set_weights([w])
        bn.set_weights([y, b, m, v])

    def copy2(conv, idx):
        keyword1 = 'conv%d.weight' % idx
        keyword2 = 'conv%d.bias' % idx
        for key in state_dict:
            value = state_dict[key].numpy()
            if keyword1 in key:
                w = value
            elif keyword2 in key:
                b = value
        w = w.transpose(2, 3, 1, 0)
        conv.set_weights([w, b])

    num_classes = 80
    num_anchors = 3

    with tf.Session(graph=tf.Graph()):
        inputs = layers.Input(shape=[], dtype='string')
        model_body = YOLOv4(inputs, num_classes, num_anchors)
        model_body.summary()
        layer_name_to_idx = {layer.name: idx for idx, layer in enumerate(model_body.layers)}

        print('\nCopying...')
        i1 = layer_name_to_idx['conv2d']
        i2 = layer_name_to_idx['batch_normalization']
        copy1(model_body.layers[i1], model_body.layers[i2], 1)
        for i in range(2, 94, 1):
            i1 = layer_name_to_idx['conv2d_%d' % (i - 1)]
            i2 = layer_name_to_idx['batch_normalization_%d' % (i - 1)]
            copy1(model_body.layers[i1], model_body.layers[i2], i)
        for i in range(95, 102, 1):
            i1 = layer_name_to_idx['conv2d_%d' % (i - 1)]
            i2 = layer_name_to_idx['batch_normalization_%d' % (i - 2,)]
            copy1(model_body.layers[i1], model_body.layers[i2], i)
        for i in range(103, 110, 1):
            i1 = layer_name_to_idx['conv2d_%d' % (i - 1)]
            i2 = layer_name_to_idx['batch_normalization_%d' % (i - 3,)]
            copy1(model_body.layers[i1], model_body.layers[i2], i)

        i1 = layer_name_to_idx['conv2d_93']
        copy2(model_body.layers[i1], 94)
        i1 = layer_name_to_idx['conv2d_101']
        copy2(model_body.layers[i1], 102)
        i1 = layer_name_to_idx['conv2d_109']
        copy2(model_body.layers[i1], 110)

        weights = model_body.get_weights()
    print('\nDone.')
    return weights


class Mish(layers.Layer):

    def __init__(self):
        super(Mish, self).__init__()

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, x):
        return x * tf.tanh(tf.math.softplus(x))


def conv2d_unit(x, filters, kernels, strides=1, padding='valid', bn=1, act='mish'):
    use_bias = (bn != 1)
    x = layers.Conv2D(filters, kernels,
                      padding=padding,
                      strides=strides,
                      use_bias=use_bias,
                      activation='linear',
                      kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01))(x)
    if bn:
        x = layers.BatchNormalization(fused=False)(x)
    if act == 'leaky':
        x = keras.layers.LeakyReLU(alpha=0.1)(x)
    elif act == 'mish':
        x = Mish()(x)
    return x


def residual_block(inputs, filters_1, filters_2):
    x = conv2d_unit(inputs, filters_1, 1, strides=1, padding='valid')
    x = conv2d_unit(x, filters_2, 3, strides=1, padding='same')
    x = layers.add([inputs, x])
    return x


def stack_residual_block(inputs, filters_1, filters_2, n):
    x = residual_block(inputs, filters_1, filters_2)
    for i in range(n - 1):
        x = residual_block(x, filters_1, filters_2)
    return x


def spp(x):
    x_1 = x
    x_2 = layers.MaxPooling2D(pool_size=5, strides=1, padding='same')(x)
    x_3 = layers.MaxPooling2D(pool_size=9, strides=1, padding='same')(x)
    x_4 = layers.MaxPooling2D(pool_size=13, strides=1, padding='same')(x)
    out = layers.Concatenate()([x_4, x_3, x_2, x_1])
    return out


def YOLOv4(inputs, num_classes, num_anchors, input_shape=(608, 608), initial_filters=32,
           fast=False, anchors=None, conf_thresh=0.05, nms_thresh=0.45, keep_top_k=100, nms_top_k=100):
    i32 = initial_filters
    i64 = i32 * 2
    i128 = i32 * 4
    i256 = i32 * 8
    i512 = i32 * 16
    i1024 = i32 * 32

    x, image_shape = layers.Lambda(lambda t: preprocessor(t, input_shape))(inputs)

    # cspdarknet53
    x = conv2d_unit(x, i32, 3, strides=1, padding='same')

    # ============================= s2 =============================
    x = layers.ZeroPadding2D(padding=((1, 0), (1, 0)))(x)
    x = conv2d_unit(x, i64, 3, strides=2)
    s2 = conv2d_unit(x, i64, 1, strides=1)
    x = conv2d_unit(x, i64, 1, strides=1)
    x = stack_residual_block(x, i32, i64, n=1)
    x = conv2d_unit(x, i64, 1, strides=1)
    x = layers.Concatenate()([x, s2])
    s2 = conv2d_unit(x, i64, 1, strides=1)

    # ============================= s4 =============================
    x = layers.ZeroPadding2D(padding=((1, 0), (1, 0)))(s2)
    x = conv2d_unit(x, i128, 3, strides=2)
    s4 = conv2d_unit(x, i64, 1, strides=1)
    x = conv2d_unit(x, i64, 1, strides=1)
    x = stack_residual_block(x, i64, i64, n=2)
    x = conv2d_unit(x, i64, 1, strides=1)
    x = layers.Concatenate()([x, s4])
    s4 = conv2d_unit(x, i128, 1, strides=1)

    # ============================= s8 =============================
    x = layers.ZeroPadding2D(padding=((1, 0), (1, 0)))(s4)
    x = conv2d_unit(x, i256, 3, strides=2)
    s8 = conv2d_unit(x, i128, 1, strides=1)
    x = conv2d_unit(x, i128, 1, strides=1)
    x = stack_residual_block(x, i128, i128, n=8)
    x = conv2d_unit(x, i128, 1, strides=1)
    x = layers.Concatenate()([x, s8])
    s8 = conv2d_unit(x, i256, 1, strides=1)

    # ============================= s16 =============================
    x = layers.ZeroPadding2D(padding=((1, 0), (1, 0)))(s8)
    x = conv2d_unit(x, i512, 3, strides=2)
    s16 = conv2d_unit(x, i256, 1, strides=1)
    x = conv2d_unit(x, i256, 1, strides=1)
    x = stack_residual_block(x, i256, i256, n=8)
    x = conv2d_unit(x, i256, 1, strides=1)
    x = layers.Concatenate()([x, s16])
    s16 = conv2d_unit(x, i512, 1, strides=1)

    # ============================= s32 =============================
    x = layers.ZeroPadding2D(padding=((1, 0), (1, 0)))(s16)
    x = conv2d_unit(x, i1024, 3, strides=2)
    s32 = conv2d_unit(x, i512, 1, strides=1)
    x = conv2d_unit(x, i512, 1, strides=1)
    x = stack_residual_block(x, i512, i512, n=4)
    x = conv2d_unit(x, i512, 1, strides=1)
    x = layers.Concatenate()([x, s32])
    s32 = conv2d_unit(x, i1024, 1, strides=1)

    # fpn
    x = conv2d_unit(s32, i512, 1, strides=1, act='leaky')
    x = conv2d_unit(x, i1024, 3, strides=1, padding='same', act='leaky')
    x = conv2d_unit(x, i512, 1, strides=1, act='leaky')
    x = spp(x)

    x = conv2d_unit(x, i512, 1, strides=1, act='leaky')
    x = conv2d_unit(x, i1024, 3, strides=1, padding='same', act='leaky')
    fpn_s32 = conv2d_unit(x, i512, 1, strides=1, act='leaky')

    # pan01
    x = conv2d_unit(fpn_s32, i256, 1, strides=1, act='leaky')
    x = layers.UpSampling2D(2)(x)
    s16 = conv2d_unit(s16, i256, 1, strides=1, act='leaky')
    x = layers.Concatenate()([s16, x])
    x = conv2d_unit(x, i256, 1, strides=1, act='leaky')
    x = conv2d_unit(x, i512, 3, strides=1, padding='same', act='leaky')
    x = conv2d_unit(x, i256, 1, strides=1, act='leaky')
    x = conv2d_unit(x, i512, 3, strides=1, padding='same', act='leaky')
    fpn_s16 = conv2d_unit(x, i256, 1, strides=1, act='leaky')

    # pan02
    x = conv2d_unit(fpn_s16, i128, 1, strides=1, act='leaky')
    x = layers.UpSampling2D(2)(x)
    s8 = conv2d_unit(s8, i128, 1, strides=1, act='leaky')
    x = layers.Concatenate()([s8, x])
    x = conv2d_unit(x, i128, 1, strides=1, act='leaky')
    x = conv2d_unit(x, i256, 3, strides=1, padding='same', act='leaky')
    x = conv2d_unit(x, i128, 1, strides=1, act='leaky')
    x = conv2d_unit(x, i256, 3, strides=1, padding='same', act='leaky')
    x = conv2d_unit(x, i128, 1, strides=1, act='leaky')

    # output_s, doesn't need concat()
    output_s = conv2d_unit(x, i256, 3, strides=1, padding='same', act='leaky')
    output_s = conv2d_unit(output_s, num_anchors * (num_classes + 5), 1, strides=1, bn=0, act=None)

    # output_m, need concat()
    x = layers.ZeroPadding2D(padding=((1, 0), (1, 0)))(x)
    x = conv2d_unit(x, i256, 3, strides=2, act='leaky')
    x = layers.Concatenate()([x, fpn_s16])
    x = conv2d_unit(x, i256, 1, strides=1, act='leaky')
    x = conv2d_unit(x, i512, 3, strides=1, padding='same', act='leaky')
    x = conv2d_unit(x, i256, 1, strides=1, act='leaky')
    x = conv2d_unit(x, i512, 3, strides=1, padding='same', act='leaky')
    x = conv2d_unit(x, i256, 1, strides=1, act='leaky')
    output_m = conv2d_unit(x, i512, 3, strides=1, padding='same', act='leaky')
    output_m = conv2d_unit(output_m, num_anchors * (num_classes + 5), 1, strides=1, bn=0, act=None)

    # output_l, need concat()
    x = layers.ZeroPadding2D(padding=((1, 0), (1, 0)))(x)
    x = conv2d_unit(x, i512, 3, strides=2, act='leaky')
    x = layers.Concatenate()([x, fpn_s32])
    x = conv2d_unit(x, i512, 1, strides=1, act='leaky')
    x = conv2d_unit(x, i1024, 3, strides=1, padding='same', act='leaky')
    x = conv2d_unit(x, i512, 1, strides=1, act='leaky')
    x = conv2d_unit(x, i1024, 3, strides=1, padding='same', act='leaky')
    x = conv2d_unit(x, i512, 1, strides=1, act='leaky')
    output_l = conv2d_unit(x, i1024, 3, strides=1, padding='same', act='leaky')
    output_l = conv2d_unit(output_l, num_anchors * (num_classes + 5), 1, strides=1, bn=0, act=None)

    def cast_float32(tensor):
        return tf.cast(tensor, tf.float32)

    output_l = layers.Lambda(cast_float32)(output_l)
    output_m = layers.Lambda(cast_float32)(output_m)
    output_s = layers.Lambda(cast_float32)(output_s)

    # originally reshape in multi_thread_post
    output_lr = layers.Reshape((1, input_shape[0] // 32, input_shape[1] // 32, 3, 5 + num_classes))(output_l)
    output_mr = layers.Reshape((1, input_shape[0] // 16, input_shape[1] // 16, 3, 5 + num_classes))(output_m)
    output_sr = layers.Reshape((1, input_shape[0] // 8, input_shape[1] // 8, 3, 5 + num_classes))(output_s)

    # originally _yolo_out
    masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    anchors = [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55],
               [72, 146], [142, 110], [192, 243], [459, 401]]

    def batch_process_feats(out, anchors, mask):
        grid_h, grid_w, num_boxes = map(int, out.shape[2:5])

        anchors = [anchors[i] for i in mask]
        anchors_tensor = np.array(anchors).reshape(1, 1, len(anchors), 2)

        # Reshape to batch, height, width, num_anchors, box_params.
        box_xy = tf.sigmoid(out[..., :2])
        box_wh = tf.exp(out[..., 2:4])
        box_wh = box_wh * anchors_tensor

        box_confidence = tf.sigmoid(out[..., 4])
        box_confidence = tf.expand_dims(box_confidence, axis=-1)
        box_class_probs = tf.sigmoid(out[..., 5:])

        col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
        row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)

        col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
        row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
        grid = np.concatenate((col, row), axis=-1).astype(np.float32)

        box_xy += grid
        box_xy /= (grid_w, grid_h)
        box_wh /= input_shape
        box_xy -= (box_wh / 2.)  # normalized xywh
        boxes = tf.concat((box_xy, box_xy + box_wh), axis=-1)

        box_scores = box_confidence * box_class_probs
        num_boxes = np.prod(boxes.shape[1:-1])
        boxes = tf.reshape(boxes, [-1, num_boxes, boxes.shape[-1]])
        box_scores = tf.reshape(box_scores, [-1, num_boxes, box_scores.shape[-1]])
        return boxes, box_scores

    def filter_boxes(outputs):
        boxes_l, boxes_m, boxes_s, box_scores_l, box_scores_m, box_scores_s, image_shape = outputs
        boxes_l, box_scores_l = filter_boxes_one_size(boxes_l, box_scores_l)
        boxes_m, box_scores_m = filter_boxes_one_size(boxes_m, box_scores_m)
        boxes_s, box_scores_s = filter_boxes_one_size(boxes_s, box_scores_s)
        boxes = tf.concat([boxes_l, boxes_m, boxes_s], axis=0)
        box_scores = tf.concat([box_scores_l, box_scores_m, box_scores_s], axis=0)
        image_shape_wh = image_shape[1::-1]
        image_shape_whwh = tf.concat([image_shape_wh, image_shape_wh], axis=-1)
        image_shape_whwh = tf.cast(image_shape_whwh, tf.float32)
        boxes *= image_shape_whwh
        boxes = tf.expand_dims(boxes, 0)
        box_scores = tf.expand_dims(box_scores, 0)
        boxes = tf.expand_dims(boxes, 2)
        nms_boxes, nms_scores, nms_classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes,
            box_scores,
            max_output_size_per_class=nms_top_k,
            max_total_size=nms_top_k,
            iou_threshold=nms_thresh,
            score_threshold=conf_thresh,
            pad_per_class=False,
            clip_boxes=False,
            name='CombinedNonMaxSuppression',
        )
        return nms_boxes[0], nms_scores[0], nms_classes[0]

    def filter_boxes_one_size(boxes, box_scores):
        box_class_scores = tf.reduce_max(box_scores, axis=-1)
        keep = box_class_scores > conf_thresh
        boxes = boxes[keep]
        box_scores = box_scores[keep]
        return boxes, box_scores

    def batch_yolo_out(outputs):
        with tf.name_scope('yolo_out'):
            b_output_lr, b_output_mr, b_output_sr, b_image_shape = outputs
            with tf.name_scope('process_feats'):
                b_boxes_l, b_box_scores_l = batch_process_feats(b_output_lr, anchors, masks[0])
            with tf.name_scope('process_feats'):
                b_boxes_m, b_box_scores_m = batch_process_feats(b_output_mr, anchors, masks[1])
            with tf.name_scope('process_feats'):
                b_boxes_s, b_box_scores_s = batch_process_feats(b_output_sr, anchors, masks[2])
            with tf.name_scope('filter_boxes'):
                b_nms_boxes, b_nms_scores, b_nms_classes = tf.map_fn(
                    filter_boxes, [b_boxes_l, b_boxes_m, b_boxes_s, b_box_scores_l, b_box_scores_m, b_box_scores_s, b_image_shape],
                    dtype=(tf.float32, tf.float32, tf.float32), back_prop=False, parallel_iterations=16)
        return b_nms_boxes, b_nms_scores, b_nms_classes

    boxes_scores_classes = layers.Lambda(batch_yolo_out)([output_lr, output_mr, output_sr, image_shape])

    model_body = keras.models.Model(inputs=inputs, outputs=boxes_scores_classes)
    return model_body


def decode_jpeg_resize(input_tensor, image_size):
    tensor = tf.image.decode_png(input_tensor, channels=3)
    shape = tf.shape(tensor)
    tensor = tf.cast(tensor, tf.float32)
    tensor = tf.image.resize(tensor, image_size)
    tensor /= 255.0
    return tf.cast(tensor, tf.float16), shape


def preprocessor(input_tensor, image_size):
    with tf.name_scope('Preprocessor'):
        tensor = tf.map_fn(
            partial(decode_jpeg_resize, image_size=image_size), input_tensor,
            dtype=(tf.float16, tf.int32), back_prop=False, parallel_iterations=16)
    return tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    args = parser.parse_args()
    if os.path.exists(args.model_dir):
        raise OSError('Directory {} already exists; please specify a different path for the tensorflow SavedModel'.format(args.model_dir))
    print('Downloading YOLO v4 checkpoint from https://docs.google.com/uc?id=1wv_LiFeCRYwtpkqREPeI13-gPELBDwuJ')
    checkpoint = download_file_from_google_drive('1wv_LiFeCRYwtpkqREPeI13-gPELBDwuJ')
    torch_weights = rename_weights(checkpoint)
    keras_weights = convert_pt_checkpoint_to_keras_h5(torch_weights)
    keras.backend.set_learning_phase(0)
    num_anchors = 3
    num_classes = 80
    input_shape = (608, 608)
    conf_thresh = 0.001
    nms_thresh = 0.45
    inputs = layers.Input(shape=[], dtype='string')
    yolo = YOLOv4(inputs, num_classes, num_anchors, input_shape, conf_thresh=conf_thresh, nms_thresh=nms_thresh)
    yolo.set_weights(keras_weights)
    sess = keras.backend.get_session()
    inputs = {'image': yolo.inputs[0]}
    output_names = ['boxes', 'scores', 'classes']
    outputs = {name: ts for name, ts in zip(output_names, yolo.outputs)}
    print('Saving YOLO v4 tensorflow SavedModel as {}'.format(args.model_dir))
    tf.saved_model.simple_save(sess, args.model_dir, inputs, outputs)


if __name__ == '__main__':
    main()
