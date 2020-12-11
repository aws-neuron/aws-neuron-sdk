import numpy as np
import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import time

if __name__ == '__main__':
    channel = grpc.insecure_channel('localhost:9000')
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'bert_mrpc_hc_gelus_b4_l24_0926_02'
    i = np.zeros([1, 128], dtype=np.int32)
    request.inputs['input_ids'].CopyFrom(tf.contrib.util.make_tensor_proto(i, shape=i.shape))
    request.inputs['input_mask'].CopyFrom(tf.contrib.util.make_tensor_proto(i, shape=i.shape))
    request.inputs['segment_ids'].CopyFrom(tf.contrib.util.make_tensor_proto(i, shape=i.shape))

    latencies = []
    for i in range(100):
        start = time.time()
        result = stub.Predict(request)
        latencies.append(time.time() - start)
        print("Inference successful: {}".format(i))

    print ("Ran {} inferences successfully. Latency average = {}".format(len(latencies), np.average(latencies)))
