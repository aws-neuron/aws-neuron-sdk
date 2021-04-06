from concurrent import futures
import torch
import torch.neuron
import os
from time import time
from queue import Queue

def consumer(model, input_queue):
    while True:
        inputs, input_id, callback_fn = input_queue.get()
        input_queue.task_done()
        # Stop execution if stopping condition is recieved
        if inputs == "stop":
            break
        start = time()
        results = model(*inputs)
        # Make the output iterable - if it is not already a tuple or list
        if not isinstance(results, tuple) or isinstance(results, list):
            results = [results]
        end = time()
        if callback_fn is not None:
            callback_fn(results, input_id, start, end)
              
class NeuronSimpleDataParallel():

    def __init__(self, model_file, num_neuron_cores, batch_size=1):
        self.num_neuron_cores = num_neuron_cores
        self.batch_size = batch_size
        
        # Important - please read:
        # https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/neuron-frameworks/tensorflow-neuron/tutorials/tutorial-tensorflow-NeuronCore-Group.html
        # For four cores we use
        ##     os.environ['NEURONCORE_GROUP_SIZES'] = "1,1,1,1"
        # when launching four threads
        # In this logic exists in worker processes, each process should use
        ##     os.environ['NEURONCORE_GROUP_SIZES'] = "1"
        nc_env = ','.join(['1'] * num_neuron_cores)
        os.environ['NEURONCORE_GROUP_SIZES'] = nc_env
        
        # Construct a list of models
        self.models = [torch.jit.load(model_file)
                       for i in range(num_neuron_cores)]
        
        # Create shared input queue
        self.input_queue = Queue(maxsize=num_neuron_cores*16)

        self.executor = futures.ThreadPoolExecutor(
            max_workers=num_neuron_cores)

    def eval(self):
        for model in self.models:
            model.eval()
            
    def train(self):
        for model in self.models:
            model.train()
            
    def start_continuous_inference(self):
        for model in self.models:
            self.executor.submit(consumer, model, self.input_queue)
    
    def infer(self, batch, input_id, callback_fn):
        self.input_queue.put((batch, input_id, callback_fn))
        
    def stop(self):
        for _ in range(self.num_neuron_cores):
            self.input_queue.put(("stop", -1, None))