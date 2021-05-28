import mxnet as mx
import mx_neuron
import os
from time import time
from queue import Queue
from multiprocessing import Process, Manager


def consumer(model_file, sample_input, input_queue, result_queue):
    sym, args, aux = mx.model.load_checkpoint(model_file, 0)
    sample_input = {key: mx.nd.array(v) for key, v in sample_input.items()}
    args.update(sample_input)
    model = sym.bind(mx.cpu(), args=args, aux_states=aux, grad_req="null")

    while True:
        inputs, input_id = input_queue.get()
        input_queue.task_done()
        # Stop execution if stopping condition is recieved
        if inputs == "stop":
            break
        inputs = {key: mx.nd.array(v) for key, v in inputs.items()}
        start = time()
        results = model.forward(**inputs)
        results[0].wait_to_read()

        # Make the output iterable - if it is not already a tuple or list
        if not isinstance(results, tuple) or isinstance(results, list):
            results = [results]
        end = time()

        if input_id != -1:
            result_queue.put((results, start, end, input_id))


class NeuronSimpleDataParallel:
    def __init__(self, model_file, num_neuron_cores, sample_input):
        self.num_neuron_cores = num_neuron_cores
        self.sample_input = sample_input
        self.model_path = model_file
        # Create shared input queue and output queue
        manager = Manager()
        self.input_queue = manager.Queue(maxsize=num_neuron_cores * 16)
        self.result_queue = manager.Queue(maxsize=num_neuron_cores * 16)

        self.processes = [
            Process(
                target=consumer,
                args=(
                    self.model_path,
                    self.sample_input,
                    self.input_queue,
                    self.result_queue,
                ),
            )
            for _ in range(num_neuron_cores)
        ]
        self.input_id = 0
        self.input_dict = set()

    def start_continuous_inference(self):
        for p in self.processes:
            p.start()

    def warmup(self, batch):
        self.input_queue.put((batch, -1))

    def infer(self, batch):
        self.input_id += 1
        self.input_dict.add(self.input_id)
        self.input_queue.put((batch, self.input_id))

    def stop(self):
        for _ in range(self.num_neuron_cores):
            self.input_queue.put(("stop", -1))

    def add_result(self, callback_fn):
        if not self.result_queue.empty():
            result, start, end, input_id = self.result_queue.get()
            self.input_dict.remove(input_id)
            self.result_queue.task_done()
            callback_fn(result, start, end)

    def add_all_results(self, callback_fn):
        results = []
        while len(self.input_dict):
            self.add_result(callback_fn)
        for p in self.processes:
            p.join()
