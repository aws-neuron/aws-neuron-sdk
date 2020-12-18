from concurrent import futures
import torch
import torch.neuron
import os


class NeuronSimpleDataParallel():

    def __init__(self, model_file, num_neuron_cores, batch_size=1):
        # Construct a list of models
        self.num_neuron_cores = num_neuron_cores
        self.batch_size = batch_size

        class SimpleWrapper():

            def __init__(self, model):
                self.model = model

            def eval(self):
                self.model.eval()

            def train(self):
                self.model.train()

            def __call__(self, *args):
                results = self.model(*args)

                # Make the output iterable - if it is not already a tuple or list
                if not isinstance(results, tuple) or isinstance(results, list):
                    results = [results]

                return results

        self.models = [SimpleWrapper(torch.jit.load(model_file))
                       for i in range(num_neuron_cores)]

        # Important - please read:
        # https://github.com/aws/aws-neuron-sdk/blob/master/docs/tensorflow-neuron/tutorial-NeuronCore-Group.md
        # For four cores we use
        ##     os.environ['NEURONCORE_GROUP_SIZES'] = "1,1,1,1"
        # when launching four threads
        # In this logic exists in worker processes, each process should use
        ##     os.environ['NEURONCORE_GROUP_SIZES'] = "1"
        nc_env = ','.join(['1'] * num_neuron_cores)
        os.environ['NEURONCORE_GROUP_SIZES'] = nc_env

        self.executor = futures.ThreadPoolExecutor(
            max_workers=self.num_neuron_cores)

    def eval(self):
        for m in self.models:
            m.eval()

    def train(self):
        for m in self.models:
            m.train()

    def __call__(self, *args):
        assert all(isinstance(a, torch.Tensor)
                   for a in args), "Non tensor input - tensors are needed to generate batches"
        assert all(a.shape[0] % self.num_neuron_cores ==
                   0 for a in args), "Batch size must be even multiple of the number of parallel neuron cores"

        args_per_core = [[] for i in range(self.num_neuron_cores)]

        # Split args
        for a in args:
            # Based on batch size for arg
            step_size = a.shape[0] // self.num_neuron_cores
            for i in range(self.num_neuron_cores):
                # Append a slice of a view
                start = i * step_size
                end = (i + 1) * step_size

                # Slice
                args_per_core[i].append(a[start:end])

        # Call each core with their split and wait to complete
        running = {self.executor.submit(
            self.models[idx], *args_per_core[idx]): idx for idx in range(self.num_neuron_cores)}

        results = [None] * self.num_neuron_cores

        for future in futures.as_completed(running):
            idx = running[future]

            results[idx] = future.result()

        # Remove zero dimensional tensors (unsqueeze)
        # Iterate results per core
        for ic in range(len(results)):
            # Iterate result tuples
            for ir in range(len(results[ic])):
                # Unsqueeze if zero dimensional or does not look batched (i.e. first dim does not match batch)
                if len(results[ic][ir].size()) == 0 or results[ic][ir].shape[0] != self.batch_size:
                    results[ic][ir] = torch.unsqueeze(
                        results[ic][ir], 0)

        # Concatenate
        output = results[0][0]

        for i in range(1, len(results)):
            for j in range(len(results[i])):
                output = torch.cat([output, results[i][j]], 0)

        return output
