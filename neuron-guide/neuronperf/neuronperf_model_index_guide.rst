.. _neuronperf_model_index_guide:

============================
NeuronPerf Model Index Guide
============================

A **model index** is a JSON file that tracks information about one or more compiled models. You can generate them using ``compile``, by using the API described here, or you may create them manually in a text editor.

After a call to ``compile`` you may notice that you now have a ``models`` directory. You will also spot a new file named something like ``model_83b3raj2.json`` in your local directory, if you didn't provide a ``filename`` yourself.

A model index is not intended to be opaque; you should feel free to open, inspect, and modify it yourself. It contains some information about the artifacts that were compiled. Individual models referenced by the index can be handed to ``benchmark`` directly along with an example input, or you may pass the entire index as in the basic example above. Here is an example index:

.. code:: bash

   python3 -m json.tool model_index.json

.. code:: json

   {
       "version": "0.0.0.0+0bc220a",
       "model_configs": [
           {
               "filename": "models/model_b1_p1_38793jda.pt",
               "input_idx": 0,
               "batch_size": 1,
               "pipeline_size": 1,
               "compile_s": 5.32
           }
       ]
   }

An index is useful for keeping track of your compiled artifacts and their parameters. The advantages of using ``neuronperf.[torch/tensorflow/mxnet].compile`` are clearer when we wish to compile multiple variants of our model and benchmark all of them at the same time. All of the model artifacts and the index can be destroyed using ``model_index.delete('model_index.json')``.

Benchmarking
============

When benchmarking with an index, there are some important details to keep in mind. If you originally built the index using a set of inputs, the model index has associated the ``inputs`` with the compiled models by their positional index.

For example:

.. code:: python

   batch_sizes = [1, 2]
   inputs = [torch.zeros((b, 100)) for b in batch_sizes]

Here, ``inputs[0]`` corresponds to batch size 1. Therefore, the model index will contain a reference to input 0 for that model. When you call ``benchmark``, you must pass inputs with the same shape in the same positions as at compile time.

.. note::

   It's only necessary that there is an input with the correct shape at``inputs[input_index]``. The example data itself is not important.


Working with Indexes
--------------------

The API detail below describes utilities for working with indexes. An ``index`` can be either a loaded index (JSON) or the path to an index (it will be loaded automatically).

Creating
========

.. code:: python

   index = neuronperf.model_index.create('/path/to/model', batch_size=1)
   filename = neuronperf.model_index.save(index)

Once you have an index, you can pass its path directly to ``benchmark``. You can also pass a custom filename instead:

.. code:: python

   index = neuronperf.model_index.create('/path/to/model', batch_size=1)
   neuronperf.model_index.save(index, 'my_index.json')

Appending
=========

If **multiple models use the same inputs**, you can append them together. For example, if you have the same batch size with multiple pipeline sizes, the inputs are the same, but the model changes.

.. code:: python

   pipeline_sizes = [1, 2, 3, 4]
   indexes = [neuronperf.model_index.create(f'/path/to/model_p{p}', pipeline_size=p, batch_size=5) for p in pipeline_sizes]
   index = neuronperf.model_index.append(*indexes)
   neuronperf.model_index.save(index, 'my_index.json')

Filtering
=========

You can construct a new model index that is filtered by some parameter. For example, to get a new index with only batch sizes [1, 2], you could do:

.. code:: python

   new_index = neuronperf.model_index.filter(index, batch_sizes=[1, 2])

You can also benchmark subset of a model index by passing only the subset parameters of interest, but remember to ensure you provide the correct number of inputs for the index (even if some are not used).

For example, if you an index with models at ``batch_sizes = [1, 2, 3]``, but only wish to benchmark batch size 2:

.. code:: python

   batch_sizes = [1, 2, 3]
   inputs = [torch.zeros((b, 100)) for b in batch_sizes]
   reports = neuronperf.torch.benchmark('model_index.json', inputs, batch_sizes=2)

Copying
=======

You can copy an index to a new location with ``neuronperf.model_index.copy(index, new_index_name, new_index_dir)``. This is mostly useful in combination with ``filter``/``append``.

Deleting
========

If you wish to keep your compiled models, just delete the model index file yourself. If you want to delete your model index and all associated artifacts, use:

.. code:: python

   neuronperf.model_index.delete('my_index.json')