.. _performance-cli-params:

Evaluating Performance of Models on Neuron Using LLMPerf
============================================

This topic guides you through determining the performance of your models on Trainium and Inferentia instances using  open-source clients.
It expands on the basic performance analysis tools provided with Neuron by incorporating the `LLMperf <https://github.com/ray-project/llmperf>`_ client to collect additional information about performance for models such as llama-3.3-70B-instruct and llama-3.1-8b.


Under the hood, this performance suite uses vLLM server to serve the model
and can use benchmarking clients such as `llm-perf <https://github.com/ray-project/llmperf>`_
to evaluate on their supported models. 

In the future we will add support for other benchmarking clients. 

The code used in this guide is located at `inference-benchmarking <https://github.com/aws-neuron/aws-neuron-samples/tree/master/inference-benchmarking/>`_.

For a tutorial that you can follow and run on a Trainium or Inferentia instance, see :ref:`/libraries/nxd-inference/tutorials/generating-results-with-performance-cli.ipynb`. 



Creating the Configuration File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a test_config.yaml file that defines your server settings and
performance test configurations and paste in the following code:

.. code:: yaml

   server:
     name: "test-model-server"
     model_path: "/path/to/model"
     model_s3_path: "s3://bucket/path/to/model"
     max_seq_len: 256
     context_encoding_len: 128
     tp_degree: 32
     server_port: 8000
     continuous_batch_size: 1
     custom_chat_template_path: "default"

   test:
     performance:
       llama_test:
         client: "llm_perf"
         client_type: "llm_perf_github_patched"
         max_concurrent_requests: 20
         timeout: 3600
         input_size: 128
         output_size: 124
         client_params:
           stddev_input_tokens: 0
           stddev_output_tokens: 1
       


Configuration Parameters
------------------------

Below is a reference for the configuration parameters you can use when configuring the server and tastes for your model performance analysis:

Server Configuration
~~~~~~~~~~~~~~~~~~~~

===================================== ===================================
Parameter                               Description
===================================== ===================================
``name``                              Identifier for your model server
``model_path``                        Local path to model files
``model_s3_path``                     S3 location of model files
``max_seq_len``                       Maximum sequence length
``context_encoding_len``              Length of context encoding
``tp_degree``                         Tensor parallelism degree
``server_port``                       Server port number
``continuous_batch_size``             Size of continuous batches
``custom_chat_template_path``         Chat template for the prompt
===================================== ===================================

if ``model_s3_path`` is specified, the model is downloaded to ``model_path``;
otherwise, the model should already be available at ``model_path``.

Performance Test Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

+-----------------------------+---------------------------------------+
| Parameter                   | Description                           |
+=============================+=======================================+
| ``client``                  | Performance framework (such as,       |
|                             | llm-perf)                             |
+-----------------------------+---------------------------------------+
| ``client_type``             | List of clients such as               |
|                             |  llm_perf_github_patched              |
+-----------------------------+---------------------------------------+
| ``max_concurrent_requests`` | Maximum parallel requests             |
+-----------------------------+---------------------------------------+
| ``timeout``                 | Maximum execution time (seconds)      |
+-----------------------------+---------------------------------------+
| ``input_size``              | Input context length                  |
+-----------------------------+---------------------------------------+
| ``output_size``             | Output length / MaxNewTokens          |
+-----------------------------+---------------------------------------+
| ``client_params``           | Client-specific parameters            |
+-----------------------------+---------------------------------------+

Client_params
-------------------

Involves ``stddev_input_tokens`` and ``stddev_output_tokens``

To prevent bucket overflow at higher batch sizes, we use the following default:

``outputlength`` = ``orig_output_length - 4* continuous_batch_size``

``stddev_output_tokens`` = ``batch_size``


Running Evaluations
-------------------

Execute performance tests using the CLI command:

.. code:: bash

   python performance.py --config perf.yaml



For more detailed information and advanced configurations, please refer
to: - `llm-perf
Documentation <https://github.com/ray-project/llmperf>`__ -


These resources provide comprehensive guides on client-specific
parameters and advanced evaluation scenarios.
