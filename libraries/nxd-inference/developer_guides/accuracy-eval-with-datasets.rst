.. _accuracy-eval-with-datasets:

Accuracy Evaluation of Models on Neuron Using Open Source Datasets
============================================

This guide demonstrates how to evaluate accuracy of models on Trainium and Inferentia instances using open source datasets. 
This approach expands on the accuracy evaluation using logits and enables you to evaluate accuracy using open source datasets 
like MMLU and GSM8K for tasks such as instruction following and mathematical reasoning.

Under the hood, this accuracy suite uses vLLM server to serve the model
and can use benchmarking clients such as `lm-eval <https://github.com/EleutherAI/lm-evaluation-harness>`__ 
and `LongBench <https://github.com/THUDM/LongBench>`__ to evaluate on their supported datasets. 
In future we will add support for other benchmarking clients. 

The code used in this guide is located at https://github.com/aws-neuron/aws-neuron-samples/tree/master/inference-benchmarking/

For a tutorial that you can follow and run on a trainium or inferentia instance please look at :ref:`Evaluating Accuracy of Llama-3.1-70B on Neuron using open source datasets <nxdi-trn1-llama3.1-70b-instruct-accuracy-eval-tutorial>`.

Configuration Setup
-------------------

Creating the Configuration File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a test_config.yaml file that defines your server settings and
accuracy test configurations:

.. code:: yaml

   server:
     name: "test-model-server"
     model_path: "/path/to/model"
     model_s3_path: "s3://bucket/path/to/model"
     max_seq_len: 2048
     context_encoding_len: 1024
     tp_degree: 2
     n_vllm_threads: 16
     server_port: 8000
     continuous_batch_size: 2

   test:
     accuracy:
       mmlu_test:
         client: "lm_eval"
         datasets: ["mmlu"]
         max_concurrent_requests: 1
         timeout: 3600
         client_params:
           limit: 100
       
       longbench_test:
         client: "longbench"
         datasets: ["qasper", "multifieldqa"]
         max_concurrent_requests: 1
         timeout: 7200
         client_params:
           max_length: 4096

Configuration Parameters
------------------------

Server Configuration
~~~~~~~~~~~~~~~~~~~~

========================= ================================
Parameter                 Description
========================= ================================
``name``                  Identifier for your model server
``model_path``            Local path to model files
``model_s3_path``         S3 location of model files
``max_seq_len``           Maximum sequence length
``context_encoding_len``  Length of context encoding
``tp_degree``             Tensor parallelism degree
``n_vllm_threads``        Number of vLLM threads
``server_port``           Server port number
``continuous_batch_size`` Size of continuous batches
========================= ================================

if ``model_s3_path`` is specified, the model will be downloaded into ``model_path``,
otherwise model should already exist in ``model_path``.

Accuracy Test Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

+-----------------------------+---------------------------------------+
| Parameter                   | Description                           |
+=============================+=======================================+
| ``client``                  | Evaluation framework (e.g.,           |
|                             | “lm_eval”, “longbench”)               |
+-----------------------------+---------------------------------------+
| ``datasets``                | List of datasets for evaluation       |
|                             | from the supported set by the client  |
+-----------------------------+---------------------------------------+
| ``max_concurrent_requests`` | Maximum parallel requests             |
+-----------------------------+---------------------------------------+
| ``timeout``                 | Maximum execution time (seconds)      |
+-----------------------------+---------------------------------------+
| ``client_params``           | Client-specific parameters            |
+-----------------------------+---------------------------------------+

Running Evaluations
-------------------

Execute accuracy tests using the CLI command:

.. code:: bash

   python accuracy.py --config test_config.yaml



For more detailed information and advanced configurations, please refer
to: - `lm-eval
Documentation <https://github.com/EleutherAI/lm-evaluation-harness>`__ -
`LongBench Documentation <https://github.com/THUDM/LongBench>`__

These resources provide comprehensive guides on client-specific
parameters and advanced evaluation scenarios.
