.. _nxdi-trn1-llama3.1-70b-instruct-accuracy-eval-tutorial:

Tutorial: Evaluating Accuracy of Llama-3.1-70B on Neuron using open source datasets
============================================

Introduction
------------

This tutorial provides a step-by-step
guide to measure the accuracy of Llama3.1 70B on Trn1 with evaluation on
two distinct tasks: mathematical reasoning and logical analysis.

For this tutorial we use two datasets available in lm-eval, namely
``gsm8k_cot`` (high school math questions) and ``mmlu_flan_n_shot_generative_logical_fallacies`` (multiple choice questions on the subject) to
demonstrate accuracy evaluation on Trn1. 
The metrics in these task are two variants of `ExactMatch <https://huggingface.co/spaces/evaluate-metric/exact_match>`__ metrics called StrictMatch and FlexibleExtract which differ in how strict they are
in extracting the final answer from the generated output from the model. To see the exact task definition used in lm-eval please look at `gsm8k-cot <https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/gsm8k-cot.yaml>`__ 
and `mmlu template <https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/mmlu/flan_n_shot/generative/_mmlu_flan_generative_template_yaml>`__.

We also need the instruction-tuned version of llama-3.1 70b
`meta-llama/Llama-3.1-70B-Instruct <https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct>`__
available hugging face. 


Task Overview
~~~~~~~~~~~~~

.. _1-gsm8k-with-chain-of-thought-gsm8k_cot:

1. GSM8K with Chain-of-Thought (gsm8k_cot)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The GSM8K dataset focuses on grade school math word problems, testing
LLMs' mathematical reasoning capabilities. Using Chain-of-Thought (CoT)
prompting, we evaluate models' ability to:

- Solve complex math word problems
- Show step-by-step reasoning
- Arrive at accurate numerical answers

.. _2-mmlu-logical-fallacies-mmlu_flan_n_shot_generative_logical_fallacies:

2. MMLU Logical Fallacies (mmlu_flan_n_shot_generative_logical_fallacies)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This evaluation focuses on the model's ability to identify and explain
logical fallacies, a subset of the MMLU benchmark. The task tests:

- Understanding of common logical fallacies
- Ability to analyze arguments
- Explanation of reasoning flaws

Environment Setup Guide
-----------------------

Prerequisites
~~~~~~~~~~~~~

This tutorial requires that you have a Trn1 instance created from a Deep
Learning AMI that has the Neuron SDK pre-installed. Also we depend on
our fork of vLLM
`aws-neuron/upstreaming-to-vllm <https://github.com/aws-neuron/upstreaming-to-vllm/tree/v0.6.x-neuron>`__.

Before running evaluations, ensure your environment is properly
configured by following these essential setup guides:

1. `NxD Inference Setup
   Guide <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/nxdi-setup.html>`__

   - Configure AWS Neuron environment
   - Set up required dependencies
   - Verify system requirements

2. `vLLM User Guide for NxD
   Inference <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/developer_guides/vllm-user-guide.html>`__

   - Setup vLLM according to the guide

Installing dependencies
~~~~~~~~~~~~~~~~~~~~~~

Copy the
`inference-benchmarkin <https://github.com/aws-neuron/aws-neuron-samples-staging/tree/aws-neuron-eval/inference-benchmarking/>`__
directory to some location on your instance. Change directory to the
your copy of
`inference-benchmarkin <https://github.com/aws-neuron/aws-neuron-samples-staging/tree/aws-neuron-eval/inference/benchmarking/aws-neuron-eval>`__. Install other required dependencies in the same python env (e.g
aws_neuron_venv_pytorch if you followed `manual install NxD
Inference <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/nxdi-setup.html#id3>`__
) by:

.. code:: python

   !pip install -r requirements.txt

Download llama-3.1 70B
~~~~~~~~~~~~~~~~~~~~~~

To use this sample, you must first download
`meta-llama/Llama-3.1-70B-Instruct <https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct>`__
model checkpoint from Hugging Face `/home/ubuntu/models/Llama-3.1-70B-Instruct/` on the Trn1 instance.
For more information, see Downloading models in the Hugging Face
documentation.

Running Evaluations
-------------------

There are two methods that you can use `the evaluation
scirpts <https://github.com/aws-neuron/aws-neuron-samples-staging/tree/aws-neuron-eval/inference/benchmarking/aws-neuron-eval>`__
to run your evaluation.

1. Using a yaml configuration file and ``accuracy.py`` script
2. writing your own python script that uses several components provided
   in ``accuracy.py`` and ``server_config.py``

We demonstrate each use case separately here. 

1. Running eval with yaml config file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this method all you need is to create a yaml config file that
specifies the server configuration and testing scenario you want to run.
Create ``config.yaml`` with the following content.

.. code:: yaml

   server:
     name: "Llama-3.1-70B-Instruct"
     model_path: "/home/ubuntu/models/Llama-3.1-70B-Instruct/"
     model_s3_path: null
     compiled_model_path: "/home/ubuntu/traced_models/Llama-3.1-70B-Instruct"
     max_seq_len: 16384
     context_encoding_len: 16384
     tp_degree: 32
     n_vllm_threads: 32
     server_port: 8000
     continuous_batch_size: 1

   test:
     accuracy:
       mytest:
         client: "lm_eval"
         datasets: ["gsm8k_cot", "mmlu_flan_n_shot_generative_logical_fallacies"]
         max_concurrent_requests: 1
         timeout: 3600
         client_params:
           limit: 200
           use_chat: True

For tasks that require higher sequence length you need to adjust ``max_seq_len``. For the tasks in this tutorial 16384 would suffice.

Run ``python accuracy.py --config config.yaml``

2. Running eval through your own python code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You might be interested in running the evaluation in you python code.
For instance if you want to change the configuration programatically or
post-process the results. This is possible using 3 main components
provided in ``accuracy.py`` and ``server_config.py``.

1. Server Configuration: Using ServerConfig to define the vLLM server
   settings
2. Accuracy Scenario: Using AccuracyScenario to specify evaluation
   parameters
3. Test Execution: Running the evaluation with the configured settings

Step-by-Step Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, import the necessary components:

.. code:: python

   from accuracy import AccuracyScenario, run_accuracy_test
   from server_config import ServerConfig

.. _1-configure-the-server:

1. Configure the Server
^^^^^^^^^^^^^^^^^^^^^^^

Set up your server configuration with ServerConfig. This example uses
Llama 3.1-8b Instruct:

.. code:: python

   name = "Llama-3.1-70B-Instruct"
   server_config = ServerConfig(
       name=name,
       model_path=f"/home/ubuntu/models/{name}",  # Local model path
       model_s3_path=None,  # S3 model path
       max_seq_len=16384,          # Maximum sequence length
       context_encoding_len=16384,  # Context window size
       tp_degree=32,               # Tensor parallel degree
       n_vllm_threads=32,          # Number of vLLM threads
       server_port=8000,           # Server port
       continuous_batch_size=1,    # Batch size for continuous batching
   )

.. _2-define-the-evaluation-scenario:

2. Define the Evaluation Scenario
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create an AccuracyScenario to specify your evaluation parameters:

.. code:: python

   scenario = AccuracyScenario(
       client="lm_eval",          # Evaluation client
       datasets=[                 # Target datasets
           "gsm8k_cot",
           "mmlu_flan_n_shot_generative_logical_fallacies",
       ],
       max_concurrent_requests=1,  # Maximum concurrent requests
       timeout=3600,              # Timeout in seconds
       client_params={"limit": 200}  # Client-specific parameters
   )

.. _3-run-the-evaluation:

3. Run the Evaluation
^^^^^^^^^^^^^^^^^^^^^

Execute the evaluation using run_accuracy_test:

.. code:: python

   # Run the test with a named scenario
   results_collection = run_accuracy_test(
       server_config=server_config,
       named_scenarios={"mytest": scenario}
   )

   # Display results
   print(results_collection)

This code will execute the evaluation on the specified datasets and
return detailed performance metrics. The results include accuracy scores
and other relevant metrics for each dataset.

