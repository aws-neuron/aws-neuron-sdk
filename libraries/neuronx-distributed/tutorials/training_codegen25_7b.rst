.. _codegen25_7b_tp_zero1_tutorial:

Training CodeGen2.5 7B with Tensor Parallelism and ZeRO-1 Optimizer (``neuronx-distributed``)
==============================================================================================

In this tutorial, we showcase how to pretrain a CodeGen2.5 7B model for program synthesis. Since Codegen2.5's architecture is identical to the one of Llama2, you may want to take a look at our `Llama2 tutorial <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/tutorials/training_llama2_7b.html>`__ first.

After setting up the environment and installing ``neuronx-distributed``, we need to download a data set containing source code (in this case Java code) and then preprocess and tokenize it to match the code-infill format (more about this below). Use the following commands to download the required files. Note, that we reuse our llama2 training files.

.. code:: bash

   mkdir -p ~/examples/tp_zero1_codegen25_7b_hf_pretrain
   cd ~/examples/tp_zero1_codegen25_7b_hf_pretrain
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/llama2/modeling_llama_nxd.py
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/llama2/tp_zero1_llama2_7b_hf_pretrain/tp_zero1_llama2_7b_hf_pretrain.py
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/llama2/tp_zero1_llama2_7b_hf_pretrain/logger.py
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/codegen25/tp_zero1_codegen25_7b_hf_pretrain.sh
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/codegen25/get_dataset_infill.py
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/codegen25/get_dataset_infill.sh
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/codegen25/requirements.txt
   chmod +x tp_zero1_codegen25_7b_hf_pretrain.sh
   chmod +x get_dataset_infill.sh
   python3 -m pip install -r requirements.txt

Data Preprocessing and Tokenization
------------------------------------

To tokenize the data, we will use the CodeGen2.5 tokenizer from the HuggingFace repository. Download it by cloning the repository.

.. code:: bash

   cd ~/examples
   git clone https://huggingface.co/Salesforce/codegen25-7b-mono
   cd codegen25-7b-mono
   rm config.json # Need to use our config.json for some Trainium-specific settings
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/codegen25/config.json
   cd ..

This tutorial makes use of a clean JAVA subset of the TheStack corpus and we preprocess it to fit the infill-format.
The infill format samples a random number of spans and formats the input the following way:

.. code:: Python

   def count_words(filename: str) -> Dict[str, int]:
      """Count the number of occurrences of each word in the file."""
      with open(filename, 'r') as f:
         word_counts = {}
         for line in f:
            if word in word_counts:
                  for word in line.split():
                     word_counts[word] += 1
            else:
                  word_counts[word] = 1
      return word_counts

becomes 

.. code:: Python

   def count_words(filename: str) -> Dict[str, int]:
      """Count the number of occurrences of each word in the file."""
      with open(filename, 'r') as f:
            <mask_1> in word_counts:
                  for word in line.split():
                        word_counts[word] += 1
               else:
                  word_counts[word] = 1
      return word_counts<|endoftext|><sep>
      <mask_1>word_counts = {}
            for line in f:
                  if word <eom>

For each span, we introduce two ``<mask_X>`` tokens. One signals the model that a span is missing at this position, and one (at the end of the code) which is followed by the original code span. Lastly, each span is suffixed with an end of mask (``<eom>``) token. 
You can preprocess and tokenize the dataset by running:

.. code:: bash

   cd ~/examples/tp_zero1_codegen25_7b_hf_pretrain
   ./get_dataset_infill.sh

This will preprocess and store the data in your home directory at ``~/example_datasets/bigcode-stack-java_tokenized_infill``.

Starting Training
-----------------
At this point, you are all set to start training.

Per default, we use a tensor parallel degree of 8, a global batch size of 256, and train for 10k steps. Feel free to change these settings in the ``tp_zero1_codegen25_7b_hf_pretrain.sh`` script.

We first pre-compile the graphs using the ``neuron_parallel_compile``. Let’s run the command below:

.. code:: Python

   sbatch --exclusive \
   --nodes 1 \
   --wrap="srun neuron_parallel_compile bash $(pwd)/tp_zero1_codegen25_7b_hf_pretrain.sh"

Once the graphs are compiled we can run training and observe our loss going down. 
To do so, we run the same command omitting ``neuron_parallel_compile``.

.. code:: Python

   sbatch --exclusive \
   --nodes 1 \
   --wrap="srun bash $(pwd)/tp_zero1_codegen25_7b_hf_pretrain.sh"


Happy training!
