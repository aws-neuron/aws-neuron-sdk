.. _running-jupyter-notebook-as-script:

Running Jupyter Notebook as script
==================================

Converting the Jupyter Notebook and running
-------------------------------------------

Go into the aws-neuron-sdk repository directory containing the Jupyter Notebook (.ipynb file),

.. code:: bash

   cd aws-neuron-sdk/src/examples/<framework like pytorch, tensorflow, etc>

The Jupyter Notebook (.ipynb) can be converted to python script using jupyter-nbconvert. For example,

.. code:: bash

  jupyter nbconvert --to script tutorial_pretrained_bert.ipynb

and can be run in the virtual env (if needed),

.. code:: bash

  # if not already in the virtual env,
  source activate <virtual env>
  # Run the converted script
  python <tutorial.py>


