
The following commands assumes you are using a Conda environment and
have already activated it. Please see
https://docs.conda.io/projects/conda/en/latest/user-guide/install/ for
installation instruction if Conda is not installed. The following steps
are example steps to install and activate Conda environment:

.. code:: bash

   curl -O https://repo.anaconda.com/miniconda/Miniconda3-4.7.12.1-Linux-x86_64.sh
   echo "bfe34e1fa28d6d75a7ad05fd02fa5472275673d5f5621b77380898dee1be15d2 Miniconda3-4.7.12.1-Linux-x86_64.sh" | sha256sum --check
   bash Miniconda3-4.7.12.1-Linux-x86_64.sh
   source ~/.bashrc
   conda create -q -y -n test_conda_env python=3.6
   source activate test_conda_env

.. code:: bash

   # Add Neuron Conda channel to Conda environment
   conda config --env --add channels https://conda.repos.neuron.amazonaws.com

