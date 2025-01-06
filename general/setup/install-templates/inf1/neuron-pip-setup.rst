Modify Pip repository configurations to point to the Neuron repository:

.. code:: bash

   tee $VIRTUAL_ENV/pip.conf > /dev/null <<EOF
   [global]
   index-url = https://pip.repos.neuron.amazonaws.com
   EOF