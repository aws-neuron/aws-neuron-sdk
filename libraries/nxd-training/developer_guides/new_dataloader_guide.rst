.. _nxdt_developer_guide_integrate_new_dataloader:

Integrating a new dataset/dataloader
====================================

In this section, we showcase how to integrate a new dataset/dataloader with the library.

.. contents:: Table of contents
   :local:
   :depth: 2

Building Dataset module
-----------------------

One can use the guide on `PyTorch docs <https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class>`_
to create a ``Dataset`` class.

Building DataModule
-------------------

To configure the dataloader, one needs to create a ``DataModule`` class. Neuronx Distributed Training library provides
a ``BaseDataModule`` which one can use to implement their new ``DataModule``. Create a new file called
``new_data_module.py`` and add the following content.

.. code-block:: python

    from neuronx_distributed_training.lightning_modules.data.base import BaseDataModule

    class NewDataModule(BaseDataModule):
        def __init__(self, cfg, trainer):
            """
            DataModule class for configuring the dataset/dataloader

            Args:
                cfg: `data` cfg in the yaml file.
                trainer: PyTorch-Lightning trainer.
            """
            super().__init__(cfg, trainer)
            # Users can use the cfg argument to pass down
            # arguments from the yaml file to the DataModule.


        def get_batch_length(self, batch):
            """
            Returns the length of the batch.
            """
            return len(batch["input_ids"])

        def process_global_batch(self, global_batch, global_batch_size=None):
            """ Any custom processing of batches can be done here.

            Args:
                global_batch: list of inputs, eg.[tokens, labels]
                global_batch_size: Length of tokens and labels
            """
            return global_batch

        def train_dataloader(self):
            """
            This API should return a torch.utils.data.dataloader.DataLoader object
            """
            ...

        def val_dataloader(self):
            """
            This API should return a torch.utils.data.dataloader.DataLoader object
            """
            ...

        def test_dataloader(self):
            """
            This API should return a torch.utils.data.dataloader.DataLoader object
            """
            ...


Plug into ``training.py``
#########################

Once the new data module is created, we can then plug this into the ``training.py`` script under ``examples``
folder. We can modify the ``training.py`` script as follows:

.. code-block:: python

    ...
    # Assuming we are using the same ModelModule we used for LLama example.
    from new_data_module import NewDataModule
    data_module = NewDataModule(cfg, trainer)
    model = HFLLamaModule(cfg, trainer)

    trainer.fit(model, datamodule=data_module)


The rest of the code can remain the same. The trainer will now use the ``NewDataModule`` for fetching the
``dataloader`` and run e2e training.

Create config file
###################

Next, we can create a config file under ``conf`` to be used for this new dataloader. We can start with a copy of
``hf_llama_7B_config.yaml``. Let's call this config file ``my_new_config.yaml``. We can edit the ``data`` key
to configure the ``DataModule``

.. note::

    For the model, we are using the same model that the llama example is using. To configure
    a new model, please check the
    :ref:`nxdt_developer_guide_integrate_new_model` section.

Launching e2e training
######################

We can now launch training using the new ``data_module``. This can be done using the following command:

.. code-block:: shell

    CONF=my_new_config.yaml ./train.sh
