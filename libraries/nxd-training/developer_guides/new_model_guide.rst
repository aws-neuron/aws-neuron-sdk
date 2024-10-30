.. _nxdt_developer_guide_integrate_new_model:

Integrating a New Model
==========================

The NeuronX Distributed Training library is a modular framework that allows users to integrate
their new modules with the framework while still utilizing the other modules provided by the
library. In this section, we showcase how to integrate a new model with the library.

.. contents:: Table of contents
   :local:
   :depth: 2

Model Building (torch.nn.Module)
--------------------------------

Users can create a torch.nn.Module using the tensor-parallel APIs provided by the
`NeuronxDistributed <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/index.html>`_
library. Let’s take an example of the
`GPT-NeoX model built inside NxD examples <https://github.com/aws-neuron/neuronx-distributed/blob/main/examples/training/tp_dp_gpt_neox_hf_pretrain/tp_dp_gpt_neox_20b_hf_pretrain/modeling_gpt_neox_nxd.py>`_.
We can copy the model file and treat it as a new model to onboard using the framework.

.. note::

    To understand more about how to build models using Tensor-parallel APIs check the
    `Developer guide here <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/tp_developer_guide.html#creating-model>`_.


Model Integration
-----------------

Once we have built the model, the next step is to integrate with the training framework. This can be done
using the following steps:

.. _nxdt_developer_guide_integrate_new_model_build_module:

Build a `Lightning Module <https://lightning.ai/docs/pytorch/stable/common/lightning_module.html>`_
####################################################################################################

Neuronx Distributed Training framework provides a ``BaseModelModule`` that implements the majority of the training
APIs. Users can subclass this base module and implement few APIs that set up the model. Here is an example to
setup the GPT-NeoX model example. Create a new file called ``new_model_module.py`` and add the following content.

.. code-block:: python

    from transformers import GPTNeoXConfig
    import neuronx_distributed as nxd
    from neuronx_distributed.parallel_layers.layer_norm import LayerNorm
    from neuronx_distributed_training.lightning_modules.model.base import BaseModelModule
    from neuronx_distributed_training.utils.model_utils import get_param_groups_by_weight_decay
    from modeling_gpt_neox_nxd import GPTNeoXForCausalLMNxD

    class MyNewModel(BaseModelModule):

        def _get_model(self,):
            model_name = "EleutherAI/gpt-neox-20b"
            config = GPTNeoXConfig.from_pretrained(model_name)
            config.use_cache = False
            # Note: We can modify the model by reading parameters from self.config.model.
            # We would have to expose those config in the self.config.model accordingly.
            # Couple of examples are here, where we have exposed num_layers and hidden_size.
            if self.config.model.get('num_layers', -1) != -1:
                config.num_hidden_layers = self.config.model.get('num_layers')
            if self.config.model.get('hidden_size', -1) != -1:
                config.hidden_size = self.config.model.get('hidden_size')
            # This is because the GPT-Neox implementation requires this in the config.
            config.sequence_parallel_enabled = self.config.distributed_strategy.get("sequence_parallel", False)
            return GPTNeoXForCausalLMNxD(config)

        def build_model(self):
            # This API is where we build the model object, and return the model.
            # However, in addition to returning the model, users need to
            # configure the nxd config too for pipeline parallelism and
            # activation checkpointing. Here is an example:
            if self.config.model.get("activations_checkpoint_granularity", None) == "selective":
                # Here just to showcase how to recompute modules, we are using
                # GPTNeoXMLPNxD, users can add their own custom modules
                self.nxd_config["activation_checkpoint_config"] = GPTNeoXMLPNxD
            elif self.config.model.get("activations_checkpoint_granularity", None) == "full":
                self.nxd_config["activation_checkpoint_config"] = "full"

            # Read more about configuring pipeline parallel config here:
            # https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/pp_developer_guide.html#pp-developer-guide
            self.nxd_config["pipeline_config"].update(
                {
                    "transformer_layer_cls": GPTNeoXLayerNxD,
                    "output_loss_value_spec": (True, False),
                    "input_names": ["input_ids", "attention_mask", "labels"],
                    "leaf_module_cls": [LayerNorm.__name__],
                }
            )
            return nxd.initialize_parallel_model(self.nxd_config, self._get_model)

        def setup_optimizer_param_groups(self):
            # Depending on what weight decay we need, users can configure
            # the params groups accordingly.
            no_decay = ["bias"]
            if self.config.model.get("do_layer_norm_weight_decay", False):
                no_decay.append("LayerNorm")
            self._optimizer_param_groups = get_param_groups_by_weight_decay(self.model, no_decay)

        def init_weights(self,):
            """
            This API is mainly to tell the framework how each layer needs
            to be initialized. This is required because NxD's PP API would
            use this to initialize the layers after model partition.
            Any layer that is unique to the model needs to be added here.
            """
            if isinstance(module, LayerNorm):
                module.weight.data.fill_(1.0)
            # The BaseModelModule already initializes the ColumnParallel, RowParallel
            # ParallelEmbedding layers.
            super().init_weights()


Plug into ``training.py``
#########################


Once the new model is created, we can then plug this into the ``training.py`` script under ``examples`` folder.
We can modify the ``training.py`` script as follows:

.. code-block:: python

    ...
    # Assuming we are using the same DataModule we used for LLama example.
    data_module = HFDataModule(cfg, trainer)
    from new_model_module import MyNewModel
    model = MyNewModel(cfg, trainer)

    trainer.fit(model, datamodule=data_module)

The rest of the code can remain the same. The trainer will now use the ``MyNewModel`` for fetching the
``model`` code and run e2e training.

Create config file
###################

Next we can create a config file under ``conf`` to be used for this new model. We can start with a copy of
``hf_llama_7B_config.yaml``. Let's call this config file ``my_new_config.yaml``. We can remove the key
``model.model_config`` as we are not using it inside our ``MyNewModel``. We can edit the
``distributed_strategy`` config depending on what we need.

.. note::

    For the dataset, we are using the same dataset that the llama example is using. To configure
    a new dataset, please check the
    :ref:`nxdt_developer_guide_integrate_new_dataloader` section

Launching e2e training
######################

We can now launch training using the new model. This can be done using the following command:

.. code-block:: shell

    CONF=my_new_config.yaml ./train.sh
