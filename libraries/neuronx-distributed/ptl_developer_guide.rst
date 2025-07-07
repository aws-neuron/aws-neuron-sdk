.. _ptl_developer_guide:

Developer guide for Neuron-PT-Lightning (PyTorch Lightning)
=================================================================

Quickstart Guide
^^^^^^^^^^^^^^^^

This quickstart guide provides a simple working example to get you started with Neuron-PT-Lightning (PyTorch Lightning with Neuron). 
The example demonstrates the basic components needed to train a model on Neuron devices using PTL.

Complete Working Example
''''''''''''''''''''''''

Here's a minimal example that trains a simple linear model using Neuron-PT-Lightning:

.. code:: ipython3

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    import neuronx_distributed as nxd
    from neuronx_distributed.lightning import NeuronLTModule
    from neuronx_distributed.lightning import NeuronXLAStrategy
    from pytorch_lightning import Trainer

    import torch_xla.core.xla_model as xm

    # Define your Neuron-compatible LightningModule
    class ToyNeuronModule(NeuronLTModule):
        def __init__(self):
            # Configure neuronx-distributed settings
            self.nxd_config = nxd.neuronx_distributed_config(
                    tensor_parallel_size=1,
                    pipeline_parallel_size=1,
                    expert_parallel_size=1,
                    context_parallel_size=1,
            )
            # Initialize the parent NeuronLTModule
            super().__init__(
                model_fn=nn.Linear,           # Model constructor function
                model_args=(32, 2),           # Arguments for model constructor (input_size, output_size)
                nxd_config=self.nxd_config,   # Neuron distributed configuration
                opt_cls=torch.optim.Adam,     # Optimizer class
                opt_kwargs={"lr": 1e-3},      # Optimizer arguments
                scheduler_cls=None,           # Learning rate scheduler (None for this example)
                scheduler_args=(),            # Scheduler arguments
                grad_accum_steps=1,           # Gradient accumulation steps
                manual_opt=True,              # Use manual optimization (required for Neuron)
            )
            # Initialize training state variables
            self.averaged_loss = torch.tensor(0.0, device="xla")
            self.should_print = False
            self.loss = None
            

        def training_step(self, batch, batch_idx):
            # Mark step to isolate the forward+backward computation graph
            xm.mark_step()
            
            # Forward pass
            x = batch[0]
            out = self.model(x)
            loss = out.sum() / self.grad_accum_steps
            
            # Backward pass
            loss.backward()
            self.averaged_loss += loss.detach()
            
            # Mark step to isolate forward+backward from optimization
            xm.mark_step()

            # Perform optimization step when gradient accumulation is complete
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                optimizer = self.optimizers()
                optimizer.step()
                optimizer.zero_grad()
                xm.mark_step()  # Isolate optimization step
                
                # Store loss for logging
                self.loss = self.averaged_loss.detach()
                self.averaged_loss.zero_()
                
            return loss

        def configure_optimizers(self):
            # Create and configure the optimizer
            opt = self.opt_cls(self.model.parameters(), **self.opt_kwargs)
            opt.zero_grad()
            return [opt]

        def on_train_batch_end(self, *args, **kwargs):
            # Log the loss when available
            if self.loss is not None:
                self.log("loss", self.loss.item(), prog_bar=True)


    # Main training script
    if __name__ == "__main__":
        # Create dummy training data
        dataset = TensorDataset(torch.randn(2048, 32))  # 2048 samples, 32 features each
        dataloader = DataLoader(dataset, batch_size=8)
        
        # Configure neuronx-distributed settings
        nxd_config = nxd.neuronx_distributed_config(
                tensor_parallel_size=1,      # No tensor parallelism for this simple example
                pipeline_parallel_size=1,    # No pipeline parallelism
                expert_parallel_size=1,      # No expert parallelism (for MoE models)
                context_parallel_size=1,     # No context parallelism
        )
        
        # Create the Neuron XLA strategy
        strategy = NeuronXLAStrategy(
            nxd_config = nxd_config
        )
        
        # Initialize the model
        model = ToyNeuronModule()
        
        # Create the trainer with Neuron strategy
        trainer = Trainer(
            strategy=strategy,        # Use Neuron XLA strategy
            max_steps=20,            # Train for 20 steps
            log_every_n_steps=1,     # Log every step
        )
        
        # Start training
        trainer.fit(model=model, train_dataloaders=dataloader)

How this example works
''''''''''''''''''''''''

**1. NeuronLTModule**: The core component that extends PyTorch Lightning's LightningModule with Neuron-specific functionality.

- Inherits from ``NeuronLTModule`` instead of standard ``LightningModule``
- Requires manual optimization (``manual_opt=True``)
- Uses ``xm.mark_step()`` to isolate computation graphs for optimal performance

**2. Neuronx-Distributed Configuration**: Defines parallelism settings for distributed training.

- ``tensor_parallel_size``: Number of devices for tensor parallelism
- ``pipeline_parallel_size``: Number of devices for pipeline parallelism  
- ``expert_parallel_size``: Number of devices for expert parallelism (MoE models)
- ``context_parallel_size``: Number of devices for context parallelism

**3. NeuronXLAStrategy**: The training strategy that handles Neuron device management and XLA compilation.

**4. Manual Optimization**: Unlike standard PyTorch Lightning, Neuron-PT-Lightning requires manual control over the optimization process:

- Call ``loss.backward()`` manually
- Get optimizer with ``self.optimizers()``
- Call ``optimizer.step()`` and ``optimizer.zero_grad()`` manually
- Use ``xm.mark_step()`` to separate computation graphs

**5. XLA Mark Steps**: Critical for performance on Neuron devices:

- ``xm.mark_step()`` before forward pass isolates the computation
- ``xm.mark_step()`` after backward pass separates forward/backward from optimization
- ``xm.mark_step()`` after optimization isolates the optimization step

This example provides a foundation that you can extend with more complex models, data loading, and training configurations as shown in the detailed sections below.

Training
^^^^^^^^

For training models with Neuron-PT-Lightning, user needs to make few
changes to their model/training script. 
In this document we explain how we can train a model using Tensor Parallelism (TP), Data Parallelism (DP) and Zero-1. 

First, let's start with the model changes. Please follow the guidelines here (`tensor parallel guidance <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/tp_developer_guide.html>`__) 
for building the model with tensor-parallelism enabled and setting up training dataset.

Next, let's walkthrough how we can build the training loop with Neuron-PT-Lightning APIs

Configure NeuronLTModule
''''''''''''''''''''''''
NeuronxDistributed overrides `LightningModule <https://lightning.ai/docs/pytorch/stable/common/lightning_module.html>`__ with built-in support for 
Neuron device. User needs to inherit from ``NeuronLTModule``

.. code:: ipython3

    class NeuronLlamaLTModule(NeuronLTModule):
        def training_step(self, batch, batch_idx):
            ...
        ...

Within LTModule, user needs to override the following methods
``training_step``
At this moment NeuronLTModule only support `manual optimization <https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html>`__, so user needs to define forward, backward and optimization steps

.. code:: ipython3

    def training_step(self, batch, batch_idx):
        xm.mark_step() # Isolate forward+backward graph
        for logger in self.trainer.loggers:
            logger.print_step = -1
        self.should_print = False
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss / self.grad_accum_steps
        loss.backward()
        self.averaged_loss += loss.detach()
        xm.mark_step() # Isolate forward+backward graph
        if not self.automatic_optimization and (batch_idx +1) % self.grad_accum_steps == 0:
            self.should_print = True
            loss_div = self.averaged_loss / self.trainer.strategy.data_parallel_size
            loss_reduced = xm.all_reduce(
                xm.REDUCE_SUM,
                loss_div,
                groups=parallel_state.get_data_parallel_group(as_list=True),
            )
            loss_reduced_detached = loss_reduced.detach()
            self.averaged_loss.zero_()
            optimizer = self.optimizers()
            scheduler = self.lr_schedulers()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            xm.mark_step() # Isolate Optimization step graph

            # Setup items for logging
            self.loss = loss_reduced_detached
        return loss

``configure_optimizers``
Configure optimizer and lr_scheduler

.. code:: ipython3

    def configure_optimizers(self):
        param_groups = self.get_param_groups_by_weight_decay()
        optimizer = initialize_parallel_optimizer(
            self.nxd_config, self.opt_cls, param_groups, **self.opt_kwargs
        )
        optimizer.zero_grad()
        scheduler = self.scheduler_cls(optimizer, *self.scheduler_args, **self.scheduler_kwargs)
        return (
            [optimizer],
            [
                {
                    "scheduler": scheduler,
                }
            ],
        )

``on_train_batch_end``
Customized behaviour at the end of each training batch, like logging

.. code:: ipython3

    def on_train_batch_end(self, *args, **kwargs):
        if self.should_print:
            if not self.automatic_optimization:
                self.log(
                    "loss",
                    self.loss.detach().cpu().item() if self.loss is not None else torch.zeros(1, device="cpu", requires_grad=False),
                    prog_bar=True,
                )
                self.log(
                    "global_step",
                    self.global_step,
                    prog_bar=True,
                    on_step=True,
                    on_epoch=True,
                )
                for logger in self.trainer.loggers:
                    logger.print_step = self.global_step

Note that NeuronLTModule has a built-in function of ``get_param_groups_by_weight_decay`` for common use case as shown in snippet below, 
users can also override with their own param_groups generation.

.. code:: ipython3

    def get_param_groups_by_weight_decay(self):
        """Get param groups. Customers can override this to have their own way of weight_decay"""
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm"]  # gamma/beta are in LayerNorm.weight

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return optimizer_grouped_parameters


Configure DataModule
''''''''''''''''''''

Create a LightningDataModule for data loading/sampling

.. code:: ipython3

    class NeuronLightningDataModule(LightningDataModule):
        def __init__(
            self, 
            dataloader_fn: Callable,
            data_dir: str, 
            batch_size: int,
            data_args: Tuple = (), 
            data_kwargs: Dict = {},
        ):
            super().__init__()
            self.dataloader_fn = dataloader_fn
            self.data_dir = data_dir
            self.batch_size = batch_size
            self.data_args = data_args,
            self.data_kwargs = data_kwargs
            

        def setup(self, stage: str):
            pass

        def train_dataloader(self):
            return self.dataloader_fn(
                self.data_dir,
                self.batch_size,
                self.trainer.strategy.data_parallel_size,
                self.trainer.strategy.data_parallel_rank,
                *self.data_args,
                **self.data_kwargs
            )

Update Training Script
''''''''''''''''''''''

For detailed introduction to each api/class, check `api guide <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/api_guide.html>`__

Create NeuronLTModule and DataModule
------------------------------------

.. code:: ipython3

    model = NeuronLlamaLTModule(
        model_fn = LlamaForCausalLM,
        nxd_config = nxd_config,
        model_args = (model_config,),
        opt_cls = optimizer_cls,
        scheduler_cls = configure_scheduler,
        opt_kwargs = {
            "lr": flags.lr,
        },
        scheduler_args = (flags.warmup_steps, flags.max_steps),
        grad_accum_steps = flags.grad_accum_usteps,
        manual_opt = True, 
    )

    dm = NeuronLightningDataModule(
        create_llama_pretraining_dataset,
        flags.data_dir,
        flags.batch_size,
        data_args = (flags.seed,),
    )

Add Strategy, Plugins, Callbacks
--------------------------------

.. code:: ipython3

    strategy = NeuronXLAStrategy(
        nxd_config = nxd_config
    )
    plugins = []
    plugins.append(NeuronXLAPrecisionPlugin())
    callbacks = []
    callbacks.append(NeuronTQDMProgressBar())

Create Trainer and Start Training
---------------------------------

.. code:: ipython3

    trainer = Trainer(
        strategy = strategy, 
        max_steps = flags.steps_this_run,
        plugins = plugins,
        enable_checkpointing = flags.save_checkpoint,
        logger = NeuronTensorBoardLogger(save_dir=flags.log_dir),
        log_every_n_steps = 1,
        callbacks = callbacks,
    )
    trainer.fit(model=model, datamodule=dm)

Checkpointing
-------------

To enable checkpoint saving, add `ModelCheckpoint <https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html>`__
to the callbacks

.. code:: ipython3

    callbacks.append(
        ModelCheckpoint(
            save_top_k = flags.num_kept_checkpoint,
            monitor="global_step",
            mode="max",
            every_n_train_steps = flags.checkpoint_freq,
            dirpath = flags.checkpoint_dir,
        )
    )

To load from specific checkpoint, add ``ckpt_path=ckpt_path`` to ``trainer.fit``

.. code:: ipython3

     trainer.fit(model=model, datamodule=dm, ckpt_path=ckpt_path)
