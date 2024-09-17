.. _ptl_developer_guide:

Developer guide for Neuron-PT-Lightning 
=================================================================

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
