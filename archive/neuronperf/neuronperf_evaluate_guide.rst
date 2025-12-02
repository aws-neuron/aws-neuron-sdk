.. _neuronperf_evaluate_guide:

.. meta::
   :noindex:
   :nofollow:
   :description: This tutorial for the AWS Neuron SDK is currently archived and not maintained. It is provided for reference only.
   :date-modified: 12-02-2025

==========================
NeuronPerf Evaluate Guide
==========================

NeuronPerf has a new API for evaluating model accuracy on Neuron hardware. This API is currently only available for PyTorch.

You can access the API through standard ``benchmark()`` by passing an additional kwarg, ``eval_metrics``.

For example:

.. code:: python

    reports = npf.torch.benchmark(
        model_index_or_path,
        dataset,
        n_models=1,
        workers_per_model=2,
        duration=0,
        eval_metrics=['accuracy', 'precision']
    )


In this example, we fix ``n_models`` and ``n_workers`` because replicating the same model will not impact accuracy. We also set ``duration=0`` to allow benchmarking to run untimed through all dataset examples.

Because this call can be tedious to type, a convenience function is provided:

.. code:: python

    reports = npf.torch.evaluate(model_index_or_path, dataset, metrics=['accuracy', 'precision'])


.. note:

    Please note that ``eval_metrics`` becomes ``metrics`` when using ``evaluate``.

The ``dataset`` can be any iterable object that produces ``tuple(*INPUTS, TARGET)``.

If ``TARGET`` does not appear in the last column for your dataset, you can customize this by passing ``eval_target_col``.

For example:

.. code:: python

    reports = npf.torch.evaluate(model_index_or_path, dataset, metrics='accuracy', eval_target_col=1)


You can list the currently available metrics.

.. code:: python

    >>> npf.list_metrics()                                                                                 │·····
    Name                     Description                                                                   │·····
    Accuracy                 (TP + TN) / (TP + TN + FP + FN)                                               │·····
    TruePositiveRate         TP / (TP + FN)                                                                │·····
    Sensitivity              Alias for TruePositiveRate                                                    │·····
    Recall                   Alias for TruePositiveRate                                                    │·····
    Hit Rate                 Alias for TruePositiveRate                                                    │·····
    TrueNegativeRate         TN / (TN + FP)                                                                │·····
    Specificity              Alias for TrueNegativeRate                                                    │·····
    Selectivity              Alias for TrueNegativeRate                                                    │·····
    PositivePredictiveValue  TP / (TP + FP)                                                                │·····
    Precision                Alias for PositivePredictiveValue                                             │·····
    NegativePredictiveValue  TN / (TN + FN)                                                                │·····
    FalseNegativeRate        FN / (FN + TP)                                                                │·····
    FalsePositiveRate        FP / (FP + TN)                                                                │·····
    FalseDiscoveryRate       FP / (FP + TN)                                                                │·····
    FalseOmissionRate        FP / (FP + TP)                                                                │·····
    PositiveLikelihoodRatio  TPR / FPR                                                                     │·····
    NegativeLikelihoodRatio  FNR / TNR                                                                     │·····
    PrevalenceThreshold      sqrt(FPR) / (sqrt(FPR) + sqrt(TPR))                                           │·····
    ThreatScore              TP / (TP + FN + FP)                                                           │·····
    F1Score                  2TP / (2TP + FN + FP)                                                         │·····
    MeanAbsoluteError        sum(|y - x|) / n                                                              │·····
    MeanSquaredError         sum((y - x)^2) / n


New metrics may appear in the list after importing a submodule. For example, ``import neuronperf.torch`` will register a new ``topk`` metric.

Custom Metrics
--------------

Simple Variants
===============

If you wish to register a metric that is a slight tweak of an existing metric with different ``init`` args, you can use ``register_metric_from_existing()``:

.. code:: python

    npf.register_metric_from_existing("topk", "topk_3", k=3)

This example registers a new metric ``topk_3`` from existing metric ``topk``, passing ``k=3`` as at ``init`` time.


New Metrics
===========

You can register your own metrics using ``register_metric()``.

You metrics must extend ``BaseEvalMetric``:

.. code:: python

    class BaseEvalMetric(ABC):
        """
        Abstract base class BaseEvalMetric from which other metrics inherit.
        """

        @abstractmethod
        def process_record(self, output: Any = None, target: Any = None) -> None:
            """Process an individual record and return the result."""
            pass

        @staticmethod
        def aggregate(metrics: Iterable["BaseEvalMetric"]) -> Any:
            """Combine a sequence of metrics into a single result."""
            raise NotImplementedError

For example:

.. code:: python

    import neuronperf as npf

    class MyCustomMetric(npf.BaseEvalMetric):
        def __init__(self):
            super().__init__()
            self.passing = 0
            self.processed = 0

        def process_record(self, outputs, target):
            self.processed += 1
            if outputs == target:
                self.passing += 1
        
        @staticmethod
        def aggregate(metrics):
            passing = 0
            processed = 0
            for metric in metrics:
                passing += metric.passing
                processed += metric.processed
            return passing / processed if processed else 0


    npf.register_metric("MyCustomMetric", MyCustomMetric)


