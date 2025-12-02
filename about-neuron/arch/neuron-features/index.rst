.. _neuron-features-index:

Neuron Features
===============
Neuron features provide insights into Neuron capabilities that enable high-performance and improve usability of developing and deploying deep learning acceleration on top of Inferentia and Trainium based instances.

.. grid:: 2
      :gutter: 2

      .. grid-item-card:: Collective communication
            :link: collective-communication
            :link-type: doc
            :class-body: sphinx-design-class-title-small

            High-performance communication primitives for distributed training and inference across multiple devices.

      .. grid-item-card:: Custom C++ operators
            :link: custom-c++-operators
            :link-type: doc
            :class-body: sphinx-design-class-title-small

            Framework for implementing custom operators in C++ to extend Neuron's built-in operation support.

      .. grid-item-card:: Data types
            :link: data-types
            :link-type: doc
            :class-body: sphinx-design-class-title-small

            Supported numerical data types including FP32, FP16, BF16, and INT8 for efficient model execution.

      .. grid-item-card:: Logical NeuronCore configuration
            :link: logical-neuroncore-config
            :link-type: doc
            :class-body: sphinx-design-class-title-small

            Configuration options for grouping and managing NeuronCores as logical units for workload distribution.

      .. grid-item-card:: Neuron persistent cache
            :link: neuron-caching
            :link-type: doc
            :class-body: sphinx-design-class-title-small

            Persistent caching system for compiled models to reduce compilation time across sessions.

      .. grid-item-card:: NeuronCore batching
            :link: neuroncore-batching
            :link-type: doc
            :class-body: sphinx-design-class-title-small

            Batching strategies to maximize throughput by processing multiple inputs simultaneously on NeuronCores.

      .. grid-item-card:: NeuronCore pipeline
            :link: neuroncore-pipeline
            :link-type: doc
            :class-body: sphinx-design-class-title-small

            Pipeline execution model that overlaps computation and data movement for improved performance.

      .. grid-item-card:: Rounding modes
            :link: rounding-modes
            :link-type: doc
            :class-body: sphinx-design-class-title-small

            Configurable numerical rounding modes for controlling precision and accuracy in computations. 

.. toctree::
    :maxdepth: 1
    :hidden:

    Collective communication <collective-communication>
    Custom C++ operators <custom-c++-operators>
    Data types <data-types>
    Logical NeuronCore configuration <logical-neuroncore-config>
    Neuron persistent cache <neuron-caching>
    NeuronCore batching <neuroncore-batching>
    NeuronCore pipeline <neuroncore-pipeline>
    Rounding modes <rounding-modes>



    
