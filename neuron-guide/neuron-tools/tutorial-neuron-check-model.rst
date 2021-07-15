.. _neuron_check_model:

Neuron Check Model
^^^^^^^^^^^^^^^^^^

Overview
========

Neuron Check Model tool provides user with basic information about the compiled and uncompiled model's operations
without the use of TensorBoard-Neuron. For additional visibility into the models, please see :ref:`tensorboard-neuron`.

Neuron Check Model tool scans the user's uncompiled model and provides a table of the operations within the uncompiled
model. By default, the table shows each operation type and number of instances of that type within model, and whether
the type is supported in Neuron. If --show_names option is specified, the table shows each operation by name and
whether the type of that operation is supported in Neuron.

If the model is already compiled, the tool also provides the table of operations as for uncompiled model. The table
include the Neuron subgraph type and number of instances of that type, along with operations that have not been
compiled to Neuron. Additionally, the tool displays a message showing the minimum number of NeuronCores required to run the
model, followed by another table which shows the list of Neuron subgraphs by name and the number of pipelined
NeuronCores used by each subgraph. More information about NeuronCore pipeline can be found in
:ref:`neuroncore-pipeline`. If --expand_subgraph option is specified, the operations within each subgraph are
printed below the subgraph information.

Neuron Check Model tool is currently available for TensorFlow and MXNet. To check PT model, please use
torch.neuron.analyze_model function as shown in PyTorch-Neuron Getting Started tutorial :ref:`/src/examples/pytorch/resnet50.ipynb`

TensorFlow-Neuron Check Model
=============================

The following example shows how to run TensorFlow-Neuron Check Model tool with TensorFlow ResNet50 tutorial.

1. Start with the TensorFlow ResNet50 tutorial at :ref:`/src/examples/tensorflow/tensorflow_resnet50/resnet50.ipynb and do the first three steps of the
tutorial. Please stay in the Python environment that you setup during the tutorial.

2. Install needed tensorflow_hub package and download the tool:

::

    pip install tensorflow_hub
    wget https://raw.githubusercontent.com/aws/aws-neuron-sdk/master/src/neuron-gatherinfo/tf_neuron_check_model.py
    python tf_neuron_check_model.py -h

::

    usage: tf_neuron_check_model.py [-h] [--show_names] [--expand_subgraph]
                                    model_path

    positional arguments:
      model_path         a TensorFlow SavedModel directory (currently supporting
                         TensorFlow v1 SaveModel only).

    optional arguments:
      -h, --help         show this help message and exit
      --show_names       list operation by name instead of summarizing by type
                         (caution: this option will generate many lines of output
                         for a large model).
      --expand_subgraph  show subgraph operations.

3. After step 3 of the TensorFlow ResNet50 tutorial, you can check the uncompiled model to see Neuron supported operations (currently supporting TensorFlow v1 SaveModel only):

::

    $ python tf_neuron_check_model.py ws_resnet50/resnet50/

    * The following table shows the supported and unsupported operations within this uncompiled model.
    * Each line shows an operation type, the number of instances of that type within model,
    * and whether the type is supported in Neuron.
    * Some operation types are excluded from table because they are no-operations or training-related operations:
     ['Placeholder', 'PlaceholderWithDefault', 'NoOp', 'Const', 'Identity', 'IdentityN', 'VarHandleOp',
     'VarIsInitializedOp', 'AssignVariableOp', 'ReadVariableOp', 'StringJoin', 'ShardedFilename', 'SaveV2',
     'MergeV2Checkpoints', 'RestoreV2']

    Op Type           Num Instances   Neuron Supported ?
    -------           -------------   ------------------
    Pad               2               Yes
    RandomUniform     54              Yes
    Sub               54              Yes
    Mul               54              Yes
    Add               54              Yes
    Conv2D            53              Yes
    BiasAdd           54              Yes
    FusedBatchNormV3  53              Yes
    Relu              49              Yes
    MaxPool           1               Yes
    AddV2             16              Yes
    Fill              56              Yes
    Mean              1               Yes
    MatMul            1               Yes
    Softmax           1               Yes
    Pack              1               Yes

    * Total inference operations: 504
    * Total Neuron supported inference operations: 504
    * Percent of total inference operations supported by Neuron: 100.0

4. You can also check the compiled model to see the number of pipeline NeuronCores for each subgraph:

::

    $ python tf_neuron_check_model.py ws_resnet50/resnet50_neuron/

    * Found 1 Neuron subgraph(s) (NeuronOp(s)) in this compiled model.
    * Use this tool on the original uncompiled model to see Neuron supported operations.
    * The following table shows all operations, including Neuron subgraphs.
    * Each line shows an operation type, the number of instances of that type within model,
    * and whether the type is supported in Neuron.
    * Some operation types are excluded from table because they are no-operations or training-related operations:
     ['Placeholder', 'PlaceholderWithDefault', 'NoOp', 'Const', 'Identity', 'IdentityN', 'VarHandleOp',
     'VarIsInitializedOp', 'AssignVariableOp', 'ReadVariableOp', 'StringJoin', 'ShardedFilename', 'SaveV2',
     'MergeV2Checkpoints', 'RestoreV2']

    Op Type   Num Instances   Neuron Supported ?
    -------   -------------   ------------------
    NeuronOp  1               Yes

    * Please run this model on Inf1 instance with at least 1 NeuronCore(s).
    * The following list show each Neuron subgraph with number of pipelined NeuronCores used by subgraph
    * (and subgraph operations if --expand_subgraph is used):

    Subgraph Name                                                                 Num Pipelined NeuronCores
    -------------                                                                 -------------------------
    conv5_block3_3_bn/FusedBatchNormV3/ReadVariableOp/neuron_op_d6f098c01c780733  1

5. When showing subgraph information, you can use --expand_subgraph to show operation types in each subgraph:

::

    $ python tf_neuron_check_model.py ws_resnet50/resnet50_neuron/ --expand_subgraph

    (output truncated to show subgraph information only)

    Subgraph Name                                                                 Num Pipelined NeuronCores
    -------------                                                                 -------------------------
    conv5_block3_3_bn/FusedBatchNormV3/ReadVariableOp/neuron_op_d6f098c01c780733  1
         Op Type         Num Instances
         -------         -------------
         MatMul          1
         Relu            49
         Add             16
         FusedBatchNorm  53
         BiasAdd         54
         Conv2D          53
         Pad             2
         Mean            1
         MaxPool         1
         Softmax         1

6. Use --show_names to see full operation names (caution: this option will generate many lines of output for a large model):

::

    $ python tf_neuron_check_model.py ws_resnet50/resnet50_neuron/ --show_names

    * Found 1 Neuron subgraph(s) (NeuronOp(s)) in this compiled model.
    * Use this tool on the original uncompiled model to see Neuron supported operations.
    * The following table shows all operations, including Neuron subgraphs.
    * Each line shows an operation name and whether the type of that operation is supported in Neuron.
    * Some operation types are excluded from table because they are no-operations or training-related operations:
     ['Placeholder', 'PlaceholderWithDefault', 'NoOp', 'Const', 'Identity', 'IdentityN', 'VarHandleOp',
     'VarIsInitializedOp', 'AssignVariableOp', 'ReadVariableOp', 'StringJoin', 'ShardedFilename', 'SaveV2',
     'MergeV2Checkpoints', 'RestoreV2']

    Op Name                                                                       Op Type   Neuron Supported ?
    -------                                                                       -------   ------------------
    conv5_block3_3_bn/FusedBatchNormV3/ReadVariableOp/neuron_op_d6f098c01c780733  NeuronOp  Yes

    * Please run this model on Inf1 instance with at least 1 NeuronCore(s).
    * The following list show each Neuron subgraph with number of pipelined NeuronCores used by subgraph
    * (and subgraph operations if --expand_subgraph is used):

    Subgraph Name                                                                 Num Pipelined NeuronCores
    -------------                                                                 -------------------------
    conv5_block3_3_bn/FusedBatchNormV3/ReadVariableOp/neuron_op_d6f098c01c780733  1


MXNet-Neuron Check Model
=======================

The following example shows how to run MXNet-Neuron Check Model tool with MXNet ResNet50 tutorial.

1. Start with the MXNet ResNet50 tutorial at :ref:`/src/examples/mxnet/resnet50/resnet50.ipynb` and do the first three steps of the tutorial.
Please stay in the Python environment that you setup during the tutorial.

2. Download the tool:

::

    wget https://raw.githubusercontent.com/aws/aws-neuron-sdk/master/src/neuron-gatherinfo/mx_neuron_check_model.py
    python mx_neuron_check_model.py -h

::

    usage: mx_neuron_check_model.py [-h] [--show_names] [--expand_subgraph]
                                    model_path

    positional arguments:
      model_path         path prefix to MXNet model (the part before -symbol.json)

    optional arguments:
      -h, --help         show this help message and exit
      --show_names       list operation by name instead of summarizing by type
                         (caution: this option will generate many lines of output
                         for a large model).
      --expand_subgraph  show subgraph operations.

3. After step 3 of MXNet ResNet50 tutorial, you can check the uncompiled model to see Neuron supported operations:

::

    $ python mx_neuron_check_model.py resnet-50

    * The following table shows the supported and unsupported operations within this uncompiled model.
    * Each line shows an operation type, the number of instances of that type within model,
    * and whether the type is supported in Neuron.
    * Some operation types are excluded from table because they are no-operations or training-related operations:
     ['null']

    Op Type         Num Instances   Neuron Supported ?
    -------         -------------   ------------------
    BatchNorm       51              Yes
    Convolution     53              Yes
    Activation      50              Yes
    Pooling         2               Yes
    elemwise_add    16              Yes
    Flatten         1               Yes
    FullyConnected  1               Yes
    SoftmaxOutput   1               No

    * Total inference operations: 175
    * Total Neuron supported inference operations: 174
    * Percent of total inference operations supported by Neuron: 99.4

4. You can also check the compiled model to see the number of pipeline NeuronCores for each subgraph:

::

    $ python mx_neuron_check_model.py resnet-50_compiled

    * Found 1 Neuron subgraph(s) (_neuron_subgraph_op(s)) in this compiled model.
    * Use this tool on the original uncompiled model to see Neuron supported operations.
    * The following table shows all operations, including Neuron subgraphs.
    * Each line shows an operation type, the number of instances of that type within model,
    * and whether the type is supported in Neuron.
    * Some operation types are excluded from table because they are no-operations or training-related operations:
     ['null']

    Op Type              Num Instances   Neuron Supported ?
    -------              -------------   ------------------
    _neuron_subgraph_op  1               Yes
    SoftmaxOutput        1               No

    * Please run this model on Inf1 instance with at least 1 NeuronCore(s).
    * The following list show each Neuron subgraph with number of pipelined NeuronCores used by subgraph
    * (and subgraph operations if --expand_subgraph is used):

    Subgraph Name         Num Pipelined NeuronCores
    -------------         -------------------------
    _neuron_subgraph_op0  1

5. When showing subgraph information, you can use --expand_subgraph to show operation types in each subgraph:

::

    $ python mx_neuron_check_model.py resnet-50_compiled --expand_subgraph

    (output truncated to show subgraph information only)

    Subgraph Name         Num Pipelined NeuronCores
    -------------         -------------------------
    _neuron_subgraph_op0  1
         Op Type         Num Instances
         -------         -------------
         BatchNorm       51
         Convolution     53
         Activation      50
         Pooling         2
         elemwise_add    16
         Flatten         1
         FullyConnected  1

6. Use --show_names to see full operation names (caution: this option will generate many lines of output for a large model):

::

    $ python mx_neuron_check_model.py resnet-50_compiled --show_names

    * Found 1 Neuron subgraph(s) (_neuron_subgraph_op(s)) in this compiled model.
    * Use this tool on the original uncompiled model to see Neuron supported operations.
    * The following table shows all operations, including Neuron subgraphs.
    * Each line shows an operation name and whether the type of that operation is supported in Neuron.
    * Some operation types are excluded from table because they are no-operations or training-related operations:
     ['null']

    Op Name               Op Type              Neuron Supported ?
    -------               -------              ------------------
    _neuron_subgraph_op0  _neuron_subgraph_op  Yes
    softmax               SoftmaxOutput        No

    * Please run this model on Inf1 instance with at least 1 NeuronCore(s).
    * The following list show each Neuron subgraph with number of pipelined NeuronCores used by subgraph
    * (and subgraph operations if --expand_subgraph is used):

    Subgraph Name         Num Pipelined NeuronCores
    -------------         -------------------------
    _neuron_subgraph_op0  1
