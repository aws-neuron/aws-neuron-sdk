.. _torch_neuron_lstm_support:

Developer Guide - PyTorch Neuron (``torch-neuron``) |LSTM| Support
==================================================================

The `torch-neuron` package can support |LSTM| operations and yield
high performance on both fixed-length and variable-length sequences. Most
network configurations can be supported, with the exception of those that
require |PackedSequence| usage outside of |LSTM| or |pad_packed_sequence|
operations. Neuron must guarantee that the shapes can remain fixed throughout
the network.

The following sections describe which scenarios can and cannot be supported.

Supported Usage
---------------

Fixed-Length Sequences
~~~~~~~~~~~~~~~~~~~~~~

In normal usage of an |LSTM|, the inputs and outputs are expected to be a fixed
size sequence length. This is the most basic usage of an |LSTM| but may not be
applicable to applications where the input sequence length may vary.

.. code-block:: python

    import torch
    import torch_neuron

    class Network(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.lstm = torch.nn.LSTM(input_size=3, hidden_size=7)

        def forward(self, inputs):
            output, (ht, ct) = self.lstm(inputs)
            return output, (ht, ct)

    # Example Inputs
    seq_len, batch_size, input_size = 5, 2, 3
    inputs = torch.rand(seq_len, batch_size, input_size)

    # Trace
    torch_neuron.trace(Network(), (inputs,))


Packed Input, Padded Output, *Pre-Sorted* Inputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A common usage of an |LSTM| is when the input sequence sizes vary according
to an input sequence lengths (such as tokens).

For example, the following sentences could result in two different
sequence lengths after tokenization:

.. code-block:: python

    # Input
    text = [
       'Hello, sailor',
       'Example',
    ]

    # ... Tokenization ...

    # Result
    tokens = [
        [101, 7592, 1010, 11803, 102],
        [101, 2742,  102,     0,   0],
    ]
    lengths = [5, 3]

Because the lengths are different, the final |LSTM| state will be dependent upon
the lengths of each sequence in the batch. Torch provides a way to deal with
these types of sequences by densely packing batches into a |PackedSequence|. The
most common way this is constructed is by using the |pack_padded_sequence|
utility function prior to feeding inputs into the |LSTM|.

Packing the above sequences would result in the following data and batch
size tensors.

.. code-block:: python

    data = [101, 101, 7592, 2742, 1010, 102, 11803, 102]
    batch_sizes = [2, 2, 2, 1, 1]


In addition to correctly computing final |LSTM| state, using a packed
sequence instead of a padded sequence also improves model performance on CPU.
On Neuron, where computation is fixed to the maximum length ahead of time,
**this is does not improve performance**.

When an |LSTM| is processing a |PackedSequence|, it must do so in a descending
sorted length order. To ensure that sequences are sorted, |pack_padded_sequence|
provides an ``enforce_sorted`` flag. When ``enforce_sorted`` is ``True``, the
input is *already expected* to contain sequences sorted by length in a
decreasing order along the batch dimension. Note that this must be enforced in
the application-level code but is only relevant when batch size > 1.

The following network can compile successfully because the input and output
to the network are guaranteed to be a fixed shape. The input shape is expected
to be a padded tensor and the output tensor is expected to be padded to the
maximum sequence length using the |pad_packed_sequence| function call:

.. code-block:: python
    :emphasize-lines: 14

    import torch
    import torch_neuron

    class Network(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.lstm = torch.nn.LSTM(input_size=3, hidden_size=7)

        def forward(self, inputs, lengths):
            packed_input = torch.nn.utils.rnn.pack_padded_sequence(
                inputs,
                lengths=lengths,
                enforce_sorted=True,
            )
            packed_result, (ht, ct) = self.lstm(packed_input)
            padded_result, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_result)
            return padded_result, ht, ct

    # Example Inputs
    seq_len, batch_size, input_size = 5, 2, 3
    inputs = torch.rand(seq_len, batch_size, input_size)
    lengths = torch.tensor([seq_len] * batch_size)

    # Trace
    torch_neuron.trace(Network(), (inputs, lengths))


Packed Input, Padded Output, *Unsorted* Inputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When ``enforce_sorted`` is ``False``, the input will be sorted unconditionally.
This causes some CPU overhead on Neuron because unsupported operators will be
inserted into the graph such as ``aten::sort`` and ``aten::scatter_``. The
``aten::lstm`` operation can still be supported, but it will be less efficient
than when ``enforce_sorted`` is ``True``.

The following code is able to be traced, but results in the sorting
operations running on CPU. This is not problematic in this case because the
``aten::sort`` and ``aten::scatter_`` are executed on CPU at the very beginning
of the graph just prior to Neuron execution.

Like the previous example, the call to |pad_packed_sequence| ensures that the
output is a fixed-shape based on the maximum sequence length.

.. code-block:: python
    :emphasize-lines: 14

    import torch
    import torch_neuron

    class Network(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.lstm = torch.nn.LSTM(input_size=3, hidden_size=7)

        def forward(self, inputs, lengths):
            packed_input = torch.nn.utils.rnn.pack_padded_sequence(
                inputs,
                lengths=lengths,
                enforce_sorted=False,
            )
            packed_result, (ht, ct) = self.lstm(packed_input)
            padded_result, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_result)
            return padded_result, ht, ct

    # Example Inputs
    seq_len, batch_size, input_size = 5, 2, 3
    inputs = torch.rand(seq_len, batch_size, input_size)
    lengths = torch.tensor([seq_len] * batch_size)

    # Trace
    trace = torch_neuron.trace(Network(), (inputs, lengths))


Packed Inputs, Final Hidden & Cell State Only
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When **only** the final |LSTM| hidden & cell state is used, it does not
matter if the inputs are packed or unpacked since these state
tensors will not vary in size.

.. code-block:: python
    :emphasize-lines: 16,17

    import torch
    import torch_neuron

    class Network(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.lstm = torch.nn.LSTM(input_size=3, hidden_size=7)

        def forward(self, inputs, lengths):
            packed_input = torch.nn.utils.rnn.pack_padded_sequence(
                inputs,
                lengths=lengths,
                enforce_sorted=True,
            )
            packed_output, (ht, ct) = self.lstm(packed_input)
            return ht, ct

    # Example Inputs
    seq_len, batch_size, input_size = 5, 2, 3
    inputs = torch.rand(seq_len, batch_size, input_size)
    lengths = torch.tensor([seq_len] * batch_size)

    # Trace
    trace = torch_neuron.trace(Network(), (inputs, lengths))

Note that when the ``packed_output`` is unused, it does not need to be passed
to the |pad_packed_sequence| to enable the |LSTM| to be compiled.

Unsupported Usage
-----------------

Neuron does not support the use of a |PackedSequence| outside of the |LSTM|
operation and the |pad_packed_sequence| operation. This is because the shape of
a |PackedSequence| can vary depending on the input data. This is incompatible
with the Neuron restriction that all tensor sizes must be known at compilation
time. When a |PackedSequence| is used only by an |LSTM| or |pad_packed_sequence|
operation, Neuron *can guarantee* the size of the intermediary tensors by
padding on behalf of the application.

This means that If the |PackedSequence| is either used by a different operation
or returned from the network this would result in all of the |LSTM| operations to
be executed on CPU or the network compilation will fail.


|PackedSequence| Returned
~~~~~~~~~~~~~~~~~~~~~~~~~

The following is unsupported because the |PackedSequence| result of the |LSTM|
is returned by the network:

.. code-block:: python
    :emphasize-lines: 14

    class Network(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.lstm = torch.nn.LSTM(input_size=3, hidden_size=7)

        def forward(self, inputs, lengths):
            packed_input = torch.nn.utils.rnn.pack_padded_sequence(
                inputs,
                lengths=lengths,
                enforce_sorted=False,
            )
            packed_result, (ht, ct) = self.lstm(packed_input)
            return packed_result.data, ht, ct


**Behavior**: In this case, compilation fails and the following warning is
generated:

.. code-block:: text

    Operator "aten::lstm" consuming a PackedSequence input can only be supported when its corresponding PackedSequence output is unused or unpacked using "aten::_pad_packed_input". Found usage by "prim::Return"


**Resolution**: To avoid this error, the ``packed_result`` should be padded
prior to being returned from the network by using |pad_packed_sequence|


Invalid |PackedSequence| Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following is unsupported because the |PackedSequence| result of the |LSTM|
is used by a non-LSTM operator:

.. code-block:: python
    :emphasize-lines: 14

    class Network(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.lstm = torch.nn.LSTM(input_size=3, hidden_size=7)

        def forward(self, inputs, lengths):
            packed_input = torch.nn.utils.rnn.pack_padded_sequence(
                inputs,
                lengths=lengths,
                enforce_sorted=False,
            )
            packed_result, (ht, ct) = self.lstm(packed_input)
            return torch.max(packed_result.data)

**Behavior**: In this case, compilation fails and the following warning is
generated:

.. code-block:: text

    Operator "aten::lstm" consuming a PackedSequence input can only be supported when its corresponding PackedSequence output is unused or unpacked using "aten::_pad_packed_input". Found usage by "aten::max"

**Resolution**: To avoid this error, the ``packed_result`` should be padded
prior to being used in the :func:`~torch.max` from the network by
using |pad_packed_sequence|.


.. |LSTM| replace:: :class:`~torch.nn.LSTM`
.. |PackedSequence| replace:: :class:`~torch.nn.utils.rnn.PackedSequence`
.. |pack_padded_sequence| replace:: :func:`~torch.nn.utils.rnn.pack_padded_sequence`
.. |pad_packed_sequence| replace:: :func:`~torch.nn.utils.rnn.pad_packed_sequence`
