.. _bucketing_app_note:

Running inference on variable input shapes with bucketing
=========================================================

.. contents::
   :local:
   :depth: 2

Introduction
------------

With Inferentia, the shape of every input must be fixed at compile time. For
applications that require multiple input sizes, we recommend using padding or
bucketing techniques. Padding requires you to compile your model with the
largest expected input size and pad every input to this maximum size. If the
performance of your model using padding is not within your targets, you can
consider implementing bucketing.

This guide introduces bucketing, a technique to run inference on inputs with
variable shapes on Inferentia. The following sections explain how bucketing can
improve the performance of inference workloads on Inferentia. It covers an
overview of how bucketing works and provides examples of using bucketing in
:ref:`computer vision <bucketing_example_cv>` and
:ref:`natural language processing<bucketing_example_nlp>` applications.

Applications that benefit from bucketing
----------------------------------------

Bucketing refers to compiling your model multiple times with different target
input shapes to create “bucketed models." :ref:`creating_buckets` provides an
overview on selecting the input shapes that you use to create bucketed models. At
inference time, each input is padded until its shape matches the next largest
bucket shape. The padded input is then passed into the corresponding bucketed model
for inference. By compiling the same model with multiple different input shapes,
the amount of input padding is reduced compared to padding every input to the
maximum size in your dataset. This technique minimizes the compute overhead
and improves inference performance compared to padding every image to the
maximum shape in your dataset.

Bucketing works best when multiple different bucketed models are created to efficiently
cover the full range of input shapes. You can fine-tune the model performance
by experimenting with different bucket sizes that correspond to the
distribution of input shapes in your dataset.

Bucketing can only be used if there is an upper bound on the shape of the
inputs. If necessary, an upper bound on the input shape can be enforced using
resizing and other forms of preprocessing.

.. _num_buckets:

The upper bound on the number of bucketed models that you use is dictated by the
total size of the compiled bucketed models. Each Inferentia chip has 8GB of
DRAM, or 2GB of DRAM per NeuronCore. An inf1.xlarge and inf1.2xlarge have
1 Inferentia chip, an inf1.6xlarge has 4 Inferentia chips, and an inf1.24xlarge
has 16 Inferentia chips. Thus, you should limit the total size of all bucketed
models to around 8GB per Inferentia chip or 2GB per NeuronCore.
The following formula provides an approximation for the number of
compiled bucketed models you can fit on each NeuronCore:

::

    number-of-buckets = round(10^9 / number-of-weights-in-model)

We recommend using :ref:`neuron-top <neuron-top-ug>` to monitor the
memory usage on your inf1 instance as you load multiple bucketed models.

Implementing bucketing
-----------------------

Implementing bucketing consists of two main parts: creating multiple bucketed
models at compile-time and running inference using the bucketed models on (padded)
inputs. The following sections describe how to implement bucketing to run
inference in applications that have variable input shapes.

.. _creating_buckets:

Creating bucketed models
^^^^^^^^^^^^^^^^^^^^^^^^^

Before running inference, models should be compiled for different input shapes
that are representative of the input dataset. The input shapes that are used
to compile the models determine the bucket shapes that are used during inference.
The bucket shapes should be chosen to minimize the amount of padding on each new input.
Additionally, there should always be a bucket that’s large enough to handle the
maximum input shape in the dataset. The limit on the number of compiled bucketed
models that can be used is described in this :ref:`section<num_buckets>`.


Running inference with bucketing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At inference time, each input should be padded to match the size of the next
largest bucket, such that the height and width (or sequence length) of the
padded input equals the size of the bucket. Then, the padded input should
be passed into the corresponding bucket for inference. If necessary, it’s
important to remove and/or crop any aberrant predictions that occur in the
padded region. For example, in object detection applications, bounding box
predictions that occur in the padded regions should be removed to avoid
erroneous predictions. 

.. _bucketing_examples:

Examples
--------

The following sections provide examples of applying the bucketing technique
to run inference in applications that have variable input shapes.

.. _bucketing_example_cv:

Computer vision bucketing
^^^^^^^^^^^^^^^^^^^^^^^^^^

As an example of implementing bucketing for computer vision models, consider an
application where the height and width of images in dataset are uniformly
distributed between `[400, 400]` and `[800, 800]`. Given that every input
shape between `[400, 400]` and `[800, 800]` is equally likely, it could
make sense to create bucketed models that divide up the range of input shapes into
equally sized chunks. For example, we could create bucketed models for the input shapes
`[500, 500]`, `[600, 600]`, `[700, 700]`, and `[800, 800]`. 

As an example of running inference with bucketing, let’s assume that we created
bucketed models for the input shapes `[500, 500]`, `[600, 600]`, `[700, 700]`, and
`[800, 800]`. If we receive an input with shape `[640, 640]`, we would
pad the input to the next largest bucket, `[700, 700]`, and use this bucket
for inference. If we receive an input with shape `[440, 540]`, we would
need to pad the input to the bucket size, `[600, 600]`, and use this bucket
for inference.

As another example of creating bucketed models, consider a computer vision
application where the dataset is not uniformly distributed. As before, let’s
assume the input shapes range between `[400, 400]` to `[800, 800]`. Now, let’s
assume the data shape distribution is bimodal, such that `[540, 540]` and
`[720, 720]` are the two most common input shapes. In this example, it might
make sense to create bucketed models for input shapes `[540, 540]`, `[720, 720]`, and
`[800, 800]` to target the most common shapes while still including the
entire range of input shapes.


End-to-end computer vision bucketing example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this example, we run inference in a computer vision application that has
variable shaped images that range in shape from `[400, 400]` to
`[800, 800]`. We create bucketed models for the input shapes `[500, 500]`,
`[600, 600]`, `[700, 700]`, and `[800, 800]` to handle the variable input
shapes.

.. code-block:: python

    import numpy as np
    import torch
    from torchvision import models
    import torch_neuron

    # Load the model and set it to evaluation mode
    model = models.resnet50(pretrained=True)
    model.eval()

    # Define the bucket sizes that will be used for compilation and inference
    bucket_sizes = [(500, 500), (600, 600), (700, 700), (800, 800)]

    # Create the bucketed models by compiling a model for each bucket size
    buckets = {}
    for bucket_size in bucket_sizes:
        # Create an example input that is the desired bucket size
        h, w = bucket_size
        image = torch.rand([1, 3, h, w])

        # Compile with the example input to create the bucketed model
        model_neuron = torch.neuron.trace(model, image)

        # Run a warm up inference to load the model into Inferentia memory
        model_neuron(image)

        # Add the bucketed model based on its bucket size
        buckets[bucket_size] = model_neuron


    def get_bucket_and_pad_image(image):
        # Determine which bucket size to use
        oh, ow = image.shape[-2:]
        target_bucket = None
        for bucket_size in bucket_sizes:
            # Choose a bucket that's larger in both the height and width dimensions
            if oh <= bucket_size[0] and ow <= bucket_size[1]:
                target_bucket = bucket_size
                break

        # Pad the image to match the size of the bucket
        h_delta = target_bucket[0] - oh
        w_delta = target_bucket[1] - ow

        b_pad = h_delta  # Bottom padding
        l_pad = 0  # Left padding
        t_pad = 0  # Top padding
        r_pad = w_delta  # Right padding

        # Pad the height and width of the image
        padding_amounts = (l_pad, r_pad, t_pad, b_pad)
        image_padded = torch.nn.functional.pad(image, padding_amounts, value=0)

        return image_padded, target_bucket


    # Run inference on inputs with different shapes
    for _ in range(10):
        # Create an image with a random height and width in range [400, 400] to [800, 800]
        h = int(np.random.uniform(low=400, high=800))
        w = int(np.random.uniform(low=400, high=800))
        image = torch.rand(1, 3, h, w)

        # Determine bucket and pad the image
        image_padded, target_bucket = get_bucket_and_pad_image(image)

        # Use the corresponding bucket to run inference
        output = buckets[target_bucket](image_padded)


.. _bucketing_example_nlp:

Natural language processing bucketing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As an example of implementing bucketing for natural language processing models,
consider an application where the lengths of tokenized sequences in a dataset are
uniformly distributed between 0 and 128 tokens. Given that every tokenized sequence
length between 0 and 128 is equally likely, it might make sense to create
bucketed models that divide up the range of tokenized sequence lengths into equally sized
chunks. For example, we could create bucketed models for tokenized sequence lengths 64
and 128.

As an example of running inference with bucketing, let's assume that we created
bucketed models for the input tokenized sequence lengths 64 and 128. If we receive a
tokenized sequence with length 55, we would need to pad it to the bucket size
64 and use this bucket for inference. If we receive a tokenized sequence with
length 112, we would need to pad it to the bucket size 128 and use this bucket
for inference.

End-to-end natural language processing bucketing example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this example, we run inference in a natural language processing application
that has variable length tokenized sequences that range from 0 to 128. We
create bucketed models for lengths 64 and 128 to handle the variable input lengths.

.. code-block:: python

    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch_neuron

    # Build tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc", return_dict=False)
    model.eval()

    # Define the bucket sizes that will be used for compilation and inference
    bucket_sizes = [64, 128]

    # Create the bucketed models by compiling a model for each bucket size
    buckets = {}
    for bucket_size in bucket_sizes:
        # Setup some example inputs
        sequence_0 = "The company HuggingFace is based in New York City"
        sequence_1 = "HuggingFace's headquarters are situated in Manhattan"

        # Create an example input that is the desired bucket size
        paraphrase = tokenizer.encode_plus(sequence_0,
                                        sequence_1,
                                        max_length=bucket_size,
                                        padding='max_length',
                                        truncation=True,
                                        return_tensors="pt")

        # Convert example inputs to a format that is compatible with TorchScript tracing
        example_inputs_paraphrase = paraphrase['input_ids'], paraphrase['attention_mask'], paraphrase['token_type_ids']

        # Compile with the example input to create the bucketed model
        model_neuron = torch.neuron.trace(model, example_inputs_paraphrase)

        # Run a warm up inference to load the model into Inferentia memory
        model_neuron(*example_inputs_paraphrase)

        # Add the bucketed model based on its bucket size
        buckets[bucket_size] = model_neuron


    def get_bucket_and_pad_paraphrase(paraphrase):
        # Determine which bucket size to use
        inputs = paraphrase['input_ids']
        attention = paraphrase['attention_mask']
        token_type = paraphrase['token_type_ids']
        paraphrase_len = inputs.shape[1]
        target_bucket = None
        for bucket_size in bucket_sizes:
            if paraphrase_len <= bucket_size:
                target_bucket = bucket_size
                break

        # Pad the paraphrase to match the size of the bucket
        delta = target_bucket - paraphrase_len
        zeros = torch.zeros([1, delta], dtype=torch.long)
        inputs = torch.cat([inputs, zeros], dim=1)
        attention = torch.cat([attention, zeros], dim=1)
        token_type = torch.cat([token_type, zeros], dim=1)

        paraphrase_padded = inputs, attention, token_type
        return paraphrase_padded, target_bucket


    # Create two sample sequences
    sequence_0 = ("The only other bear similar in size to the polar bear is the "
                  "Kodiak bear, which is a subspecies of the brown bear. Adult male "
                  "polar bears weigh 350–700 kg and measure 2.4–3 meters in total "
                  "length. All bears are short-tailed, the polar bear's tail is "
                  "relatively the shortest amongst living bears.")
    sequence_1 = ("Around the Beaufort Sea, however, mature males reportedly "
                  "average 450 kg. Adult females are roughly half the size of males "
                  "and normally weigh 150–250 kg, measuring 1.8–2.4 meters in length. "
                  "The legs are stocky and the ears and tail are small.")

    # Run inference on inputs with different shapes
    # We create the variable shapes by randomly cropping the sequences
    for _ in range(10):
        # Get random sequence lengths between 0 and 128
        paraphrase_len = int(np.random.uniform(128))

        # Crop the paraphrase
        paraphrase_cropped = tokenizer.encode_plus(sequence_0,
                                        sequence_1,
                                        max_length=paraphrase_len,
                                        padding='max_length',
                                        truncation=True,
                                        return_tensors="pt")

        # Determine bucket and pad the paraphrase
        paraphrase_padded, target_bucket = get_bucket_and_pad_paraphrase(paraphrase_cropped)

        # Use the corresponding bucket to run inference
        output = buckets[target_bucket](*paraphrase_padded)
