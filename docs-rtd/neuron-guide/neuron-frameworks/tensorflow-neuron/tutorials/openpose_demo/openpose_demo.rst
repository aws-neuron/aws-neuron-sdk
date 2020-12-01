.. _tensorflow-openpose:

Running OpenPose on Inferentia
==============================

Acknowledgement
---------------

Many thanks to https://github.com/ildoonet/tf-pose-estimation for
providing pretrained model as well as the image preprocessing/pose
estimating infrastructure.

Steps
-----

1. (Optional) Launch compilation instance. We recommend *z1d.xlarge*
   (highest single-core performance) + *Deep Learning AMI (Ubuntu 18.04)
   Version 29.0* (``ami-043f9aeaf108ebc37`` in N. Virginia region).

   1. Since neuron-cc is a cross compiler, for the compilation step you
      may choose to install neuron-cc and tensorflow-neuron packages on
      your existing EC2 instance or local Linux machine.

2. Activate ``aws_neuron_tensorflow_p36`` conda environment (or your own
   virtual/conda environment) and upgrade AWS Neuron packages to latest.

   1. ``source activate aws_neuron_tensorflow_p36``
   2. ``conda update tensorflow-neuron``
   3. ``conda update numpy``

3. Download tensorflow pose net frozen graph.

   1. ``wget -c --tries=2 $( wget -q -O - http://www.mediafire.com/file/qlzzr20mpocnpa3/graph_opt.pb | grep -o 'http*://download[^"]*' | tail -n 1 ) -O graph_opt.pb``

4. Compile the pose net frozen graph into AWS Neuron compatible form.
   Network input image resolution is adjustable with argument
   ``--net_resolution`` (e. g., ``--net_resolution=656x368``). The
   compiled model can accept arbitrary batch size input at runtime.

   1. ``python convert_graph_opt.py graph_opt.pb graph_opt_neuron_656x368.pb``

5. Launch a deployment inf1 instance and copy the AWS Neuron optimized
   tensorflow frozen graph ``graph_opt_neuron_656x368.pb`` to the
   deployment inf1 instance. The smallest instance type inf1.xlarge is
   sufficient for this demo. If you are not on Deep Learning AMI, please
   remember to install runtime dependencies by
   ``pip install -r requirements-infer.txt --extra-index-url=https://pip.repos.neuron.amazonaws.com``
   and ``sudo apt install aws-neuron-runtime`` (detail in
   https://github.com/aws/aws-neuron-sdk/blob/master/docs/neuron-runtime/nrt_start.md).
6. Measure performance on the compiled frozen graph using dummy inputs.

   1. ``python performance_pb.py graph_opt_neuron_656x368.pb --net_resolution 656x368``

Your ``graph_opt_neuron_656x368.pb`` can now be plugged into
https://github.com/ildoonet/tf-pose-estimation seemlessly if you have
tensorflow-neuron installed. When it is used at runtime, please ensure
that the image resolution is the same as compile-time image resolution,
i. e., ``656x368``.
