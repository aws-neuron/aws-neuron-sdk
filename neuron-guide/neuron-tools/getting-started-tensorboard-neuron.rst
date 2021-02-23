.. _tensorboard-neuron:

Getting Started: TensorBoard-Neuron
===================================

This guide is for developers who want to better understand how their
model runs on Neuron Cores.

TensorBoard-Neuron is adapted to provide useful information related to
Neuron devices, such as compatibility and profiling. It also preserves
TensorBoard’s existing features, including the Debugger plugin, which
may be useful in finding numerical mismatches.

Installation
------------

.. warning::

  When profiling with PyTorch in a conda environment please re-install 
  the correct version of torch as a workaround for https://github.com/aws/aws-neuron-sdk/issues/230.  See issue for details.


Note: on DLAMI v26.0, please do
``conda install numpy=1.17.2 --yes --quiet`` before following the Conda
installation instructions, as the installed numpy version prevents the
update. See :ref:`dlami-neuron-rn` for more info.

This section assumes the Neuron repos have been configured as shown
here: :ref:`neuron-install-guide`

By default, TensorBoard-Neuron will be installed when you install
TensorFlow-Neuron.

If using Conda, there is no standalone package for
``tensorboard-neuron`` at this time, it is currently packaged together
in the ``tensorflow-neuron`` conda package.

Pip
~~~

::

   $ pip install tensorflow-neuron

It can also be installed separately.

::

   $ pip install tensorboard-neuron

Additionally, if you would like to profile your model (see below), you
will also need to have Neuron tools installed.

::

   $ sudo apt install aws-neuron-tools

Note: TensorBoard does not need to be installed to use
TensorBoard-Neuron, and should be replaced with TensorBoard-Neuron if
already installed.

::

   $ pip uninstall tensorboard
   $ pip install tensorboard-neuron

OR

::

   $ pip install tensorboard-neuron --force-reinstall

If TensorBoard-Neuron is not properly installed, the added
functionalities for AWS Neuron may not work. For example, errors such as
``tensorboard: error: unrecognized arguments: --run_neuron_profile`` may
occur when attempting to profile an inference.

Conda
~~~~~

TensorBoard-Neuron is included under the ``tensorflow-neuron`` conda
package.

::

   $ conda update tensorflow-neuron

Profile the network and collect inference traces
------------------------------------------------

When using TensorFlow-Neuron, MXNet-Neuron, or PyTorch-Neuron, raw
profile data will be collected if NEURON_PROFILE environment variable is
set. The raw profile is dumped into the directory pointed by
NEURON_PROFILE environment variable.

The steps to do this:

-  Set NEURON_PROFILE environment variable, e.g.:

::

   export NEURON_PROFILE=/some/output/directory

NOTE: this directory must exist before you move on to the next step.
Otherwise, profile data will not be emitted.

-  Run inference through the framework. See the tutorials for each
   framework for more info.

Visualizing data with TensorBoard-Neuron
----------------------------------------

To view data in TensorBoard-Neuron, run the command below, where
“logdir” is the directory where TensorFlow logs are kept. This logdir
may or may not have any existing logs, or may not even exist yet. AWS
Neuron will populate this directory when using the
``--run_neuron_profile`` option. (Note that this "logdir" is *not* the
same as the NEURON_PROFILE directory that you set during inference, and
in fact, depending on your configuration you may not have any tensorflow
logs. For this step, NEURON_PROFILE still needs to be set to the same
directory you used during your inference run. ``tensorboard_neuron``
will process the neuron profile data from the NEURON_PROFILE directory
at startup.)

::

   $ tensorboard_neuron --logdir /path/to/logdir --run_neuron_profile

By default, TensorBoard-Neuron will be launched at “localhost:6006,” by
specifying "--host" and "--port" option the URL can be changed.

Now, in a browser visit `localhost:6006 <http://localhost:6006/>`__ to
view the visualization or and enter the host and port if specified
above.

.. _tensorboard-howto-check-compatibility:

How to: Check Neuron compatibility
----------------------------------

TensorBoard-Neuron can visualize which operators are supported on Neuron
devices. All Neuron compatible operators would run on Neuron Cores and
other operators would run on CPU.

Step 1: Generate the EVENT files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the TensorFlow APIs to create the event file. See the sample Python
code snippet below for TensorFlow:

::

   import tensorflow as tf

   graph_file = '/path/to/graph_def.pb' # Change path here
   graph_def = tf.GraphDef()
   with open(graph_file, 'rb') as f:
       graph_def.ParseFromString(f.read())

   graph = tf.Graph()
   with graph.as_default():
       tf.import_graph_def(graph_def, name='')

   fw = tf.summary.FileWriter(graph=graph, logdir='/path/to/logdir') # Change logdir here
   fw.flush()

Step 2: Launch Tensorboard-Neuron and navigate to the webpage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See the above section Visualizing data with TensorBoard-Neuron.

Step 3: select “Neuron Compatibility“
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the navigation pane on the left, under the “Color” section, select
“Neuron Compatibility.” |image|

Step 4: View compatible operators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now, the graph should be colored red and/or green. Green indicates that
an operator that is compatible with Neuron devices, while red indicates
that the operator is currently not supported. If there are unsupported
operators, all of these operators’ names will be listed under the
“Incompatible Operations” section. |image1|

How to: Visualize graphs run on a Neuron device
-----------------------------------------------

After successfully analyzing the profiled run on a Neuron device, you
can launch TensorBoard-Neuron to view the graph and see how much time
each operator is taking.

Step 1: Generate the Files
~~~~~~~~~~~~~~~~~~~~~~~~~~

This step requires Neuron tools in order to work.

.. _step-2-launch-tensorboard-neuron-and-navigate-to-the-webpage-1:

Step 2: Launch Tensorboard-Neuron and navigate to the webpage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See the above section Visualizing data with TensorBoard-Neuron

Step 3: select the “Neuron_profile” tag
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The “neuron_profile” tag contains timing information regarding the
inference you profiled. |image2|

Step 4: select “Compute Time”
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the navigation pane on the left, under the “Color” section, select
“Compute time.” |image3|

Step 5: View time taken by various layers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This view will show time taken by each layer and will be colored
according to how much relative time the layer took to compute. A lighter
shade of red means that a relatively small portion of compute time was
spent in this layer, while a darker red shows that more compute time was
used. Some layers may also be blank, which indicates that these layers
may have been optimized out to improve inference performance. Clicking
on a node will show the compute time, if available. |image4|

How to: View detailed profile using the Neuron Profile plugin
-------------------------------------------------------------

To get a better understanding of the profile, you can check out the
Neuron Profile plugin. Here, you will find more information on the
inference, including an overview, a list of the most time-consuming
operators (op profile tool), and an execution timeline view (Chrome
trace).

.. _step-1-generate-the-files-1:

step 1: Generate the files
~~~~~~~~~~~~~~~~~~~~~~~~~~

This step requires Neuron tools in order to work.

.. _step-2-launch-tensorboard-neuron-and-navigate-to-the-webpage-2:

Step 2: Launch Tensorboard-Neuron and navigate to the webpage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See the above section Visualizing data with TensorBoard-Neuron

Step 3: Select the “Neuron Profile” plugin
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On the navigation bar at the top of the page, there will be a list of
active plugins. In this case, you will need to use the “Neuron Profile”
plugin. |image5|\ The plugin may take a while to register on first load.
If this tab does not show initially, please refresh the page.

Step 4a: the profile overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first page you will land on in the Neuron Profile plugin is the
overview page. It contains various information regarding the inference.
|image6| In the “Performance Summary” section, you will see execution
stats, such as the total execution time, the average layer execution
time, and the utilization of NeuronMatrix Units.

The “Neuron Time Graph” shows how long a portion of the graph (a
NeuronOp) took to execute.

The “Top TensorFlow operations executed on Neuron Cores” sections gives
a quick summary of the most time-consuming operators that were executed
on the device.

“Run Environment” shows the information on devices used during this
inference.

Finally, the “Recommendation for Next Steps” section gives helpful
pointers to place to learn more about what to do next

STEP 4B: THE OPERATOR PROFILE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the “Tools” dropdown menu, select “op_profile.”

The “op profile” tool displays the percentage of overall time taken for
each operator, sorted by the most expensive operators at the top. It
gives a better understanding of where the bottlenecks in a model may be.
|image7|

Step 4c: Chrome trace
~~~~~~~~~~~~~~~~~~~~~

In the “Tools” dropdown menu, select “trace_viewer.”

For developers wanting to better understand the timeline of the
inference, the Chrome trace view is the tool for you. It shows the
history of execution organized by the operator names.

Please note that this tool can only be used in Chrome browsers. |image8|

How to: Debug an inference
--------------------------

To make use of the Debugger plugin, you must specify your desired output
tensors before creating the saved model. See :ref:`tensorflow-serving`
for how to create the saved model. Essentially, adding these tensors to
the “outputs” dictionary will allow you to view them in the debugger
later on.

Please note that this feature is currently only available for TensorFlow
users.

Step 1: Launch TensorBoard-Neuron and navigate to the webpage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To use the Debugger plugin, you will need to launch with an extra flag:

::

   $ tensorboard_neuron --logdir /path/to/logdir --debugger_port PORT

where PORT is your desired port number.

Step 2: Modify and run your inference script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to run the inference in “debug mode,” you must use TensorFlow’s
debug wrapper. The following lines will need to be added to your script.

::

   from tensorflow.python import debug as tf_debug

   # The port must be the same as the one used for --debugger_port above
   # in this example, PORT is 7000
   DEBUG_SERVER_ADDRESS = 'localhost:7000'

   # create your TF session here

   sess = tf_debug.TensorBoardDebugWrapperSession(
               sess, DEBUG_SERVER_ADDRESS)

   # run inference using the wrapped session

After adding these modifications, run the script to begin inference. The
execution will be paused before any calculation starts.

Step 3: Select the “debugger” plugin
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On the navigation bar at the top of the page, there will be a list of
active plugins. In this case, you will need to use the “Debugger”
plugin. |image9|

Step 4: Enable watchpoints
~~~~~~~~~~~~~~~~~~~~~~~~~~

In the “Runtime Node List” on the left, there will be a list of
operators and a checkbox next to each. Select all of the operators that
you would like the view the tensor output of. |image10|

step 5: execute inference
~~~~~~~~~~~~~~~~~~~~~~~~~

On the bottom left of the page, there will be a “Continue...” button
that will resume the inference execution. As the graph is executed,
output tensors will be saved for later viewing.

|image11|

Step 6: View tensors
~~~~~~~~~~~~~~~~~~~~

At the bottom of the page, there will be a“Tensor Value Overview”
section that shows a summary of all the output tensors that were
selected as watchpoints in Step 4. |image12| To view more specific
information on a tensor, you can click on a tensor’s value. You may also
hover over the bar in the “Health Pill” column for a more detailed
summary of values. |image13|

.. |image| image:: /images/tb-img1.png
.. |image1| image:: /images/tb-img2.png
.. |image2| image:: /images/tb-img3.png
.. |image3| image:: /images/tb-img4.png
.. |image4| image:: /images/tb-img5.png
.. |image5| image:: /images/tb-img6.png
.. |image6| image:: /images/tb-img7.png
.. |image7| image:: /images/tb-img8.png
.. |image8| image:: /images/tb-img9.png
.. |image9| image:: /images/tb-img10.png
.. |image10| image:: /images/tb-img11.png
.. |image11| image:: /images/tb-img12.png
.. |image12| image:: /images/tb-img13.png
.. |image13| image:: /images/tb-img14.png
