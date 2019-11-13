# Getting Started: TensorBoard-Neuron

This guide is for developers who want to better understand how their model runs on Neuron devices.

TensorBoard-Neuron is adapted to provide useful information related to Neuron devices, such as compatibility and profiling.  It also preserves TensorBoard’s Debugger plugin, which may be useful in finding numerical mismatches.

## Installation

By default, TensorBoard-Neuron will be installed when you install TensorFlow-Neuron.

```
$ pip install tensorflow-neuron
```

It can also be installed separately.

```
$ pip install tensorboard-neuron
```

Additionally, if you would like to profile your model (see below), you will also need to have Neuron tools installed.

```
$ sudo apt install aws-neuron-tools
```

## Collecting and post-processing inference traces

### Using a supported framework

When using TensorFlow-Neuron, MXNet-Neuron, or PyTorch-Neuron, raw profile data will automatically be collected and dumped to the directory defined by the NEURON_PROFILE environment variable if set.  There should be at least three files: one compiled model (.neff), one profile session file (.ntff) per inference, and one TensorFlow GraphDef protobuf file (.pb).

After collecting the profile data, launch TensorBoard-Neuron with the special flag “--run_neuron_profile" to analyze the raw data and generate the output files before starting the server.

```
$ export NEURON_PROFILE=/some/output/directory

*** Run inference through the framework ***

$ tensorboard_neuron --logdir /path/to/logdir --run_neuron_profile
```

### Standalone

Analysis of raw profile data can also be done without the help of the frameworks.  However, you will need to collect the data by yourself.  See the “neuron-cli” tool as an example of how to do this.

Then, we will need to analyze this data with the profiler.  You will need the NEFF (your compiled model), the NTFF (the collected profile data), your desired output directory, and the TensorFlow GraphDef of your model.

Finally, launch TensorBoard-Neuron with the logdir set to the output directory specified above.

```
*** Run inference and collect profile data ***

$ neuron-profile analyze \
--neff-path /path/to/neff \
--session-file /path/to/ntff \
--output-dir /path/to/logdir \
--graph-file=/path/to/graphdef

$ tensorboard_neuron --logdir /path/to/logdir
```

## Visualizing data with TensorBoard-Neuron

To view data in TensorBoard-Neuron, run the command below, where “logdir” is the directory with the generated profile data.  See the Collecting and post-processing inference traces section for more info.

```
tensorboard_neuron --logdir /path/to/logdir
```

Optionally, you can specify the host and port.  By default, TensorBoard-Neuron will be launched at “localhost:6006,” but adding the  "--host"  and  "--port" flag will change this to whatever you wish.

Open your favorite browser and enter the host and port if specified above; otherwise, simply go to the default “localhost:6006” instead.


## How to: Check Neuron compatibility

TensorBoard-Neuron can help you visualize which operators are supported on Neuron devices.

### Step 1: Generate the files

If you have already run an inference and collected the profile artifacts, see the above section the Collecting and post-processing inference traces to generate the TensorBoard-Neuron files.

Otherwise, please use the TensorFlow APIs to create the event file.  See the sample Python code snippet below for TensorFlow:

```
import tensorflow as tf

your_graph_file = '/path/to/graph/file'
your_graph_def = tf.GraphDef()
with open(your_graph_file, 'rb') as f:
    graph_def.ParseFromString(f.read())
    
your_graph = tf.Graph()
with your_graph.as_default():
    tf.import_graph_def(your_graph_def, name='')
    
fw = tf.summary.FileWriter(graph=yourgraph, logdir='/path/to/logdir'
fw.flush()
```

### Step 2: Launch Tensorboard-Neuron and navigate to the webpage

See the above section Visualizing data with TensorBoard-Neuron.

### Step 3: select “Neuron MLA Compatibility“

In the navigation pane on the left, under the “Color” section, select “Neuron MLA Compatibility.”
![image](https://github.com/aws/aws-neuron-sdk/blob/master/docs/images/Screen%20Shot%202019-11-11%20at%202.18.17%20PM.png)

### Step 4: View compatible operators

Now, the graph should be colored red and/or green.  Green indicates that an operator that is compatible with Neuron devices, while red indicates that the operator is currently not supported.  If there are unsupported operators, all of these operators’ names will be listed under the “Incompatible Operations” section.
![image](https://github.com/aws/aws-neuron-sdk/blob/master/docs/images/Screen%20Shot%202019-11-11%20at%202.15.17%20PM.png)

## How to: Visualize graphs run on a Neuron device

After successfully analyzing the profiled run on a Neuron device, you can launch TensorBoard-Neuron to view the graph and see how much time each operator is taking.

### Step 1: Generate the Files

This step requires Neuron tools in order to work.  See the above section Collecting and post-processing inference traces to generate the TensorBoard-Neuron files.

### Step 2: Launch Tensorboard-Neuron and navigate to the webpage

See the above section Visualizing data with TensorBoard-Neuron

### Step 3: select the “Neuron_profile” tag

The “neuron_profile” tag contains timing information regarding the inference you profiled.
![image](https://github.com/aws/aws-neuron-sdk/blob/master/docs/images/Screen%20Shot%202019-11-11%20at%202.32.13%20PM.png)

### Step 4: select “Compute Time”

In the navigation pane on the left, under the “Color” section, select “Compute time.”

![image](https://github.com/aws/aws-neuron-sdk/blob/master/docs/images/Screen%20Shot%202019-11-11%20at%202.32.46%20PM.png)

### Step 5: View time taken by various layers

This view will show time taken by each layer and will be colored according to how much relative time the layer took to compute.  A lighter shade of red means that a relatively small portion of compute time was spent in this layer, while a darker red shows that more compute time was used.  Some layers may also be blank, which indicates that these layers may have been optimized out to improve inference performance.  Clicking on a node will show the compute time, if available.

![image](https://github.com/aws/aws-neuron-sdk/blob/master/docs/images/Screen%20Shot%202019-11-12%20at%2011.09.58%20AM.png)

## How to: View detailed profile using the Neuron Profile plugin

To get a better understanding of the profile, you can check out the Neuron Profile plugin.  Here, you will find more information on the inference, including an overview, a list of the most time-consuming operators (op profile tool), and an execution timeline view (Chrome trace).

### step 1: Generate the files

This step requires Neuron tools in order to work.  See the above section Collecting and post-processing inference traces to generate the TensorBoard-Neuron files.

### Step 2: Launch Tensorboard-Neuron and navigate to the webpage

See the above section Visualizing data with TensorBoard-Neuron

### Step 3: Select the “Neuron Profile” plugin

On the navigation bar at the top of the page, there will be a list of active plugins.  In this case, you will need to use the “Neuron Profile” plugin.

![image](https://github.com/aws/aws-neuron-sdk/blob/master/docs/images/Screen%20Shot%202019-11-11%20at%203.22.54%20PM.png)

The plugin may take a while to register on first load.  If this tab does not show initially, please refresh the page.

### Step 4a: the profile overview

The first page you will land on in the Neuron Profile plugin is the overview page.  It contains various information regarding the inference.

![image](https://github.com/aws/aws-neuron-sdk/blob/master/docs/images/Screen%20Shot%202019-11-11%20at%203.05.41%20PM.png)

In the “Performance Summary” section, you will see execution stats, such as the total execution time, the average layer execution time, and the utilization of Neuron MLA Matrix Units.

The “Neuron MLA Time Graph” shows how long a portion of the graph (a NeuronOp) took to execute.

The “Top TensorFlow operations executed on Neuron MLA” sections gives a quick summary of the most time-consuming operators that were executed on the device.

“Run Environment” shows the information on devices used during this inference.

Finally, the “Recommendation for Next Steps” section gives helpful pointers to place to learn more about what to do next

### STEP 4B: THE OPERATOR PROFILE

In the “Tools” dropdown menu, select “op_profile.”

The “op profile” tool displays the percentage of overall time taken for each operator, sorted by the most expensive operators at the top.  It gives a better understanding of where the bottlenecks in a model may be.

![image](https://github.com/aws/aws-neuron-sdk/blob/master/docs/images/Screen%20Shot%202019-11-11%20at%203.16.37%20PM.png)

### Step 4c: Chrome trace

In the “Tools” dropdown menu, select “trace_viewer.”

For developers wanting to better understand the timeline of the inference, the Chrome trace view is the tool for you.  It shows the history of execution organized by the operator names.

Please note that this tool can only be used in Chrome browsers.

![image](https://github.com/aws/aws-neuron-sdk/blob/master/docs/images/Screen%20Shot%202019-11-11%20at%203.20.11%20PM.png)

## How to: Debug an inference

To make use of the Debugger plugin, you must specify your desired output tensors before creating the saved model.  See [Step 1: Get a TensorFlow SavedModel that runs on Inferentia: Getting Started: TensorFlow-Neuron](LINK) for how to create the saved model.  Essentially, adding these tensors to the “outputs” dictionary will allow you to view them in the debugger later on.

Please note that this feature is currently only available for TensorFlow users.

### Step 1: Launch TensorBoard-Neuron and navigate to the webpage

To use the Debugger plugin, you will need to launch with an extra flag:

```
$ tensorboard_neuron --logdir /path/to/logdir --debugger_port PORT
```

where PORT is your desired port number.

See the above section Visualizing data with TensorBoard-Neuron for more details.

### Step 2: Modify and run your inference script

In order to run the inference in “debug mode,” you must use TensorFlow’s debug wrapper.  The following lines will need to be added to your script.

```
from tensorflow.python import debug as tf_debug

# The port must be the same as the one used for --debugger_port above
# in this example, PORT is 7000
DEBUG_SERVER_ADDRESS = 'localhost:7000'

# create your TF session here

sess = tf_debug.TensorBoardDebugWrapperSession(
            sess, DEBUG_SERVER_ADDRESS)
            
# run inference using the wrapped session
```

After adding these modifications, run the script to begin inference.  The execution will be paused before any calculation starts.

### Step 3: Select the “debugger” plugin

On the navigation bar at the top of the page, there will be a list of active plugins.  In this case, you will need to use the “Debugger” plugin.

![image](https://github.com/aws/aws-neuron-sdk/blob/master/docs/images/Screen%20Shot%202019-11-11%20at%205.05.06%20PM.png)

### Step 4: Enable watchpoints

In the “Runtime Node List” on the left, there will be a list of operators and a checkbox next to each.  Select all of the operators that you would like the view the tensor output of.

![image](https://github.com/aws/aws-neuron-sdk/blob/master/docs/images/Screen%20Shot%202019-11-12%20at%2010.45.32%20AM.png)

### step 5: execute inference

On the bottom left of the page, there will be a “Continue...” button that will resume the inference execution.  As the graph is executed, output tensors will be saved for later viewing.

![image](https://github.com/aws/aws-neuron-sdk/blob/master/docs/images/Screen%20Shot%202019-11-12%20at%2010.46.14%20AM.png)

### Step 6: View tensors

At the bottom of the page, there will be a“Tensor Value Overview” section that shows a summary of all the output tensors that were selected as watchpoints in Step 4.

![image](https://github.com/aws/aws-neuron-sdk/blob/master/docs/images/Screen%20Shot%202019-11-12%20at%2010.47.32%20AM.png)

To view more specific information on a tensor, you can click on a tensor’s value.  You may also hover over the bar in the “Health Pill” column for a more detailed summary of values.

![image](https://github.com/aws/aws-neuron-sdk/blob/master/docs/images/Screen%20Shot%202019-11-12%20at%2010.48.15%20AM.png)
