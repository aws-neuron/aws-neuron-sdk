# TensorBoard-Neuron Release Notes

# Known Issues and Limitations - updated 09/22/2020
* Issue: The Chrome trace view in the Neuron Profile plugin may not work due to some APIs being deprecated in Chrome 80+ (see https://github.com/tensorflow/tensorboard/issues/3209).  Solution:
	1. Find the specific run that you would like to view the Chrome trace for.
	2. Go to `<tensorboard url>/data/plugin/neuron_profile/data?host=&run=<run>&tag=trace_viewer` and save the chrome trace data.  By default the tensorboard url is `localhost:6006`.  For example, if the run you want to view is `node3/1590687636`, go to `http://localhost:6006/data/plugin/neuron_profile/data?host=&run=node3%2F1590687636&tag=trace_viewer`.
	3. In Chrome, go to `chrome://tracing`, click the load button, and select to saved data.
	
# [1.15.0.1.0.600.0]

Date: 09/22/2020

## Summary
* Minor internal improvements.



# [1.15.0.1.0.570.0]

Date: 08/08/2020

## Summary
* Minor internal improvements.



# [1.15.0.1.0.513.0]

Date: 07/16/2020

## Summary
* Minor internal improvements.

	

# [1.15.0.1.0.491.0]

Date 6/11/2020

## Summary
Fix issue where utilization was missing in the op-profile view.

## Resolved Issues
* The op-profile view in the Neuron Profile plugin now correctly shows the overall NeuronCore utilization.


# [1.15.0.1.0.466.0]

Date 5/11/2020

## Summary
Fix potential installation issue when installing both tensorboard and tensorboard-neuron.

## Resolved Issues
* Added tensorboard as a dependency in tensorboard-neuron.  This prevents the issue of overwriting tensorboard-neuron features when tensorboard is installed after tensorboard-neuron.

## Other Notes

# [1.15.0.1.0.392.0]

Date 3/26/2020

## Summary
Added ability to view CPU node latency in the Graphs plugin and the Neuron Profile plugins.

## Major New Features

* Added an aggregate view in addition to the current Neuron subgraph view for both the Graphs plugin and the Neuron Profile plugin.
 * When visualizing a graph executed on a Neuron device, CPU node latencies are available when coloring the graph by "Compute time" using the "neuron_profile" tag.
 * The Neuron Profile plugin now has an overview page to compare time spent on Neuron device versus on CPU.

## Other Notes

* Requires Neuron-RTD config option "enable_node_profiling" to be set to "true"

# [1.15.0.1.0.366.0]

Date 02/27/2020

## Summary

Reduced load times and fixed crashes when loading large models for visualization.

## Resolved Issues

* Enable large attribute filtering by default
* Reduced load time for graphs with attributes larger than 1 KB
* Fixed a fail to load graphs with many large attributes totaling more than 1 GB in size

# [1.15.0.1.0.315.0]

Date 12/20/2019

## Summary 

No major chages or fixes. Released with other Neuron packages.

# [1.15.0.1.0.306.0]

Date 12/1/2019

## Summary

## Major New Features

## Resolved Issues

## Known Issues & Limits

Same as prior release

## Other Notes

# [1.15.0.1.0.280.0]

Date 11/29/2019

## Summary

Initial release packaged with DLAMI.

## Major New Features

N/A, initial release. 

 See user guide here: https://github.com/aws/aws-neuron-sdk/blob/master/docs/neuron-tools/getting-started-tensorboard-neuron.md

## Resolved Issues

N/A - first release

## Known Issues & Limits

* Must install TensorBoard-Neuron by itself, or after regular TensorBoard is installed. If regular Tensorboard is installed after TensorBoard-Neuron, it may overwrite some needed files.
* Utilization missing in Op Profile due to missing FLOPs calculation (see overview page instead)
* Neuron Profile plugin may not immediately show up on launch (try reloading the page)
* Graphs with NeuronOps may take a long time to load due to attribute size
* Instructions that cannot be matched to a framework layer/operator name show as “” (blank)
* CPU Usage section in chrome-trace is not applicable
* Debugger currently supports TensorFlow only
* Visualization requires a TensorFlow-compatible graph

## Other Notes


