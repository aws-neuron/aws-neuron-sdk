# TensorBoard-Neuron Release Notes

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


