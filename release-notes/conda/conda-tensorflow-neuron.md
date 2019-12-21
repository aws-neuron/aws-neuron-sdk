# Conda-TensorFlow Release Notes

This document lists the release notes for the Neuron Conda-TensorFlow package.

# [1.15.0.1.0.803.0_1.0.611.0]

Date 12/20/2019

## Included Neuron Packages

neuron-cc-1.0.5939.0

tensorflow-neuron-1.15.0.1.0.803.0

tensorboard-neuron-1.15.0.1.0.315.0


# [1.15.0.1.0.749.0_1.0.474.0]

Date 12/1/2019

## Included Neuron Packages

neuron-cc-1.0.5301.0

tensorflow-neuron-1.15.0.1.0.749.0

tensorboard-neuron-1.15.0.1.0.306.0


## Known Issues and Limitations


# [1.15.0.1.0.663.0_1.0.298.0]

Date:  11/25/2019

This version is only available from the release DLAMI v26.0. Please [update](../dlami-release-notes.md#known-issues) to latest version.

## Included Neuron Packages

neuron-cc-1.0.4680.0

tensorflow-neuron-1.15.0.1.0.663.0

tensorboard-neuron-1.15.0.1.0.280.0

## Known Issues and Limitations

Please update to the latest conda package release.

```bash
source activate <conda environment>
conda update tensorflow-neuron
```
In TensorFlow-Neuron conda environment (aws_neuron_tensorflow_p36) of DLAMI v26.0, the installed numpy version prevents update to latest conda package version. Please do "conda install numpy=1.17.2 --yes --quiet" before "conda update tensorflow-neuron". (See [DLAMI Release Notes](../dlami-release-notes.md)).

```bash
source activate aws_neuron_tensorflow_p36
conda install numpy=1.17.2 --yes --quiet
conda update tensorflow-neuron
```
