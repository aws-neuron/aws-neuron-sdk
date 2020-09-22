# Conda-TensorFlow Release Notes

This document lists the release notes for the Neuron Conda-TensorFlow package.

# Known Issues

When running TensorFlow script in latest TensorFlow-Neuron conda environment, you may see errors "AttributeError: module 'numpy' has no attribute 'integer'" and "ModuleNotFoundError: No module named 'numpy.core._multiarray_umath'". This is due to older version of numpy. Please update numpy to version 1.18 using the command "conda update numpy".


# [1.15.3.1.0.2043.0_2.0.894.0]

Date: 09/21/2020

## Included Neuron Packages

neuron_cc-1.0.20600.0

tensorboard_neuron-1.15.0.1.0.600.0

tensorflow_neuron-1.15.3.1.0.2043.0

# Known Issues

When running TensorFlow script in latest TensorFlow-Neuron conda environment, you may see errors "AttributeError: module 'numpy' has no attribute 'integer'" and "ModuleNotFoundError: No module named 'numpy.core._multiarray_umath'". This is due to older version of numpy. Please update numpy to version 1.18 using the command "conda update numpy".


# [1.15.3.1.0.1965.0_2.0.778.0]

Date: 08/08/2020

## Included Neuron Packages

neuron_cc-1.0.18001.0

tensorboard_neuron-1.15.0.1.0.570.0

tensorflow_neuron-1.15.3.1.0.1965.0



# [1.15.3.1.0.1953.0_2.0.769.0]

Date: 08/05/2020

## Included Neuron Packages

neuron_cc-1.0.17937.0

tensorboard_neuron-1.15.0.1.0.513.0

tensorflow_neuron-1.15.3.1.0.1889.0


# [1.15.3.1.0.1891.0-2.0.706.0]

Date: 07/16/2020

Now supporting Python 3.7 Conda packages in addition to Python 3.6 Conda packages.

## Included Neuron Packages

neuron_cc-1.0.16861.0

tensorboard_neuron-1.15.0.1.0.513.0

tensorflow_neuron-1.15.3.1.0.1891.0

# [1.15.2.1.0.1782.0-2.0.593.0]

Date: 06/11/2020

## Included Neuron Packages

neuron_cc-1.0.15275.0

tensorboard_neuron-1.15.0.1.0.491.0

tensorflow_neuron-1.15.0.1.0.1796.0


# [1.15.2.1.0.1572.0-2.0.329.0]

Date 5/11/2020

## Included Neuron Packages

neuron-cc-1.0.12696.0

tensorboard_neuron-1.15.0.1.0.466.0

tensorflow_neuron-1.15.2.1.0.1572.0

# [1.15.0.1.0.1333.0-2.0.63.0]

Date 3/26/2020

## Included Neuron Packages

neuron-cc-1.0.9410.0

tensorflow_neuron-1.15.0.1.0.1333.0

tensorboard_neuron-1.15.0.1.0.392.0

# [1.15.0.1.0.1240.0-1.0.918.0]

Date 2/27/2020

## Included Neuron Packages

neuron_cc-1.0.7668.0

tensorflow_neuron-1.15.0.1.0.1240.0

tensorboard_neuron-1.15.0.1.0.366.0

# [1.15.0.1.0.997.0-1.0.733.0]

Date 1/27/2020

## Included Neuron Packages

neuron-cc-1.0.6801.0

tensorflow-neuron-1.15.0.1.0.997.0

tensorboard-neuron-1.15.0.1.0.315.0


# [1.15.0.1.0.803.0-1.0.611.0]

Date 12/20/2019

## Included Neuron Packages

neuron-cc-1.0.5939.0

tensorflow-neuron-1.15.0.1.0.803.0

tensorboard-neuron-1.15.0.1.0.315.0


# [1.15.0.1.0.749.0-1.0.474.0]

Date 12/1/2019

## Included Neuron Packages

neuron-cc-1.0.5301.0

tensorflow-neuron-1.15.0.1.0.749.0

tensorboard-neuron-1.15.0.1.0.306.0


## Known Issues and Limitations


# [1.15.0.1.0.663.0-1.0.298.0]

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
