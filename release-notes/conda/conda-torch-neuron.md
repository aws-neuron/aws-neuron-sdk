# PyTorch-Neuron Conda Package Release notes

This document lists the release notes for the Neuron Conda-PyTorch package.

# Known Issues (updated 11/17/2020)

- Conda environment aws_neuron_pytorch_p36 of Conda DLAMI v36 cannot be updated to this latest (1.5.1.1.0.1978.0) PyTorch-Neuron Conda package using "conda update torch-neuron" command. To use the latest PyTorch-Neuron Conda package, please create a new Conda environment and install PyTorch-Neuron Conda package there using "conda install -c https://conda.repos.neuron.amazonaws.com torch-neuron".

- Conda environment aws_neuron_pytorch_p36 of Conda DLAMI v30 to v35 can be updated using the following commands:
```bash
conda install --force torch-neuron=1.5.1.1.0.1978.0
conda install --force numpy=1.18.1
```

# [1.5.1.1.0.1978.0]

Date: 11/17/2020

## Included Neuron Packages

[neuron-cc-1.0.23977.0](../neuron-cc.md)

[torch_neuron-1.5.1.1.0.1978.0](../torch-neuron.md)

# Known Issues

- Conda environment aws_neuron_pytorch_p36 of Conda DLAMI v36 cannot be updated to this latest (1.5.1.1.0.1978.0) PyTorch-Neuron Conda package using "conda update torch-neuron" command. To use the latest PyTorch-Neuron Conda package, please create a new Conda environment and install PyTorch-Neuron Conda package there using "conda install -c https://conda.repos.neuron.amazonaws.com torch-neuron".

- Conda environment aws_neuron_pytorch_p36 of Conda DLAMI v30 to v35 can be updated using the following commands:
```bash
conda install --force torch-neuron=1.5.1.1.0.1978.0
conda install --force numpy=1.18.1
```

# [1.5.1.1.0.1721.0_2.0.1017.0]

Date: 09/22/2020

## Included Neuron Packages

[neuron-cc-1.0.20600.0](../neuron-cc.md)

[torch_neuron-1.0.1721.0](../torch-neuron.md)

## Resolved Issues

When TorchVision is updated to version >= 0.5, running Neuron compilation would crash with "Segmentation fault (core dumped)" error.

## Known Issues

- When TorchVision is updated to version >= 0.5, running Neuron compilation would crash with "Segmentation fault (core dumped)" error. This issue is resolved with version 1.5.1.1.0.1721.0_2.0.1017.0 of PyTorch-Neuron Conda package (9/22/2020 release).
- When running PyTorch script in latest Torch-Neuron conda environment, you may see errors "AttributeError: module 'numpy' has no attribute 'integer'" and "ModuleNotFoundError: No module named 'numpy.core.\_multiarray_umath'". This is due to older version of numpy. Please update numpy to version 1.18 using the command "conda install --force numpy=1.18.1".  
- Due to changes to PyTorch-Neuron Conda package content in this release, updating from aws_neuron_pytorch_p36 of Conda DLAMI (v35 or earlier) would require the following to update:
```bash
conda install --force torch-neuron=1.5.1.1.0.1721.0
conda install --force numpy=1.18.1
```

# [1.5.1.1.0.298.0_2.0.880.0]

Date: 08/08/2020

## Included Neuron Packages

[neuron-cc-1.0.18001.0](../neuron-cc.md)

[torch_neuron-1.0.1532.0](../torch-neuron.md)

torch_neuron_base-1.5.1.1.0.298.0


# [1.5.1.1.0.258.0_2.0.871.0]

Date: 08/05/2020

## Included Neuron Packages

[neuron-cc-1.0.17937.0](../neuron-cc.md)

[torch_neuron-1.0.1522.0](../torch-neuron.md)

torch_neuron_base-1.5.1.1.0.258.0


# [1.5.1.1.0.251.0_2.0.783.0]

Date: 07/16/2020

Now supporting Python 3.7 Conda packages in addition to Python 3.6 Conda packages.

## Included Neuron Packages

[neuron-cc-1.0.16861.0](../neuron-cc.md)

[torch_neuron-1.0.1386.0](../torch-neuron.md)

torch_neuron_base-1.5.1.1.0.251.0

# [1.3.0.1.0.215.0-2.0.633.0]

Date 6/11/2020

## Included Neuron Packages

[neuron-cc-1.0.15275.0](../neuron-cc.md)

[torch_neuron-1.0.1168.0](../torch-neuron.md)

torch_neuron_base-1.3.0.1.0.215.0

# [1.3.0.1.0.170.0-2.0.349.0]

Date 5/11/2020

## Included Neuron Packages

[neuron-cc-1.0.12696.0](../neuron-cc.md#1068010)

[torch_neuron-1.0.1001.0](../torch-neuron.md#106720)

torch_neuron_base-1.3.0.1.0.170.0

# [1.3.0.1.0.90.0_2.0.62.0]

Date 3/26/2020

## Included Neuron Packages

[neuron-cc-1.0.9410.0](../neuron-cc.md#1068010)

[torch_neuron-1.0.825.0](../torch-neuron.md#106720)

torch_neuron_base-1.3.0.1.0.90.0

# [1.3.0.1.0.90.0-1.0.918.0]

Date: 2/27/2020

## Included Neuron Packages

[neuron_cc-1.0.7878.0](../neuron-cc.md#1068010)

[torch_neuron-1.0.763.0](../torch-neuron.md#106720)

torch_neuron_base-1.3.0.1.0.90.0

## Known Issues and Limitations


### [Conda Tensorflow Release Notes](../tensorflow-neuron.md)

# [1.3.0.1.0.41.0-1.0.737.0]

Date: 1/27/2020

## Included Neuron Packages

[neuron-cc-1.0.6801.0](../neuron-cc.md#1068010)

[torch-neuron-1.0.672.0](../torch-neuron.md#106720)

torch-neuron-base-1.3.0.1.0.41.0

## Known Issues and Limitations
