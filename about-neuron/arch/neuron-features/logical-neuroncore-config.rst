.. _logical-neuroncore-config:

################################
Logical NeuronCore configuration
################################

Logical NeuronCore configuration (LNC) is a set of compiler and runtime settings for instances powered by AWS Trainium2 that 
determines the number of NeuronCores exposed to your machine learning (ML) applications. LNC configuration works by combining 
the compute and memory resources of multiple physical NeuronCores into a single logical NeuronCore. You can configure these settings 
to reduce the number of worker process needed for training and deployment of large-scale models. 



.. contents:: Concepts
    :depth: 1
    :local:
    :backlinks: none

===================
Logical NeuronCores
===================

A logical NeuronCore is a grouping of physical NeuronCores that the Neuron Compiler, Neuron Runtime, Neuron Tools, and Frameworks 
handle as a single unified NeuronCore. Every Trainium2 device contains eight physical NeuronCore-v3. 

=============================
Compiler and runtime settings
=============================
 
LNC configuration is controlled with the following runtime and compiler settings:

| **Neuron Runtime**
| The ``NEURON_LOGICAL_NC_CONFIG`` runtime environment variable controls how many physical NeuronCores are grouped to make up a logical NeuronCore.


| **Neuron compiler flags** 
| The ``--logical-nc-config`` or ``-lnc`` command-line options control the degree of model sharding the compiler performs on an input graph. You must compile your Models to use the LNC configuration set by the Neuron Runtime environment variable. AWS Neuron currently doesn't support setting the compiler flag to a different LNC configuration than the Neuron Runtime environment variable. 

=================================
Logical NeuronCore configurations
=================================

AWS Neuron supports the following Logical NeuronCore configurations:

.. tab-set::

    .. tab-item:: LNC = 2

        A Logical NeuronCore configuration (LNC) of two is the default setting on Trainium2 devices. It combines two physical 
        NeuronCore-v3 into a logical NeuronCore with the software id ``NC_V3d``. When you set Logical NeuronCore configuration to 
        two, it directs Trainium2 devices to expose four ``NC_v3d`` to your machine learning applications. On this setting, 
        a ``Trn2.48xlarge`` instance presents 64 available NeuronCores. The folowing high-level diagram shows a ``Trn2.48xlarge`` 
        instance, connected in a 2D torus topology, with the Logical NeuronCore configuration set to two.

        .. image:: /images/architecture/Trn2/trn2_lnc2.png
            :align: center
            :width: 750
        |

        Trainium2 devices contain four 24GB HBM banks. Each bank is shared by two physical NeuronCore-v3. 
        When LNC=2, the two physical NeuronCores share a single address space. Workers on each of the 
        two physical NeuronCores can access tensors and perform local collective operations without 
        accessing the network. The following diagram shows how a logical NeuronCore is presented to the 
        software under this configuration.

        .. image:: /images/architecture/NeuronCore/lnc_2.png
            :align: center
            :width: 450
        |

        To set the Logical NeuronCore configuration to two, use the following runtime and compiler flag combination:

        | **Runtime environment variable:**
        | ``NEURON_LOGICAL_NC_CONFIG`` = 2

        | **Compiler flag:**
        | ``-lnc`` = 2 
        |

    .. tab-item:: LNC = 1

        When you set the Logical NeuronCore configuration to one, it assigns each physical NeuronCore-v3 to a single logical 
        NeuronCore with the software id ``NC_V3``. This directs Trainium2 devices to expose eight ``NC_v3`` to your machine learning 
        applications. On this setting, a ``Trn2.48xlarge`` instance presents 128 available NeuronCores. 
        The following high-level diagram shows a ``Trn2.48xlarge`` instance, connected in a 2D torus topology, 
        with the Logical NeuronCore configuration set to one.

        .. image:: /images/architecture/Trn2/trn2_lnc1.png
            :align: center
            :width: 750
        |

        Trainium2 devices contain four 24GB HBM banks. Each bank is shared by two physical NeuronCore-v3. 
        When the Logical NeuronCore configuration is set to one, both physical NeuronCores have access to the entire 24GB HBM bank. The following 
        diagram shows how logical NeuronCores are presented to the software under this configuration.

        .. image:: /images/architecture/NeuronCore/lnc_1.png
            :align: center
            :width: 475
        |

        To set the Logical NeuronCore configuration to one, use the following runtime and compiler flag combination:

        | **Runtime environment variable:**
        | ``NEURON_LOGICAL_NC_CONFIG`` = 1

        | **Compiler flag:**
        | ``-lnc`` = 1
        |

        
        


 