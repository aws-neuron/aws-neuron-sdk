.. _k8s-neuron-scheduler-flow:

Neuron Scheduler Extension Flow Diagram
---------------------------------------

::




                                                                           +----------------------------+
                                                                           | POD Manifest               |
                                                                           | with Request               |
                                                                           | aws.amazon.com/neuroncore:2|
                                                                           |                            |
                                                                           |                            |
                                                       2                   +-------------+--------------+
                                            +--------------------------------+           |
                                            |                                |           |
                                            |                                |           | 3
             +------------------------------+-----+                          |           |
             |           Kubelet in INF1/TRN1 Node|                          |           |
             |                                    +<-----------+             |           |
             +-----+---------------------+--------+            |       +-----v-----------v--------------+
                   |                     ^                     |       |          Kube-Scheduler        |
                   |                     |                     |       |                                |
                   |                     |                     |       +--^------+---------------+------+
                 9 |                  1  |                     |          |      |               |
                   |                     |                    8|         5|      |4              |
                   |                     |                     |          |      |               |
                   |                     |                     |          |      |               |6
                   v                     |                     |          |      |               |
             +-----+---------------------+--------+            |       +--+------v---------------v------+
             |    neuron-device-plugin            |            +-------+       neuron|scheduler|ext     |
             |    in INF1/TRN1 node               |                    +---------------------+----------+
             +----+----------------------+--------+                                          |
                  |                      |                                                   |7
                  |                      |10                                                 |
                  |                      |                                                   v
                11|                      |                                         +---------+-------+
                  |                      |                                         |POD Manifest:    |
                  |                      |                                         |Annotation:      |
                  |                      |                                         |NEURON_CORES:2,3 |
                  v                      +---------------------------------------->+                 |
   --device=/dev/neuron1 --env NEURON_RT_VISIBLE_CORES=2,3                         |                 |
                                                                                   |                 |
                                                                                   +-----------------+

   1. neuron-device-plugin returns the list of Neuron cores/devices to kublet
   2. Kubelet advertises the Core/Device list to K8s API server (in turn to kube-scheduler)
   3. POD Request for neuron cores/devices [Kube-Scheduler picks up the POD creation request]
   4. kube-scheduler calls the neuron-scheduler-extn filter function with list of nodes and POD Specification
   5. neuron-scheduler-extn scans through the nodes and filters out nodes with non
   contiguous cores/devices and returns the nodes that are capable of supporing the given POD specification
   6. kube-scheduler calls the neuron-scheduler-extn bind function with pod and node
   7. neuron-scheduler-extn updates the POD annotation with allocated neuron core/device Ids (contiguous)
   8. neuron-scheduler-extn sends the bind request to kubelet of the selected node
   9. Kubelet calls the Alloc function of the neuron-device-plugin
   10. neuron-device-plugin queries the POD Annotation for allocated core/device Ids
   11. neuron-device-plugin exports the devices & visisble cores to container runtime
