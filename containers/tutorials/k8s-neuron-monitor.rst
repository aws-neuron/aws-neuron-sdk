.. _k8s-neuron-monitor:

 Neuron monitor Container
 ========================

 Neuron monitor is primary observability tool for neuron devices. For details of neuron monitor, please refer to the `neuron monitor guide <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-sys-tools/neuron-monitor-user-guide.html>`_. This tutorial describes deploying neuron monitor as a daemonset on the kubernetes cluster.

  
* Download the neuron monitor  yaml file. :download:`k8s-neuron-monitor-daemonset.yml </src/k8/k8s-neuron-monitor-daemonset.yml>`
* Apply the Neuron monitor yaml to create a daemonset on the cluster with the following command

    .. code:: bash

        kubectl apply -f k8s-neuron-monitor.yml
    
* Verify that neuron monitor daemonset is running

    .. code:: bash

        kubectl get ds neuron-monitor --namespace neuron-monitor

    Expected result (with 2 nodes in cluster):

    .. code:: bash

        NAME                             DESIRED   CURRENT   READY   UP-TO-DATE   AVAILABLE   NODE SELECTOR   AGE
        neuron-monitor                     2         2         2       2            2           <none>          27h


* Get the neuron-monitor pod names
    .. code:: bash

        kubectl get pods

    Expected result

    .. code:: bash 

        NAME                   READY   STATUS    RESTARTS   AGE
        neuron-monitor-slsxf   1/1     Running   0          17m
        neuron-monitor-wc4f5   1/1     Running   0          17m
    

* Verify the prometheus endpoint is available 
    .. code:: bash

        kubectl exec neuron-monitor-wc4f5 -- wget -q --output-document - http://127.0.0.1:8000

    Expected result

    .. code:: bash

        # HELP python_gc_objects_collected_total Objects collected during gc
        # TYPE python_gc_objects_collected_total counter
        python_gc_objects_collected_total{generation="0"} 362.0
        python_gc_objects_collected_total{generation="1"} 0.0
        python_gc_objects_collected_total{generation="2"} 0.0
        # HELP python_gc_objects_uncollectable_total Uncollectable objects found during GC
        # TYPE python_gc_objects_uncollectable_total counter
