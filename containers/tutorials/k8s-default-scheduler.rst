
.. _k8s-default-scheduler:

* Make sure :ref:`Neuron device plugin<k8s-neuron-device-plugin>` is running
* Enable the kube-scheduler with option to use configMap for scheduler policy. In your cluster.yml Please update the spec section with the following

   .. code:: bash

      spec:
        kubeScheduler:
        usePolicyConfigMap: true

* Launch the cluster

   .. code:: bash

      kops create -f cluster.yml
      kops create secret --name neuron-test-1.k8s.local sshpublickey admin -i ~/.ssh/id_rsa.pub
      kops update cluster --name neuron-test-1.k8s.local --yes

* Install the neuron-scheduler-extension [Registers neuron-scheduler-extension with kube-scheduler]

    .. code:: bash

        helm upgrade --install neuron-helm-chart oci://public.ecr.aws/neuron/neuron-helm-chart \
            --set "scheduler.enabled=true" \
            --set "scheduler.customScheduler.enabled=false" \
            --set "scheduler.defaultScheduler.enabled=true" \
            --set "npd.enabled=false"
