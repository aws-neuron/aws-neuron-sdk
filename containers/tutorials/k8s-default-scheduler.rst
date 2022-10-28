
.. _k8s-default-scheduler:

* Make sure :ref:`Neuron device plugin<k8s-neuron-device-plugin>` is running
* Download the scheduler config map :download:`k8s-neuron-scheduler-configmap.yml </src/k8/k8s-neuron-scheduler-configmap.yml>`
* Download the scheduler extension :download:`k8s-neuron-scheduler.yml </src/k8/k8s-neuron-scheduler.yml>`
* Enable the kube-scheduler with option to use configMap for scheduler policy. In your cluster.yml Please update the spec section with the following

   ::

      spec:
        kubeScheduler:
        usePolicyConfigMap: true

* Launch the cluster

   ::

      kops create -f cluster.yml
      kops create secret --name neuron-test-1.k8s.local sshpublickey admin -i ~/.ssh/id_rsa.pub
      kops update cluster --name neuron-test-1.k8s.local --yes

* Apply the k8s-neuron-scheduler-configmap.yml [Registers neuron-scheduler-extension with kube-scheduler]

   ::

      kubectl apply -f k8s-neuron-scheduler-configmap.yml

* Launch the neuron-scheduler-extension

   ::

      kubectl apply -f k8s-neuron-scheduler.yml