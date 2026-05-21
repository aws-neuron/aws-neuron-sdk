This approach integrates the Neuron Scheduler Extension directly with the Kubernetes default scheduler. This method requires access to modify the default scheduler configuration.

**Prerequisites**

Ensure that the Neuron Device Plugin is running.

**Step 1: Configure kube-scheduler**

Enable the kube-scheduler to use a ConfigMap for scheduler policy. In your ``cluster.yml``, update the spec section with the following:

.. code:: yaml

    spec:
      kubeScheduler:
        usePolicyConfigMap: true

**Step 2: Launch the Cluster**

Create and launch the cluster:

.. code:: bash

    kops create -f cluster.yml
    kops create secret --name neuron-test-1.k8s.local sshpublickey admin -i ~/.ssh/id_rsa.pub
    kops update cluster --name neuron-test-1.k8s.local --yes

**Step 3: Install Neuron Scheduler Extension**

Install the Neuron Scheduler Extension and register it with kube-scheduler:

.. code:: bash

    helm upgrade --install neuron-helm-chart oci://public.ecr.aws/neuron/neuron-helm-chart \
        --set "scheduler.enabled=true" \
        --set "scheduler.customScheduler.enabled=false" \
        --set "scheduler.defaultScheduler.enabled=true" \
        --set "npd.enabled=false"
