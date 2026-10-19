.. meta::
   :description: Install and configure Neuron Observer on an Amazon EKS cluster with Trainium nodes, and export metrics to Prometheus, Amazon Managed Service for Prometheus, or Amazon CloudWatch.
   :keywords: Neuron Observer, install, configure, EKS, Helm, DaemonSet, Trainium, OpenTelemetry, ADOT, CloudWatch, Prometheus, AMP
   :date-modified: 10/15/2026
   :content-type-template: _content-types/procedural-how-to.rst

.. _neuron-observer-install-configure:

==========================================================
How to install and configure Neuron Observer on Amazon EKS
==========================================================

Overview
-------------

The Helm chart deploys Neuron Observer as a DaemonSet on each Trainium node in your Amazon EKS cluster. Configure Prometheus, Amazon Managed Service for Prometheus (AMP), or Amazon CloudWatch as the export destination. The AMP and CloudWatch options also deploy an OpenTelemetry collector.

.. note::

   Neuron Observer is in public beta. APIs, metric names, and installation steps can change before general availability.

Prerequisites
-------------

- **EKS cluster** with Trainium 2 (Trn2) or Trainium 3 (Trn3) node groups.
- **Neuron driver loaded** on nodes (``/dev/neuron0`` must exist on each Neuron node).
- **Nodes labeled** with ``neuron.amazonaws.com/present: "true"`` (applied automatically by the Neuron device plugin).
- **Network access to Amazon ECR Public** from the workstation and cluster nodes:

  - Image — ``public.ecr.aws/neuron/neuron-observer:0.2.0``
  - Chart — ``oci://public.ecr.aws/neuron/neuron-helm-chart`` version ``1.10.0``

- **AWS Distro for OpenTelemetry (ADOT) EKS add-on** (installs the OpenTelemetry Operator, required only for the AMP and CloudWatch options).
- **One or more export destinations:**

  - Prometheus installed in-cluster to scrape the daemon's native ``/metrics`` endpoint.
  - An AMP workspace and IAM permission for ``aps:RemoteWrite``, reachable from the collector pod through Pod Identity or IRSA.
  - IAM permissions for CloudWatch metrics and logs, reachable from the collector pod through Pod Identity or IRSA.

- **helm 3.8+** on the workstation (needed for OCI-registry chart pulls).

Deploy Neuron Observer
----------------------

**1:** (AMP and CloudWatch only) Install the ADOT operator

The OpenTelemetry Operator deploys the collector used by the AMP and CloudWatch pipelines as an ``OpenTelemetryCollector`` custom resource. Install the operator on EKS with the ADOT EKS add-on. Skip this step if you use only Prometheus.

.. code-block:: bash

   # Install cert-manager (required by the operator)
   kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.17.0/cert-manager.yaml

   # Wait for cert-manager to be ready
   kubectl wait --for=condition=Available deployment/cert-manager -n cert-manager --timeout=120s
   kubectl wait --for=condition=Available deployment/cert-manager-webhook -n cert-manager --timeout=120s

   # Install the ADOT EKS add-on
   aws eks create-addon --cluster-name <your-cluster-name> --addon-name adot --region <your-region>

Verify the operator is running:

.. code-block:: bash

   kubectl get pods -n opentelemetry-operator-system
   # Expected: one pod in Running state

If you already have the ADOT add-on or the OpenTelemetry Operator installed, skip this step.

**2:** (Optional) Install Prometheus

If Prometheus is not already installed in your cluster, install the ``kube-prometheus-stack`` Helm chart.

.. code-block:: bash

   # Add the Helm repo
   helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
   helm repo update

   # Install into a monitoring namespace
   kubectl create namespace monitoring
   helm install prometheus prometheus-community/kube-prometheus-stack \
     --namespace monitoring \
     --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false

The chart installs Prometheus and Grafana. If Prometheus is already running, skip this step.

To access Grafana, retrieve the admin password and port-forward:

.. code-block:: bash

   kubectl --namespace monitoring get secrets prometheus-grafana \
     -o jsonpath="{.data.admin-password}" | base64 -d ; echo
   kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80
   # Open http://localhost:3000 (username: admin)

**3:** (AMP and CloudWatch only) Set up the collector IAM role

The ADOT collector requires AWS permissions to push metrics to AMP or CloudWatch. Grant the permissions through EKS Pod Identity to the ``neuron-observer-otel`` service account. Skip this step if you use only Prometheus.

.. code-block:: bash

   # Install the EKS Pod Identity Agent add-on
   aws eks create-addon --cluster-name <your-cluster-name> --region <REGION> \
     --addon-name eks-pod-identity-agent
   aws eks wait addon-active --cluster-name <your-cluster-name> --region <REGION> \
     --addon-name eks-pod-identity-agent

   # Create the IAM role the collector assumes via Pod Identity.
   # The trust policy allows pods.eks.amazonaws.com to assume it.
   cat > /tmp/trust.json <<'EOF'
   {
     "Version": "2012-10-17",
     "Statement": [{
       "Effect": "Allow",
       "Principal": {"Service": "pods.eks.amazonaws.com"},
       "Action": ["sts:AssumeRole", "sts:TagSession"]
     }]
   }
   EOF
   aws iam create-role --role-name neuron-observer-pod-identity \
     --assume-role-policy-document file:///tmp/trust.json

Attach permissions for the pipeline you use. For AMP, grant ``aps:RemoteWrite`` on the AMP workspace ARN. For CloudWatch, grant ``cloudwatch:PutMetricData`` (optionally scoped to the ``NeuronObserver`` namespace) plus CloudWatch Logs permissions.

.. code-block:: bash

   # AMP permissions example
   cat > /tmp/amp-policy.json <<EOF
   {
     "Version": "2012-10-17",
     "Statement": [{
       "Effect": "Allow",
       "Action": ["aps:RemoteWrite"],
       "Resource": "<AMP_WORKSPACE_ARN>"
     }]
   }
   EOF
   aws iam put-role-policy --role-name neuron-observer-pod-identity \
     --policy-name neuron-observer-amp-permissions \
     --policy-document file:///tmp/amp-policy.json

.. code-block:: bash

   # CloudWatch permissions example
   cat > /tmp/policy.json <<EOF
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
         "Resource": "arn:aws:logs:<REGION>:<ACCOUNT>:log-group:/metrics/neuron-observer*"
       },
       {
         "Effect": "Allow",
         "Action": ["cloudwatch:PutMetricData"],
         "Resource": "*",
         "Condition": {"StringLike": {"cloudwatch:namespace": "NeuronObserver*"}}
       }
     ]
   }
   EOF
   aws iam put-role-policy --role-name neuron-observer-pod-identity \
     --policy-name neuron-observer-cloudwatch-permissions \
     --policy-document file:///tmp/policy.json

The collector uses this role through the ``neuron-observer-otel`` service account.

**4:** Install Neuron Observer with your export path

Every export path installs from the same Neuron Helm chart, and every install needs ``--set observer.enabled=true``. The chart also packages other Neuron components: the device plugin (``devicePlugin``), node problem detector (``npd``), scheduler extension (``scheduler``), DRA driver (``draDriver``), and UltraServer operator (``ultraserverOperator``). The device plugin and node problem detector are enabled by default.

The commands in this guide install Neuron Observer only, so they turn the other components off. This doesn't affect components you've already installed in another Helm release. To install one of them in this release, remove its flag from ``COMMON``.

For a full list of the ``observer.*`` chart values and their defaults, see :ref:`neuron-observer-configuration-reference`.

Common flags for every install:

.. code-block:: bash

   CHART=oci://public.ecr.aws/neuron/neuron-helm-chart
   CHART_VERSION=1.10.0
   # Turn off the chart's other Neuron components; this release installs
   # Neuron Observer only.
   COMMON="--set draDriver.enabled=false \
           --set devicePlugin.enabled=false \
           --set scheduler.enabled=false \
           --set ultraserverOperator.enabled=false \
           --set npd.enabled=false"

Run the following ``helm upgrade --install`` command for your export path. Prometheus scrapes the daemon directly and does not require an OpenTelemetry collector. AMP and CloudWatch use the daemon collector.

**Option A: Prometheus (in-cluster scrape)**

.. code-block:: bash

   helm upgrade --install neuron-observer $CHART \
     --version $CHART_VERSION \
     --namespace neuron-observer --create-namespace \
     $COMMON \
     --set observer.enabled=true \
     --set observer.monitoring.releaseLabel=prometheus \
     --set observer.monitoring.grafanaDashboards.enabled=true

This example creates the dashboard ``ConfigMap`` resources in the ``monitoring`` namespace. The ``kube-prometheus-stack`` Grafana sidecar container discovers them. If your sidecar container watches another namespace or label, set the corresponding ``observer.monitoring.grafanaDashboards.*`` values.

**Option B: Amazon Managed Service for Prometheus (AMP)**

Requires the Pod Identity or IRSA permission from step 3 for ``neuron-observer-otel`` in the ``neuron-observer`` namespace.

.. code-block:: bash

   helm upgrade --install neuron-observer $CHART \
     --version $CHART_VERSION \
     --namespace neuron-observer --create-namespace \
     $COMMON \
     --set observer.enabled=true \
     --set observer.daemon.config.export.enabled=true \
     --set observer.otelDaemonCollector.pipelines.amp.enabled=true \
     --set observer.otelDaemonCollector.pipelines.amp.endpoint=https://aps-workspaces.<REGION>.amazonaws.com/workspaces/<WORKSPACE_ID>/api/v1/remote_write \
     --set observer.otelDaemonCollector.awsRegion=<REGION>

**Option C: Amazon CloudWatch**

Requires the Pod Identity or IRSA permissions from step 3 for ``neuron-observer-otel`` in the ``neuron-observer`` namespace.

.. code-block:: bash

   helm upgrade --install neuron-observer $CHART \
     --version $CHART_VERSION \
     --namespace neuron-observer --create-namespace \
     $COMMON \
     --set observer.enabled=true \
     --set observer.daemon.config.export.enabled=true \
     --set observer.otelDaemonCollector.pipelines.cloudwatch.enabled=true \
     --set observer.otelDaemonCollector.awsRegion=<REGION>

For Option B or Option C, bind the collector role created in step 3 to the service account:

.. code-block:: bash

   aws eks create-pod-identity-association \
     --cluster-name <your-cluster-name> --region <REGION> \
     --namespace neuron-observer --service-account neuron-observer-otel \
     --role-arn <collector-role-arn>

For CloudWatch, verify that CloudWatch receives data:

.. code-block:: bash

   # Verify — one entry per node, keyed by instance_id
   aws cloudwatch list-metrics --region <REGION> --namespace NeuronObserver \
     --metric-name neuron.system.neuron_device_count

**5:** Verify the installation

To confirm the installation succeeded, check that pods are running on each Neuron node:

.. code-block:: bash

   kubectl get pods -n neuron-observer -o wide

Expected output: one ``neuron-observer`` pod per Neuron node in ``Running`` state. The AMP and CloudWatch options also create one collector pod per Neuron node.

Verify the daemon is healthy:

.. code-block:: bash

   kubectl exec -n neuron-observer <observer-pod> -- nobs status
   # Expected output:
   # version        <installed version>
   # live           true
   # ready          true
   # neuron driver  <installed driver version>
   # efa driver     <installed EFA version, when available>

Verify that the daemon produces metrics at its native Prometheus endpoint:

.. code-block:: bash

   kubectl port-forward -n neuron-observer <observer-pod> 7001:7001

In another terminal:

.. code-block:: bash

   curl -s http://localhost:7001/metrics | head -30

The response contains Prometheus exposition-format output with metrics prefixed with ``neuron_``.

Configure Prometheus scraping
-----------------------------

When the Prometheus Operator CRD is installed, Option A creates a ``PodMonitor`` that points Prometheus at the daemon's native ``/metrics`` endpoint. With the ``kube-prometheus-stack`` release name used in this guide, no extra scrape configuration is required.

If your Prometheus installation does not use the Prometheus Operator or does not select this ``PodMonitor``, add a scrape job:

.. code-block:: yaml

   - job_name: neuron-observer
     kubernetes_sd_configs:
       - role: pod
         namespaces:
           names:
             - neuron-observer
     relabel_configs:
       - source_labels: [__meta_kubernetes_pod_label_app_kubernetes_io_name]
         regex: neuron-observer
         action: keep
       - source_labels: [__meta_kubernetes_pod_ip]
         regex: (.+)
         replacement: ${1}:7001
         target_label: __address__
       - source_labels: [__meta_kubernetes_pod_node_name]
         target_label: node
     metrics_path: /metrics
     scrape_interval: 30s

With ``kube-prometheus-stack``, apply the job as an additional scrape configuration. The file must be a bare YAML list (starting with ``- job_name:``), not wrapped in a ``scrape_configs:`` key.

.. code-block:: bash

   helm upgrade prometheus prometheus-community/kube-prometheus-stack \
     --namespace monitoring --reuse-values \
     --set-file prometheus.prometheusSpec.additionalScrapeConfigs=neuron-observer-scrape.yaml

Verify that Prometheus scrapes the targets about 30 seconds after you apply the configuration:

.. code-block:: bash

   kubectl port-forward -n monitoring \
     svc/prometheus-kube-prometheus-prometheus 9090:9090

In another terminal:

.. code-block:: bash

   curl -s "http://localhost:9090/api/v1/query?query=neuron_system_neuron_device_count"

The response contains ``"status":"success"`` and results with your node's ``instance_id``.

Uninstall Neuron Observer
-------------------------

Remove Neuron Observer:

.. code-block:: bash

   helm uninstall neuron-observer --namespace neuron-observer
   kubectl delete namespace neuron-observer

If you installed the monitoring stack only for Neuron Observer and no longer need it elsewhere, remove it:

.. code-block:: bash

   helm uninstall prometheus -n monitoring

If you installed the ADOT add-on and cert-manager only for the AMP or CloudWatch option and no other workloads use them, remove them:

.. code-block:: bash

   aws eks delete-addon --cluster-name <your-cluster-name> --addon-name adot --region <REGION>
   kubectl delete -f https://github.com/cert-manager/cert-manager/releases/download/v1.17.0/cert-manager.yaml

Related information
-------------------

- :doc:`/tools/neuron-observer/get-started` — a quickstart from installation to first metrics.
- :doc:`/tools/neuron-observer/configuration-reference` — the Helm chart configuration values, their types, and defaults.
- :doc:`/tools/neuron-observer/metrics-fault-reference` — the metrics and fault events Neuron Observer reports.
- :doc:`/tools/neuron-observer/troubleshooting-faq` — common issues, FAQs, and beta limitations.
