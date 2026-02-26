.. meta::
    :description: Complete release notes for the Neuron Containers component across all AWS Neuron SDK versions.
    :keywords: neuron containers, dlc, kubernetes, k8s, release notes, aws neuron sdk
    :date-modified: 02/26/2026

.. _containers_rn:

Component Release Notes for Neuron Containers
==============================================

The release notes for the Neuron Containers component. Read them for the details about the changes, improvements, and bug fixes for all release versions of the AWS Neuron SDK.

.. _containers-2-28-0-rn:   

Neuron Containers [2.28.0] (Neuron 2.28.0 Release)
--------------------------------------------------------------------------------------

Date of Release: 02/26/2026

Improvements
~~~~~~~~~~~~~~~

**Kubernetes Support**

* Introduced the Neuron DRA Driver, which enables advanced resource allocation capabilities using the Kubernetes Dynamic Resource Allocation (DRA) API for more flexible and efficient management of Neuron devices. For more details, see :doc:`/containers/neuron-dra`.
* Added Neuron DRA Driver support to the Neuron Helm Charts. For more details, see :doc:`the updated Helm documentation under the Kubernetes Getting Started page </containers/kubernetes-getting-started>`.

.. _containers-2-27-0-rn:

Neuron Containers [2.27.0] (Neuron 2.27.0 Release)
---------------------------------------------------

Date of Release: 12/19/2025

Improvements
~~~~~~~~~~~~~~~

**DLC Support**

* Added new pytorch-inference-vllm-neuronx 0.11.0 DLC with PyTorch 2.8, vLLM V1 with the vLLM-Neuron Plugin, tools, NxDI and all dependencies to run vLLM out of the box.
* Upgraded pytorch-training-neuronx and pytorch-inference-neuronx DLCs to PyTorch 2.9.0 with related dependencies.
* Upgraded jax-training-neuronx DLC to JAX 0.7.0 with related dependencies.
* Upgraded base image to Ubuntu 24.04 and Python 3.12 in all DLCs.
* Upgraded all Neuron packages and dependencies to support AWS Neuron SDK version 2.27.

Known Issues
~~~~~~~~~~~~

**Note**: Common Vulnerability and Exposure (CVE) identifiers are assigned to publicly disclosed cybersecurity vulnerabilities. CVE identifiers help security professionals and software vendors coordinate their efforts to address and mitigate vulnerabilities.

* ``pytorch-training-neuronx``: 0.9.0 DLC has multiple CRITICAL and HIGH CVEs. We are actively working to resolve them.
   * `CVE-2021-44906 <https://nvd.nist.gov/vuln/detail/CVE-2021-44906>`_ - Prototype Pollution vulnerability in minimist package
   * `CVE-2023-38039 <https://nvd.nist.gov/vuln/detail/CVE-2023-38039>`_ - Memory exhaustion vulnerability in curl/libcurl from unlimited header processing
   * `CVE-2021-35517 <https://nvd.nist.gov/vuln/detail/CVE-2021-35517>`_ - Denial of service vulnerability in Apache Commons Compress TAR archive processing
   * `CVE-2022-29217 <https://nvd.nist.gov/vuln/detail/CVE-2022-29217>`_ - JWT signing algorithm confusion vulnerability in PyJWT library
   * `CVE-2025-58056 <https://nvd.nist.gov/vuln/detail/CVE-2025-58056>`_ - HTTP request smuggling vulnerability in Netty codec
   * `CVE-2024-45337 <https://nvd.nist.gov/vuln/detail/CVE-2024-45337>`_ - Authorization bypass vulnerability in golang.org/x/crypto SSH implementation
   * `CVE-2024-56201 <https://nvd.nist.gov/vuln/detail/CVE-2024-56201>`_ - Remote code execution vulnerability in Jinja templating engine
   * `CVE-2025-0725 <https://nvd.nist.gov/vuln/detail/CVE-2025-0725>`_ - Buffer overflow vulnerability in curl/libcurl gzip decompression
   * `CVE-2023-36665 <https://nvd.nist.gov/vuln/detail/CVE-2023-36665>`_ - Prototype Pollution vulnerability in protobufjs library
   * `CVE-2023-45288 <https://nvd.nist.gov/vuln/detail/CVE-2023-45288>`_ - HTTP/2 CONTINUATION frame DoS vulnerability in golang.org/x/net
   * `CVE-2021-33194 <https://nvd.nist.gov/vuln/detail/CVE-2021-33194>`_ - Infinite loop vulnerability in golang.org/x/net ParseFragment
   * `CVE-2023-41419 <https://nvd.nist.gov/vuln/detail/CVE-2023-41419>`_ - Privilege escalation vulnerability in gevent WSGIServer
   * `CVE-2021-35516 <https://nvd.nist.gov/vuln/detail/CVE-2021-35516>`_ - Memory exhaustion vulnerability in Apache Commons Compress 7Z processing
   * `CVE-2022-24771 <https://nvd.nist.gov/vuln/detail/CVE-2022-24771>`_ - RSA signature verification vulnerability in node-forge
   * `CVE-2022-41723 <https://nvd.nist.gov/vuln/detail/CVE-2022-41723>`_ - HTTP/2 HPACK decoder DoS vulnerability in golang.org/x/net
   * `CVE-2025-66031 <https://nvd.nist.gov/vuln/detail/CVE-2025-66031>`_ - Uncontrolled recursion DoS vulnerability in node-forge ASN.1 parsing
   * `CVE-2025-58057 <https://nvd.nist.gov/vuln/detail/CVE-2025-58057>`_ - Memory exhaustion vulnerability in Netty BrotliDecoder
   * `CVE-2023-50782 <https://nvd.nist.gov/vuln/detail/CVE-2023-50782>`_ - TLS RSA key exchange vulnerability in python-cryptography
   * `CVE-2022-24772 <https://nvd.nist.gov/vuln/detail/CVE-2022-24772>`_ - RSA signature verification vulnerability in node-forge DigestInfo
   * `CVE-2022-27664 <https://nvd.nist.gov/vuln/detail/CVE-2022-27664>`_ - HTTP/2 connection hang DoS vulnerability in golang.org/x/net
   * `CVE-2024-56326 <https://nvd.nist.gov/vuln/detail/CVE-2024-56326>`_ - Sandbox bypass vulnerability in Jinja str.format detection
   * `CVE-2024-3651 <https://nvd.nist.gov/vuln/detail/CVE-2024-3651>`_ - Quadratic complexity DoS vulnerability in idna.encode() function
   * `CVE-2023-49083 <https://nvd.nist.gov/vuln/detail/CVE-2023-49083>`_ - NULL-pointer dereference vulnerability in cryptography PKCS7 processing
   * `CVE-2024-22189 <https://nvd.nist.gov/vuln/detail/CVE-2024-22189>`_ - Memory exhaustion vulnerability in quic-go NEW_CONNECTION_ID frames
   * `CVE-2025-47273 <https://nvd.nist.gov/vuln/detail/CVE-2025-47273>`_ - Path traversal vulnerability in setuptools PackageIndex
   * `CVE-2025-66418 <https://nvd.nist.gov/vuln/detail/CVE-2025-66418>`_ - Unbounded decompression chain vulnerability in urllib3
   * `CVE-2021-23337 <https://nvd.nist.gov/vuln/detail/CVE-2021-23337>`_ - Command injection vulnerability in lodash template function
   * `CVE-2023-29824 <https://nvd.nist.gov/vuln/detail/CVE-2023-29824>`_ - Use-after-free vulnerability in SciPy Py_FindObjects() function
   * `CVE-2025-12816 <https://nvd.nist.gov/vuln/detail/CVE-2025-12816>`_ - ASN.1 schema validation bypass vulnerability in node-forge
   * `CVE-2025-22869 <https://nvd.nist.gov/vuln/detail/CVE-2025-22869>`_ - SSH file transfer DoS vulnerability in golang.org/x/crypto
   * `CVE-2025-59530 <https://nvd.nist.gov/vuln/detail/CVE-2025-59530>`_ - HANDSHAKE_DONE frame DoS vulnerability in quic-go
   * `CVE-2024-6345 <https://nvd.nist.gov/vuln/detail/CVE-2024-6345>`_ - Remote code execution vulnerability in setuptools package_index
   * `CVE-2023-27533 <https://nvd.nist.gov/vuln/detail/CVE-2023-27533>`_ - TELNET protocol input validation vulnerability in curl/libcurl
   * `CVE-2021-36090 <https://nvd.nist.gov/vuln/detail/CVE-2021-36090>`_ - Memory exhaustion vulnerability in Apache Commons Compress ZIP processing
   * `CVE-2025-66471 <https://nvd.nist.gov/vuln/detail/CVE-2025-66471>`_ - Highly compressed data handling vulnerability in urllib3 Streaming API
   * `CVE-2023-43804 <https://nvd.nist.gov/vuln/detail/CVE-2023-43804>`_ - Cookie header information leak vulnerability in urllib3 redirects
   * `CVE-2022-25878 <https://nvd.nist.gov/vuln/detail/CVE-2022-25878>`_ - Prototype Pollution vulnerability in protobufjs util.setProperty
   * `CVE-2021-35515 <https://nvd.nist.gov/vuln/detail/CVE-2021-35515>`_ - Infinite loop vulnerability in Apache Commons Compress 7Z codec construction
   * `CVE-2021-38561 <https://nvd.nist.gov/vuln/detail/CVE-2021-38561>`_ - Out-of-bounds read vulnerability in golang.org/x/text BCP 47 parsing
   * `CVE-2022-43551 <https://nvd.nist.gov/vuln/detail/CVE-2022-43551>`_ - HSTS bypass vulnerability in curl/libcurl IDN handling
   * `CVE-2022-27191 <https://nvd.nist.gov/vuln/detail/CVE-2022-27191>`_ - SSH server crash vulnerability in golang.org/x/crypto AddHostKey
   * GHSA-m425-mq94-257g - HTTP/2 concurrent stream limit bypass vulnerability in gRPC-Go
   * `CVE-2023-39325 <https://nvd.nist.gov/vuln/detail/CVE-2023-39325>`_ - HTTP/2 request reset DoS vulnerability in golang.org/x/net
   * `CVE-2024-2398 <https://nvd.nist.gov/vuln/detail/CVE-2024-2398>`_ - Memory leak vulnerability in curl/libcurl HTTP/2 server push
   * `CVE-2023-44487 <https://nvd.nist.gov/vuln/detail/CVE-2023-44487>`_ - HTTP/2 Rapid Reset DoS vulnerability in multiple packages
   * `CVE-2025-55163 <https://nvd.nist.gov/vuln/detail/CVE-2025-55163>`_ - MadeYouReset DDoS vulnerability in Netty HTTP/2 implementation
   * `CVE-2023-27534 <https://nvd.nist.gov/vuln/detail/CVE-2023-27534>`_ - SFTP path traversal vulnerability in curl/libcurl tilde handling
   * `CVE-2022-32149 <https://nvd.nist.gov/vuln/detail/CVE-2022-32149>`_ - Accept-Language header DoS vulnerability in golang.org/x/text
   * `CVE-2025-47913 <https://nvd.nist.gov/vuln/detail/CVE-2025-47913>`_ - SSH agent panic vulnerability in golang.org/x/crypto
   * `CVE-2022-40898 <https://nvd.nist.gov/vuln/detail/CVE-2022-40898>`_ - DoS vulnerability in Python wheel CLI
   * `CVE-2023-23914 <https://nvd.nist.gov/vuln/detail/CVE-2023-23914>`_ - HSTS functionality failure vulnerability in curl/libcurl
   * `CVE-2023-0286 <https://nvd.nist.gov/vuln/detail/CVE-2023-0286>`_ - X.400 address processing vulnerability in cryptography
   * `CVE-2022-25647 <https://nvd.nist.gov/vuln/detail/CVE-2022-25647>`_ - Deserialization vulnerability in Gson writeReplace() method
   * `CVE-2021-43565 <https://nvd.nist.gov/vuln/detail/CVE-2021-43565>`_ - SSH server panic vulnerability in golang.org/x/crypto
   * `CVE-2024-7254 <https://nvd.nist.gov/vuln/detail/CVE-2024-7254>`_ - Stack overflow vulnerability in Protocol Buffers nested groups parsing
   * `CVE-2023-2976 <https://nvd.nist.gov/vuln/detail/CVE-2023-2976>`_ - Temporary directory access vulnerability in Google Guava FileBackedOutputStream
   * `CVE-2026-21441 <https://nvd.nist.gov/vuln/detail/CVE-2026-21441>`_ - Decompression bomb vulnerability in urllib3 HTTP redirect responses
   * `CVE-2023-38545 <https://nvd.nist.gov/vuln/detail/CVE-2023-38545>`_ - Heap buffer overflow vulnerability in curl/libcurl SOCKS5 proxy handshake
   * GHSA-xpw8-rcwv-8f8p - HTTP/2 RST frame DoS vulnerability in Netty
   * `CVE-2022-42920 <https://nvd.nist.gov/vuln/detail/CVE-2022-42920>`_ - Arbitrary bytecode generation vulnerability in Apache Commons BCEL
   * `CVE-2024-24786 <https://nvd.nist.gov/vuln/detail/CVE-2024-24786>`_ - Infinite loop vulnerability in google.golang.org/protobuf JSON unmarshaling
 
* ``pytorch-inference-vllm-neuronx``: 0.11.0 DLC has multiple HIGH CVEs. We are actively working to resolve these high CVEs:
   * `CVE-2026-21441 <https://nvd.nist.gov/vuln/detail/CVE-2026-21441>`_ - Decompression bomb vulnerability in urllib3 HTTP redirect responses
   * `CVE-2025-62164 <https://nvd.nist.gov/vuln/detail/CVE-2025-62164>`_ - Memory corruption vulnerability in vLLM Completions API endpoint
   * `CVE-2025-69223 <https://nvd.nist.gov/vuln/detail/CVE-2025-69223>`_ - Zip bomb DoS vulnerability in AIOHTTP server
   * GHSA-mcmc-2m55-j8jj - Insufficient fix for CVE-2025-62164 in vLLM sparse tensor validation
   * `CVE-2025-66448 <https://nvd.nist.gov/vuln/detail/CVE-2025-66448>`_ - Remote code execution vulnerability in vLLM config class auto_map
   * `CVE-2025-66418 <https://nvd.nist.gov/vuln/detail/CVE-2025-66418>`_ - Unbounded decompression chain vulnerability in urllib3
   * `CVE-2025-66471 <https://nvd.nist.gov/vuln/detail/CVE-2025-66471>`_ - Highly compressed data handling vulnerability in urllib3 Streaming API

* ``pytorch-training-neuronx``: 0.9.0 DLC has multiple HIGH CVEs. We are actively working to resolve these high CVEs:
   * `CVE-2025-66418 <https://nvd.nist.gov/vuln/detail/CVE-2025-66418>`_ - Unbounded decompression chain vulnerability in urllib3
   * `CVE-2025-66471 <https://nvd.nist.gov/vuln/detail/CVE-2025-66471>`_ - Highly compressed data handling vulnerability in urllib3 Streaming API
   * `CVE-2026-21441 <https://nvd.nist.gov/vuln/detail/CVE-2026-21441>`_ - Decompression bomb vulnerability in urllib3 HTTP redirect responses


----

.. _containers-2-26-0-rn:

Neuron Containers [2.26.0] (Neuron 2.26.0 Release)
---------------------------------------------------

Date of Release: 09/18/2025

Improvements
~~~~~~~~~~~~~~~

**DLC Support**

* Both pytorch-training-neuronx and pytorch-inference-neuronx DLCs have been upgraded to version 2.8.0 along with their related dependencies.
* Upgraded Python version to 3.11 in all Deep Learning Containers.
* All Neuron packages and their dependencies have been upgraded to support version 2.26.0 of the AWS Neuron SDK.

Breaking Changes
~~~~~~~~~~~~~~~~

* End-of-support for the Transformers NeuronX library starts with the 2.26.0 release of the AWS Neuron SDK. With this support ended, the PyTorch inference Deep Learning Container (DLC) will no longer include the transformers-neuronx package.

Known Issues
~~~~~~~~~~~~

* ``pytorch-training-neuronx`` 2.7.0 DLC has two HIGH CVEs related to ``sagemaker-python-sdk`` package. We are actively working to resolve these high CVEs:
  * `CVE-2024-34072 <https://nvd.nist.gov/vuln/detail/CVE-2024-34072>`_ - Vulnerability in sagemaker-python-sdk package
  * `CVE-2024-34073 <https://nvd.nist.gov/vuln/detail/CVE-2024-34073>`_ - Vulnerability in sagemaker-python-sdk package


----

.. _containers-2-25-0-rn:

Neuron Containers [2.25.0] (Neuron 2.25.0 Release)
---------------------------------------------------

Date of Release: 07/31/2025

Improvements
~~~~~~~~~~~~~~~

**DLC Support**

* All Neuron packages and their dependencies have been upgraded to support AWS Neuron SDK version 2.25.0.
* The pytorch-inference-vllm-neuronx Deep Learning Container has been upgraded to version 0.9.1.

Known Issues
~~~~~~~~~~~~

* ``pytorch-training-neuronx`` 2.7.0 DLC has two HIGH CVEs related to ``sagemaker-python-sdk`` package. We are actively working to resolve these high CVEs:
  * `CVE-2024-34072 <https://nvd.nist.gov/vuln/detail/CVE-2024-34072>`_ - Vulnerability in sagemaker-python-sdk package
  * `CVE-2024-34073 <https://nvd.nist.gov/vuln/detail/CVE-2024-34073>`_ - Vulnerability in sagemaker-python-sdk package
* ``pytorch-inference-vllm-neuronx`` 0.9.1 DLC has CRITICAL and HIGH CVEs. We are actively working to resolve them.


----

.. _containers-2-24-0-rn:

Neuron Containers [2.24.0] (Neuron 2.24.0 Release)
---------------------------------------------------

Date of Release: 06/24/2025

Improvements
~~~~~~~~~~~~~~~

**DLC Support**

* Added new pytorch-inference-vllm-neuronx 0.7.2 DLC that contains all dependencies including drivers, tools, NxDI and other packages to run vLLM out of the box.
* Upgraded pytorch-training-neuronx DLC to 2.7 version along with its related dependencies.
* Upgraded pytorch-inference-neuronx DLC to 2.7 version along with its related dependencies.
* Upgraded jax-training-neuronx DLC to 0.6 version along with its related dependencies.
* Updated Neuron SDK to latest 2.24.0 release for all Neuron DLCs.


----

.. _containers-2-23-0-rn:

Neuron Containers [2.23.0] (Neuron 2.23.0 Release)
---------------------------------------------------

Date of Release: 05/19/2025

Improvements
~~~~~~~~~~~~~~~

**DLC Support**

* Upgraded pytorch-training-neuronx DLC to 2.6 version along with its related dependencies.
* Upgraded pytorch-inference-neuronx DLC to 2.6 version along with its related dependencies.
* Updated Neuron SDK to latest 2.23.0 release for all Neuron DLCs.


----

.. _containers-2-22-0-rn:

Neuron Containers [2.22.0] (Neuron 2.22.0 Release)
---------------------------------------------------

Date of Release: 04/04/2025

Improvements
~~~~~~~~~~~~~~~

**DLC Support**

* Upgraded jax-training-neuronx DLC to 0.5 version.
* Updated Neuron SDK to latest 2.22.0 release for all Neuron DLCs.
* Restructure all Dockerfiles by combining RUN commands for faster build time.

**Kubernetes Support**

* This release introduces the Neuron Helm Chart, which helps streamline the deployment of AWS Neuron components on Amazon EKS.
* Adds ECS support for the "Neuron Node Problem Detector and Recovery" artifact.
* Improves scalability and performance of the Neuron Device Plugin and Neuron Scheduler Extension by skipping "list" calls from the device plugin to the scheduler in situations where the pod allocation request either needs one or all the available resources in the node.
* Ends support for resource name 'neurondevice' with the Neuron Device Plugin.

Breaking Changes
~~~~~~~~~~~~~~~~

* Ends support for resource name 'neurondevice' with the Neuron Device Plugin.


----

.. _containers-2-21-1-rn:

Neuron Containers [2.21.1] (Neuron 2.21.1 Release)
---------------------------------------------------

Date of Release: 01/14/2025

Improvements
~~~~~~~~~~~~~~~

**DLC Support**

* Minor improvements and bug fixes.

Bug Fixes
~~~~~~~~~

* Minor improvements and bug fixes.


----

.. _containers-2-21-0-rn:

Neuron Containers [2.21.0] (Neuron 2.21.0 Release)
---------------------------------------------------

Date of Release: 12/19/2024

Improvements
~~~~~~~~~~~~~~~

**DLC Support**

* Added new jax-training-neuronx 0.4 Training DLC that contains all dependencies including drivers, tools and other packages to run JAX out of the box.
* Added new pytorch-inference-neuronx 2.5.1 and pytorch-training-neuronx 2.5.1 DLCs.
* PyTorch 1.13.1 and 2.1.2 DLCs reached end of support phase, We now recommend customers to use PyTorch 2.5.1 DLCs by default.
* All Neuron supported DLCs to use latest Neuron SDK 2.21.0 version.
* All Neuron supported DLCs are now updated to Ubuntu 22.
* pytorch-inference-neuronx now supports both NxD Inference and Transformers NeuronX libraries for inference.

Breaking Changes
~~~~~~~~~~~~~~~~

* PyTorch 1.13.1 and 2.1.2 DLCs reached end of support phase.


----

.. _containers-2-20-2-rn:

Neuron Containers [2.20.2] (Neuron 2.20.2 Release)
---------------------------------------------------

Date of Release: 11/20/2024

Improvements
~~~~~~~~~~~~~~~

**DLC Support**

* Neuron 2.20.2 DLC fixes dependency bug for NxDT use case by pinning the correct torch version.

**Kubernetes Support**

* This release addresses a stability issue in the Neuron Scheduler Extension that previously caused crashes shortly after installation.

Bug Fixes
~~~~~~~~~

* Fixed dependency bug for NxDT use case by pinning the correct torch version.
* Addressed stability issue in the Neuron Scheduler Extension that previously caused crashes shortly after installation.


----

.. _containers-2-20-1-rn:

Neuron Containers [2.20.1] (Neuron 2.20.1 Release)
---------------------------------------------------

Date of Release: 10/25/2024

Improvements
~~~~~~~~~~~~~~~

**DLC Support**

* Neuron 2.20.1 DLC includes prerequisites for NxDT installation. Customers can expect to use NxDT out of the box.


----

.. _containers-2-20-0-rn:

Neuron Containers [2.20.0] (Neuron 2.20.0 Release)
---------------------------------------------------

Date of Release: 09/16/2024

Improvements
~~~~~~~~~~~~~~~

**DLC Support**

* Updated Neuron SDK to latest 2.20.0 release for PyTorch Neuron DLCs.
* Added new NxD Training package to pytorch-training-neuronx DLCs.


----

.. _containers-2-19-0-rn:

Neuron Containers [2.19.0] (Neuron 2.19.0 Release)
---------------------------------------------------

Date of Release: 07/03/2024

Improvements
~~~~~~~~~~~~~~~

**DLC Support**

* Updated Neuron SDK to latest 2.19.0 release for PyTorch Neuron DLCs.
* Updated TorchServe to 0.11.0 for PyTorch Neuron DLCs.

**Kubernetes Support**

* Critical Security Patch: Updated the dependencies used by the Neuron Device Plugin and the Neuron Kubernetes Scheduler to fix several important security vulnerabilities.
* This release introduces Neuron Node Problem Detector And Recovery artifact to enable fast error detection and recovery in Kubernetes environment. Current version supports EKS managed and self-managed node groups for all EKS supported Kubernetes versions.
* This release introduces a container image for neuron monitor to make it easy to run neuron monitor along with Prometheus and Grafana to monitor neuron metrics in Kubernetes environments.

Bug Fixes
~~~~~~~~~

* This release contains changes to improve performance of the device plugin at scale.


----

.. _containers-2-5-0-rn:

Neuron Containers [2.5.0] (Neuron 2.5.0 Release)
-------------------------------------------------

Date of Release: 11/07/2022

Improvements
~~~~~~~~~~~~~~~

**DLC Support**

* Neuron now supports trn1-based training in Sagemaker and Deep Learning Containers using PyTorch.

**Neuron Containers**

* Neuron now supports trn1-based training in Sagemaker and Deep Learning Containers using PyTorch.


----

.. _containers-2-4-0-rn:

Neuron Containers [2.4.0] (Neuron 2.4.0 Release)
-------------------------------------------------

Date of Release: 10/27/2022

Improvements
~~~~~~~~~~~~~~~

**Neuron Containers**

* Neuron now supports Kubernetes work scheduling at the level of NeuronCore. Updates on how to use the new core allocation method is captured in the Kubernetes documentation on this site.

**Kubernetes Support**

* Added support for NeuronCore based scheduling to the Neuron Kubernetes Scheduler. Learn more about how to use NeuronCores for finer grain control over container scheduling by following the K8 tutorials documentation.


----

.. _containers-2-3-0-rn:

Neuron Containers [2.3.0] (Neuron 2.3.0 Release)
-------------------------------------------------

Date of Release: 10/10/2022

Improvements
~~~~~~~~~~~~~~~

**Neuron Containers**

* Now supporting TRN1 and INF1 EC2 instance types as part of Neuron. There is an optional aws-neuronx-oci-hooks package users may install for convenience that supports use of the AWS_NEURON_VISIBLE_DEVICES environment variable when launching containers. New DLC containers will be coming soon in support of training workloads on TRN1.

**Kubernetes Support**

* Added support for TRN1 and INF1 EC2 instance types.


----

.. _containers-1-19-0-rn:

Neuron Containers [1.19.0] (Neuron 1.19.0 Release)
---------------------------------------------------

Date of Release: 04/29/2022

Improvements
~~~~~~~~~~~~~~~

**Neuron Containers**

* Neuron Kubernetes device driver plugin now can figure out communication with the Neuron driver without the oci hooks. Starting with Neuron 1.19.0 release, installing aws-neuron-runtime-base and oci-add-hooks are no longer a requirement for Neuron Kubernetes device driver plugin.

**Kubernetes Support**

* Minor updates.


----

.. _containers-2-16-0-rn:

Neuron Containers [2.16.0] (Neuron 2.16.0 Release)
---------------------------------------------------

Date of Release: 09/01/2023

Improvements
~~~~~~~~~~~~~~~

**Kubernetes Support**

* This release enables easier programmability by using 0-based indexing for Neuron Devices and NeuronCores in EKS container environments. Previously, the Neuron Device indexing was assigned randomly. This change requires Neuron Driver version 2.12.14 or newer.
* Improved logging when Neuron Driver not installed/present.

Bug Fixes
~~~~~~~~~

* Fixed Neuron Device Plugin crash when Neuron Driver is not installed/present on the host.
* Fixed issue where pods fail to deploy when multiple containers are requesting Neuron resources.
* Fixed issue where launching many pods each requesting Neuron cores fails to deploy.


----

.. _containers-2-1-0-rn:

Neuron Containers [2.1.0] (Neuron 2.1.0 Release)
-------------------------------------------------

Date of Release: 10/27/2022

Improvements
~~~~~~~~~~~~~~~

**Kubernetes Support**

* Added support for NeuronCore based scheduling to the Neuron Kubernetes Scheduler. Learn more about how to use NeuronCores for finer grain control over container scheduling by following the K8 tutorials documentation.


----

.. _containers-2-0-0-rn:

Neuron Containers [2.0.0] (Neuron 2.0.0 Release)
-------------------------------------------------

Date of Release: 10/10/2022

Improvements
~~~~~~~~~~~~~~~

**Kubernetes Support**

* Added support for TRN1 and INF1 EC2 instance types.


----

.. _containers-1-9-3-rn:

Neuron Containers [1.9.3] (Neuron 1.9.3 Release)
-------------------------------------------------

Date of Release: 08/02/2022

Improvements
~~~~~~~~~~~~~~~

**Kubernetes Support**

* Minor updates.


----

.. _containers-1-9-2-rn:

Neuron Containers [1.9.2] (Neuron 1.9.2 Release)
-------------------------------------------------

Date of Release: 05/27/2022

Improvements
~~~~~~~~~~~~~~~

**Kubernetes Support**

* Minor updates.


----

.. _containers-1-9-0-rn:

Neuron Containers [1.9.0] (Neuron 1.9.0 Release)
-------------------------------------------------

Date of Release: 04/29/2022

Improvements
~~~~~~~~~~~~~~~

**Kubernetes Support**

* Minor updates.


----

.. _containers-1-8-2-rn:

Neuron Containers [1.8.2] (Neuron 1.8.2 Release)
-------------------------------------------------

Date of Release: 03/25/2022

Improvements
~~~~~~~~~~~~~~~

**Kubernetes Support**

* Minor updates.


----

.. _containers-1-7-7-rn:

Neuron Containers [1.7.7] (Neuron 1.7.7 Release)
-------------------------------------------------

Date of Release: 01/20/2022

Improvements
~~~~~~~~~~~~~~~

**Kubernetes Support**

* Minor updates.


----

.. _containers-1-7-3-rn:

Neuron Containers [1.7.3] (Neuron 1.7.3 Release)
---------------------------------------------------

Date of Release: 10/27/2021

Improvements
~~~~~~~~~~~~~~~

**Neuron Containers**

* Starting with Neuron 1.16.0, use of Neuron ML Frameworks now comes with an integrated Neuron Runtime as a library, as a result it is no longer needed to deploy neuron-rtd.
* When using containers built with components from Neuron 1.16.0, or newer, please use aws-neuron-dkms version 2.1 or newer and the latest version of aws-neuron-runtime-base. Passing additional system capabilities is no longer required.

**Kubernetes Support**

* Minor updates.

Breaking Changes
~~~~~~~~~~~~~~~~

* Starting with Neuron 1.16.0, use of Neuron ML Frameworks now comes with an integrated Neuron Runtime as a library, as a result it is no longer needed to deploy neuron-rtd.

Known Issues
~~~~~~~~~~~~

* None reported for this release.