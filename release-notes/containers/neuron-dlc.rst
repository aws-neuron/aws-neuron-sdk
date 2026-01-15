.. _neuron-dlc-release-notes:

Neuron DLC Release Notes
===============================

.. contents:: Table of contents
   :local:
   :depth: 1

.. note:: 
  For Neuron DLC release notes on Neuron 2.25.0 up to the current release, see :doc:`/release-notes/prev/by-component/containers`.

----

Known Issues
------------

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

Neuron 2.26.0
--------------
Date: 6/24/2025

- pytorch-training-neuronx 2.7.0 DLC has two HIGH CVEs related to `sagemaker-python-sdk` package. We are actively working to resolve these high CVEs:
- * `CVE-2024-34072 <https://nvd.nist.gov/vuln/detail/CVE-2024-34072>`_
- * `CVE-2024-34073 <https://nvd.nist.gov/vuln/detail/CVE-2024-34073>`_


Neuron 2.24.0
-------------
Date: 06/24/2025

- Added new pytorch-inference-vllm-neuronx 0.7.2 DLC that contains all dependencies including drivers, tools, NxDI and other packages to run vLLM out of the box
- Upgraded pytorch-training-neuronx DLC to 2.7 version along with its related dependencies
- Upgraded pytorch-inference-neuronx DLC to 2.7 version along with its related dependencies
- Upgraded jax-training-neuronx DLC to 0.6 version along with its related dependencies
- Updated Neuron SDK to latest 2.24.0 release for all Neuron DLCs


Neuron 2.23.0
-------------
Date: 05/19/2025

- Upgraded pytorch-training-neuronx DLC to 2.6 version along with its related dependencies
- Upgraded pytorch-inference-neuronx DLC to 2.6 version along with its related dependencies
- Updated Neuron SDK to latest 2.23.0 release for all Neuron DLCs


Neuron 2.22.0
-------------
Date: 04/04/2025

- Upgraded jax-training-neuronx DLC to 0.5 version
- Updated Neuron SDK to latest 2.22.0 release for all Neuron DLCs
- Restructure all Dockerfiles by combining RUN commands for faster build time


Neuron 2.21.1
-------------
Date: 01/14/2025

- Minor improvements and bug fixes.


Neuron 2.21.0
-------------
Date: 12/19/2024

- Added new jax-training-neuronx 0.4 Training DLC that contains all dependencies including drivers, tools and other packages to run JAX out of the box.
- Added new pytorch-inference-neuronx 2.5.1 and pytorch-training-neuronx 2.5.1 DLCs
- PyTorch 1.13.1 and 2.1.2 DLCs reached end of support phase, We now recommend customers to use PyTorch 2.5.1 DLCs by default.
- All Neuron supported DLCs to use latest Neuron SDK 2.21.0 version.
- All Neuron supported DLCs are now updated to Ubuntu 22. Here is the list:
  
   * pytorch-inference-neuron 2.5.1 with Ubuntu 22
   * pytorch-training-neuron 2.5.1 with Ubuntu 22
   * jax-training-neuronx 0.4 with Ubuntu 22
  
- pytorch-inference-neuronx now supports both NxD Inference and Transformers NeuronX libraries for inference.


Neuron 2.20.2
-------------
Date: 11/20/2024

- Neuron 2.20.2 DLC fixes dependency bug for NxDT use case by pinning the correct torch version. 


Neuron 2.20.1
-------------

Date: 10/25/2024

- Neuron 2.20.1 DLC includes prerequisites for :ref:`nxdt_installation_guide`. Customers can expect to use NxDT out of the box.


Neuron 2.20.0
-------------

Date: 09/16/2024

- Updated Neuron SDK to latest 2.20.0 release for PyTorch Neuron DLCs.
- Added new NxD Training package to `pytorch-training-neuronx DLCs <https://github.com/aws-neuron/deep-learning-containers/tree/main?tab=readme-ov-file#pytorch-training-neuronx>`_.
- See `here <https://github.com/aws-neuron/deep-learning-containers/tree/2.20.0>`__ for the new DLC details.


Neuron 2.19.0
-------------

Date: 07/03/2024

- Updated Neuron SDK to latest 2.19.0 release for PyTorch Neuron DLCs.
- Updated TorchServe to 0.11.0 for PyTorch Neuron DLCs.
- See `here <https://github.com/aws-neuron/deep-learning-containers/tree/2.19.0>`__ for the new DLC details.
