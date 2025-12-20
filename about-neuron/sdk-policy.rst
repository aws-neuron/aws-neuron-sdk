.. _sdk-maintenance-policy:

Neuron Software Maintenance policy
==================================

.. contents:: Table of Contents
   :local:
   :depth: 3

Overview
--------

This document outlines software maintenance policy for AWS Neuron
Software Development Kit (SDK), Neuron Components, both extension and
standalone components, supported model classes, features, APIs, DLAMIs
and DLCs, and dependency software. AWS Neuron is the SDK for Amazon EC2
`Inferentia <https://aws.amazon.com/machine-learning/inferentia/>`__ and
Amazon EC2
`Trainium <https://aws.amazon.com/machine-learning/trainium/>`__ based
instances purpose-built for deep learning. Neuron integrates with
popular Machine Learning (ML) frameworks like PyTorch, JAX, and
TensorFlow and includes a compiler, runtime, driver, profiling tools,
and libraries to support high performance training of generative AI
models on Trainium and Inferentia powered instances.

This document addresses Neuron Software life-cycle and the Neuron SDK
release versioning.

.. _neuron-software-definitions:

Neuron Software Definitions
---------------------------

Neuron Software refers to the complete set of software elements
provided by AWS Neuron, including:

Neuron SDK
~~~~~~~~~~

The core software development kit that enables users to build, train,
and deploy machine learning models on Inferentia and Trainium based
instances. The Neuron SDK encompasses the entire set of components,
features, APIs, and other elements that are bundled together and made
available in a particular version of the Neuron SDK release.

Neuron components
~~~~~~~~~~~~~~~~~

Neuron components refer to any packages or libraries within the Neuron
SDK that offer specific functionality. These components are typically
accessible through PIP, RPM, or Debian packages for easy installation
and usage. There are two main categories of Neuron components: Neuron
extension components and Neuron standalone components.

Neuron extension components
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Neuron extension components are components that integrate Neuron support
into open source machine learning frameworks, libraries or tools
enhancing their functionality and extending their capabilities as
necessary. When referring to Neuron extension components, we are also
referring to the parts of the open source machine learning framework or
library that are supported by Neuron. The software life-cycle of the
open source machine learning frameworks, libraries or tools that are
extended by Neuron is managed and maintained by their respective
communities or the vendors responsible for those specific components.
Examples for Neuron extension components are:

-  **Third party ML Library**: Examples include Neuron Nemo Megatron.
-  **Third party ML Framework**: Examples include PyTorch NeuronX and
   TensorFlow Neuron.

Neuron standalone components
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Neuron standalone components are self-contained components within the
Neuron SDK. Examples of such components are Neuron Compiler, Neuron
Tools and Neuron Runtime.

Neuron Model Classes
~~~~~~~~~~~~~~~~~~~~

A Neuron supported model class is tightly coupled with a specific Neuron
extension component (e.g. PyTorch NeuronX) or Neuron library (e.g.
NeuronX Distributed) and the workload type (e.g. Training or Inference).
For example a model can be supported at Beta level in PyTorch NeuronX
for training and Stable level in PyTorch NeuronX for inference.

Neuron features
~~~~~~~~~~~~~~~

A Neuron feature refers to any functionality or attribute that is part
of the Neuron SDK, whether it belongs to the entire Neuron SDK or to one
of its specific components.

Neuron APIs
~~~~~~~~~~~

A Neuron API refers to any API, CLI, environment variables, or flag that
belong to to the entire Neuron SDK or to one the Neuron components. A
Neuron API allows developers to interact with and leverage the
capabilities of the Neuron SDK and its components.

Examples include :ref:`Neuron Trace API <torch_neuron_trace_api>` and :ref:`Neuron Compiler flags <neuron-compiler-cli-reference-guide>`

Dependency software components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

External software components or frameworks that the Neuron
SDK and its components rely on for proper functioning and compatibility,
such as language runtimes or operating systems.

The software life-cycle of the dependency software components, is
managed and maintained by their respective communities or the vendors
responsible for those specific dependency software components. The
following terms are examples of underlying dependency software
components:

-  **Operating System (OS)**: Examples include Ubuntu 22 and Amazon
   Linux 2023
-  **Language Runtime**: Examples include Python 3.10

Neuron Deep Learning AMIs and Deep Learning Containers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:ref:`Neuron Deep Learning AMIs
(DLAMIs) <neuron-dlami-overview>`
and :ref:`Neuron Deep Learning Containers
(DLCs) <neuron_containers>` are pre-configured Amazon Machine Images and Docket container that
come with the Neuron SDK and necessary dependencies pre-installed,
providing a ready-to-use environment for machine learning development.

.. _neuron-software-lifecycle:

Neuron Software Life-cycle
--------------------------

The typical life-cycle for Neuron software consists of several phases, though not all phases are applicable to every type of Neuron software. The phases are as follows:

-  **Developer Preview or Beta** (these terms are used interchangeably in
   Neuron collaterals)
-  **Release Candidate (RC)**
-  **General Availability (GA) or Stable** (these terms are used
   interchangeably in Neuron collaterals)
-  **Maintenance**
-  **End-of-Support (EOS)**

The following table outlines the details for each phase for Neuron software:

+-------------------------------+----------------------------------------------------------------------------------------------------------------------+--------------------------------------------------+
|                               | Description                                                                                                          | Comments                                         |
+-------------------------------+----------------------------------------------------------------------------------------------------------------------+--------------------------------------------------+
| Developer Preview (Beta)      | In this phase, Neuron Software is not supported, should not be used in production environments,                      |                                                  |
|                               | and is meant for early access and feedback purposes only. It is possible for future releases                         |                                                  |
|                               | to introduce breaking changes.                                                                                       |                                                  |
|                               | See :ref:`Neuron Software Classification <sdk-classification>` for more information                                  |                                                  |
+-------------------------------+----------------------------------------------------------------------------------------------------------------------+--------------------------------------------------+
| Release Candidate (RC)        | Once AWS identifies a release to be a stable product, it may be marked as a Release Candidate (RC).                  | This phase applies only to Neuron SDK            |
|                               | This phase is usually short and during it AWS will provide for Neuron Software on an as-needed basis.                | and Neuron components                            |
+-------------------------------+----------------------------------------------------------------------------------------------------------------------+--------------------------------------------------+
| General Availability (Stable) | During this phase, AWS releases :ref:`regular <neuron-regular-updates>`updates for the Neuron Software based         |                                                  |
|                               | on a predefined release cadence of the Neuron SDK or provides :ref:`maintenance updates <neuron-maintenance-updates>`|                                                  |
|                               | for Neuron Software on an as-needed basis.                                                                           |                                                  |
|                               | See :ref:`Neuron Software Classification <sdk-classification>` for more information                                  |                                                  |
+-------------------------------+----------------------------------------------------------------------------------------------------------------------+--------------------------------------------------+
| Maintenance                   | During the maintenance phase, AWS will provide :ref:`maintenance updates <neuron-maintenance-updates>`               | This phase does not apply to Dependency Software |
|                               | for Neuron Software on an as-needed basis. Any new PIP, RPM, and Debian packages for the Neuron                      | Components, Neuron DLCs,                         |
|                               | Software, as well as updated versions of the Neuron DLAMIs and Neuron DLCs, will be released                         | Neuron DLAMIs, Neuron Features and APIs          |
|                               | only when deemed necessary by the AWS Neuron team.                                                                   |                                                  |
|                               | Users can expect updates to be less frequent compared to :ref:`regular <neuron-regular-updates>`                     |                                                  |
|                               | as the focus will be on addressing critical issues and ensuring the stability of the software.                       |                                                  |
|                               |                                                                                                                      |                                                  |
|                               | Maintenance Announcement: AWS will make a public :ref:`announcement <neuron-communication>` at least one month       |                                                  |
|                               | before the Neuron Software enters Maintenance phase.                                                                 |                                                  |
+-------------------------------+----------------------------------------------------------------------------------------------------------------------+--------------------------------------------------+
| End of Support (EOS)          | When Neuron Software reaches the end of its support lifecycle, it will no longer receive                             |                                                  |
|                               | :ref:`regular <neuron-regular-updates>` updates and :ref:`maintenance updates <neuron-maintenance-updates>`          |                                                  |
|                               | (including security updates). While AWS will continue to provide access to all previously released                   |                                                  |
|                               | PIP, RPM, and Debian packages for the Neuron Software, as well as earlier versions of the Neuron DLAMIs              |                                                  |
|                               | and Neuron DLCs, it's important to note that these older versions will not receive any updates or support.           |                                                  |
|                               | Customers can still use these resources at their own discretion, but it is highly recommended to upgrade             |                                                  |
|                               | to the latest available versions                                                                                     |                                                  |
|                               |                                                                                                                      |                                                  |
|                               | End of Support Announcement: AWS will make a public :ref:`announcement <neuron-communication>` at least one month    |                                                  |                                     
|                               | before a Neuron Software enters End of Support.                                                                      |                                                  |
+-------------------------------+----------------------------------------------------------------------------------------------------------------------+--------------------------------------------------+

.. _neuron-regular-updates:

Neuron Software Regular Updates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Regular updates for Neuron Software address the following areas: new
features, feature improvements, performance enhancements, bug
resolution, security vulnerability fixes, upgrades to Neuron dependency
software components and upgrades to Neuron extension components. To
handle these regular updates, AWS will release a new version of the
Neuron SDK, incrementing the minor version (the second digit in the
version number) for a minor release or incrementing the major version
(the first digit in the version number) for a major release when
significant changes that break compatibility are introduced. It's
important to note that any bug-fixes or security issues in regular
updates are not applied retroactively to previous versions of the Neuron
SDK. To benefit from these updates, users must adopt the latest release.

For more information see:

-  :ref:`Neuron DLAMIs and DLCs Updates <neuron-dlami-dlc-updates>`
-  :ref:`Neuron Extension Components Updates <neuron-extension-components-updates>`
-  :ref:`Neuron Software Versioning <neuron-software-versioning>`

**Neuron SDK Installation and Update instructions**
To install and update to the latest Neuron packages, customers need to pin the major
version of the Neuron package. For example, to install latest Neuron
tools package, call ``sudo apt-get install aws-neuronx-tools=2.*`` and
to install latest PyTorch Neuron package for Trn1, call
``pip install torch-neuronx==2.1.0.1.*``. This is done to future-proof
instructions for new, backwards-incompatible major version releases.

.. _neuron-maintenance-updates:

Neuron Software Maintenance Updates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Maintenance updates for Neuron Software address three key areas:
resolving bugs, fixing security vulnerabilities, and upgrading
dependency software components. At AWS discretion, additional critical
features or performance enhancement may also be included. To handle
these maintenance updates, AWS will release a new version of the Neuron
SDK, incrementing the patch number (the last digit in the version
number) to indicate a patch release. Major or minor releases may also
contain maintenance updates. It's important to note that these
maintenance updates are not applied retroactively to previous versions
of the Neuron SDK. To take advantage of these updates, users must adopt
the latest patch release.

For more information see:

-  :ref:`Neuron DLAMIs and DLCs Updates <neuron-dlami-dlc-updates>`
-  :ref:`Neuron Extension Components Updates <neuron-extension-components-updates>`
-  :ref:`Neuron Software Versioning <neuron-software-versioning>`

.. _neuron-dlami-dlc-updates:

Neuron DLAMIs and DLCs Updates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AWS will address :ref:`regular <neuron-regular-updates>` updates, life-cycle changes, maintenance
updates, and security issues related to any third-party software
included in the Neuron DLAMI or DLCs by releasing new versions of the
Neuron DLAMI or DLCs. However, updates won't be applied retroactively to
older versions of the Neuron DLAMI or DLCs. Instead, users will need to
use the new versions to get the latest updates. Generally, Neuron DLAMIs and Deep Learning Containers (DLCs) will support one latest LTS Linux Distribution version (Ubuntu, Amazon Linux, and Rocky9), with exceptions. Neuron Base DLAMIs (which come pre-installed with Neuron driver, EFA, and Neuron tools) will support the two latest versions of LTS Linux Distributions.


For more information see:

-  :ref:`Neuron Extension Components Updates <neuron-extension-components-updates>`
-  :ref:`Neuron Software Versioning <neuron-software-versioning>`

.. _neuron-extension-components-updates:

Neuron Extension Components Updates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a new version of an open source ML framework (e.g. PyTorch) is
supported by a Neuron extension component (e.g., PyTorch NeuronX), the
Neuron extension component for the latest supported ML framework version
will become the default for installation. If users wish to use a Neuron
extension component for an earlier supported ML framework version, they
will need to explicitly specify the desired version during installation.
After upgrading a Neuron extension component to support a newer version
of an ML framework, AWS will continue to provide :ref:`regular updates <neuron-regular-updates>`
for the Neuron extension component that supports the earlier ML
framework version for a minimum of 6 months. After the 6 months period,
the Neuron extension component for the earlier supported ML framework
version may transition into a maintenance mode. In the maintenance mode,
updates for the older Neuron extension component versions will be
provided on an as-needed basis, focusing on critical bug fixes and
security patches. For more information see: :ref:`Neuron extension component versioning <neuron-extension-components-versioning>`

.. _neuron-communication:

Communication methods
~~~~~~~~~~~~~~~~~~~~~

Neuron software classification and lifecycle announcements are
communicated as follows:

-  Neuron SDK documentation under
   `Announcements <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/announcements/index.html>`__

To see the list of available Neuron SDK versions and supported
dependency software components versions:

-  Neuron SDK documentation under `Release
   Content <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/releasecontent.html#latest-neuron-release-artifacts>`__
-  Neuron SDK documentation under `What’s
   New <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/index.html#neuron-whatsnew>`__

.. _neuron-software-versioning:

Neuron Software Versioning
--------------------------

Neuron SDK Documentation Versioning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Neuron SDK documentation is versioned and maps to the corresponding
Neuron SDK version. Users can switch to earlier versions of the Neuron
SDK documentation by selecting the version from the dropdown in bottom
left portion of the side bar.

Neuron SDK Versioning
~~~~~~~~~~~~~~~~~~~~~

The AWS SDK release versions are in the form of ``[A.B.C]`` where
``(A)`` represents the major version, ``(B)`` represents
the minor version, and ``(C)`` represents the patch version.

.. _neuron-extension-components-versioning:

Neuron extension components Versioning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Neuron extension components versioning (like PyTorch NeuronX) is in the
form ``[X.Y.Z].[A.B.C]``, where ``[X.Y.Z]`` represents the
third party component’s major (``X``), minor (``Y``), and patch
(``Z``) versions and ``[A.B.C]`` represents the Neuron extension
components (``A``), minor (``B``), and patch (``C``)
versions.

Neuron Standalone Component Versioning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Neuron Component versioning (except of Neuron extension components like
PyTorch NeuronX) is in the form ``[A.B.C.D]``, where ``A``
represents the major version, ``B`` represents the minor version,
and ``C.D`` represents the patch version.

.. _neuron-releases-types:

Neuron Software Release Types
-----------------------------

Major release
~~~~~~~~~~~~~~~~~

Increasing the major version indicates that the Neuron software
underwent significant and substantial changes in an incompatible manner.
Applications need to be updated in order for them to work with the
newest SDK version. It is important to update major versions carefully
and in accordance with the upgrade guidelines provided by AWS. After
increasing the major version, the Neuron software may not maintain
compatibility with previous supported versions of :ref:`Neuron
Runtime <nrt-api-guide>`, :ref:`Neuron Compiler <neuron_cc>`, and
:ref:`NEFF <neff-format>`.

Minor release
~~~~~~~~~~~~~~~~~

Increasing the minor version indicates that the Neuron software added
functionality in a backwards compatible manner.

Patch release
~~~~~~~~~~~~~~~~~

Increasing the patch version indicates that the Neuron software
added backward compatible bug or security fixes. A bug fix is defined as
an internal change that fixes incorrect behavior.

Pre-releases
~~~~~~~~~~~~~~~~

-  **Developer Preview (Beta)**: During this phase, the Neuron software
   is not supported, should not be used in production environments, and
   is meant for early access and feedback purposes only. It is possible
   for future releases to introduce breaking changes. In the case of a
   Developer Preview (Beta) release, the minor version will include a
   lower case ``b`` along with a (Beta) tag.
-  **Release Candidate (RC)**: Once Neuron identifies a release to be a
   stable product, it may mark it as a Release Candidate. Release
   Candidates are ready for GA release unless significant bugs emerge,
   and will receive full AWS Neuron support. In the case of a RC
   release, the minor version will include a lower case ``rc``
   along with a (RC) tag.

.. _sdk-classification:

Neuron Software Classification
------------------------------

This section explains the Neuron software classification for APIs,
libraries, packages, features, and Neuron supported model classes
mentioned in the Neuron documentation.

Neuron SDK and Neuron components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+-----------------+-----------------+-----------------+-------------+
|                 | Testing         | Features        | Performance |
+=================+=================+=================+=============+
| Developer       | Basic           | Minimal Viable  |             |
| Preview (Beta)  |                 | Product (MVP) \*|             |
+-----------------+-----------------+-----------------+-------------+
| Release         | Basic           | Minimal Viable  | Tested      |
| Candidate (RC)  |                 | Product (MVP)\* |             |
+-----------------+-----------------+-----------------+-------------+
| GA (Stable)     | Standard        | Incremental     | Tested      |
|                 | Product Testing | additions or    |             |
|                 |                 | changes         |             |
|                 |                 | in new releases |             |
+-----------------+-----------------+-----------------+-------------+

\* A minimum viable product (MVP) for a Neuron Component contains just
enough features to be usable by early customers who can then provide
feedback for future development. MVP can be different per use case
and depends on the specific package/library of interest. Please note
that in many cases, an MVP can also represent an advanced level of
features.

.. _neuron-apis-classification:

Neuron APIs
~~~~~~~~~~~

+----------------------+----------------------+----------------------+
|                      | API Contract         | API Backward         |
|                      |                      | Compatibility        |
+======================+======================+======================+
|       Alpha          |   Unstable and       |    No                |
|                      |   undocumented       |                      |
+----------------------+----------------------+----------------------+
| Developer Preview    | Major changes may    |    No                |
| (Beta)               | happen               |                      |
+----------------------+----------------------+----------------------+
| GA (Stable)          | Incremental changes  | Yes \*               |
|                      | in new releases      |                      |
|                      | (without breaking    |                      |
|                      | the API contract)    |                      |
+----------------------+----------------------+----------------------+

\* In certain cases, when necessary, AWS may introduce API changes that may break compatibility, with notice provided ahead of time.

.. _neuron-features-classification:

Neuron Features
~~~~~~~~~~~~~~~

+-----------------+-----------------+------------------------+-------------+
|                 | Testing         | Functionality          | Performance |
+=================+=================+========================+=============+
|                 | No formal       | Partial funcitonality  | Not tested  |
|     Alpha       | testing done    | with limited set of    | or          |
|                 |                 | core capabilities,     | evaluated   |
|                 |                 | far from Minium Viable |             |
|                 |                 | Product (MVP) \*       |             |
+-----------------+-----------------+------------------------+-------------+
| Developer       | Basic           | Minimum Viable         |             |
| Preview (Beta)  |                 | Product (MVP) \*       |             |
+-----------------+-----------------+------------------------+-------------+
| GA (Stable)     | Standard        | Incremental            | Tested      |
|                 | Product Testing | additions or changes   |             |
|                 |                 | in new releases        |             |
+-----------------+-----------------+------------------------+-------------+

\* A minimum viable product (MVP) for a Neuron Feature contains just
enough functionality to be usable by early customers who can then
provide feedback for future development. MVP can be different per use
case and depends on the specific feature of interest. Please note
that in many cases, an MVP can also represent an advanced level of
functionality.

.. _neuron-models-classification:

Neuron Supported Model Classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+----------------------+----------------------+----------------------+
|                      | Accuracy /           | Throughput / Latency |
|                      | Convergence          |                      |
+======================+======================+======================+
| Developer Preview    | Validated            | Tested               |
| (Beta)               |                      |                      |
+----------------------+----------------------+----------------------+
| GA (Stable)          | Validated            | Tested               |
+----------------------+----------------------+----------------------+
