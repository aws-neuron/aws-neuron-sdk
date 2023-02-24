.. _update-neuron-pytorch:

Update to latest PyTorch Neuron (``torch-neuron``)
==================================================

.. include:: /general/setup/install-templates/inf1/note-setup-cntr.rst

.. contents:: Table of contents
   :local:
   :depth: 2


Develop on AWS ML accelerator instance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: /general/setup/install-templates/inf1/develop_mode.rst

.. include :: /general/setup/install-templates/inf1/note-setup-libnrt-warning.rst

.. tab-set::

   .. tab-item:: PyTorch 1.12.1

      .. tab-set::

         .. tab-item:: Ubuntu 20 DLAMI Base

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=update --category=compiler_framework --framework=pytorch --framework-version=1.12.1 --file=src/helperscripts/n2-manifest.json --os=ubuntu20 --instance=inf1 --ami=non-dlami

         .. tab-item:: Amazon Linux 2 DLAMI Base

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=update --category=compiler_framework --framework=pytorch --framework-version=1.12.1 --file=src/helperscripts/n2-manifest.json --os=amazonlinux2 --instance=inf1 --ami=non-dlami

   .. tab-item:: PyTorch 1.11.0

      .. tab-set::

         .. tab-item:: Ubuntu 20 DLAMI Base

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=update --category=compiler_framework --framework=pytorch --framework-version=1.11.0 --file=src/helperscripts/n2-manifest.json --os=ubuntu20 --instance=inf1 --ami=non-dlami

         .. tab-item:: Amazon Linux 2 DLAMI Base

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=update --category=compiler_framework --framework=pytorch --framework-version=1.11.0 --file=src/helperscripts/n2-manifest.json --os=amazonlinux2 --instance=inf1 --ami=non-dlami

   .. tab-item:: PyTorch 1.10.2

      .. tab-set::

         .. tab-item:: Ubuntu 20 DLAMI Base

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=update --category=compiler_framework --framework=pytorch --framework-version=1.10.2 --file=src/helperscripts/n2-manifest.json --os=ubuntu20 --instance=inf1 --ami=non-dlami

         .. tab-item:: Amazon Linux 2 DLAMI Base

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=update --category=compiler_framework --framework=pytorch --framework-version=1.10.2 --file=src/helperscripts/n2-manifest.json --os=amazonlinux2 --instance=inf1 --ami=non-dlami

   .. tab-item:: PyTorch 1.9.1

      .. tab-set::

         .. tab-item:: Ubuntu 20 DLAMI Base

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=update --category=compiler_framework --framework=pytorch --framework-version=1.9.1 --file=src/helperscripts/n2-manifest.json --os=ubuntu20 --instance=inf1 --ami=non-dlami

         .. tab-item:: Amazon Linux 2 DLAMI Base

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=update --category=compiler_framework --framework=pytorch --framework-version=1.9.1 --file=src/helperscripts/n2-manifest.json --os=amazonlinux2 --instance=inf1 --ami=non-dlami


   





Compile on compute instance
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: /general/setup/install-templates/inf1/compile_mode.rst


.. tab-set::

   .. tab-item:: PyTorch 1.12.1

      .. tab-set::

         .. tab-item:: Ubuntu 20 DLAMI Base

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=update --mode=compile --category=compiler_framework --framework=pytorch --framework-version=1.12.1 --file=src/helperscripts/n2-manifest.json --os=ubuntu20 --instance=inf1 --ami=non-dlami

         .. tab-item:: Amazon Linux 2 DLAMI Base

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=update --mode=compile --category=compiler_framework --framework=pytorch --framework-version=1.12.1 --file=src/helperscripts/n2-manifest.json --os=amazonlinux2 --instance=inf1 --ami=non-dlami

   .. tab-item:: PyTorch 1.11.0

      .. tab-set::

         .. tab-item:: Ubuntu 20 DLAMI Base

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=update --mode=compile --category=compiler_framework --framework=pytorch --framework-version=1.11.0 --file=src/helperscripts/n2-manifest.json --os=ubuntu20 --instance=inf1 --ami=non-dlami

         .. tab-item:: Amazon Linux 2 DLAMI Base

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=update --mode=compile --category=compiler_framework --framework=pytorch --framework-version=1.11.0 --file=src/helperscripts/n2-manifest.json --os=amazonlinux2 --instance=inf1 --ami=non-dlami

   .. tab-item:: PyTorch 1.10.2

      .. tab-set::

         .. tab-item:: Ubuntu 20 DLAMI Base

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=update --mode=compile --category=compiler_framework --framework=pytorch --framework-version=1.10.2 --file=src/helperscripts/n2-manifest.json --os=ubuntu20 --instance=inf1 --ami=non-dlami

         .. tab-item:: Amazon Linux 2 DLAMI Base

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=update --mode=compile --category=compiler_framework --framework=pytorch --framework-version=1.10.2 --file=src/helperscripts/n2-manifest.json --os=amazonlinux2 --instance=inf1 --ami=non-dlami

   .. tab-item:: PyTorch 1.9.1

      .. tab-set::

         .. tab-item:: Ubuntu 20 DLAMI Base

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=update --mode=compile --category=compiler_framework --framework=pytorch --framework-version=1.9.1 --file=src/helperscripts/n2-manifest.json --os=ubuntu20 --instance=inf1 --ami=non-dlami

         .. tab-item:: Amazon Linux 2 DLAMI Base

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=update --mode=compile --category=compiler_framework --framework=pytorch --framework-version=1.9.1 --file=src/helperscripts/n2-manifest.json --os=amazonlinux2 --instance=inf1 --ami=non-dlami




Deploy on AWS ML accelerator instance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: /general/setup/install-templates/inf1/deploy_mode.rst

.. include :: /general/setup/install-templates/inf1/note-setup-libnrt-warning.rst


.. tab-set::

   .. tab-item:: PyTorch 1.12.1

      .. tab-set::

         .. tab-item:: Ubuntu 20 DLAMI Base

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=update --mode=deploy --category=compiler_framework --framework=pytorch --framework-version=1.12.1 --file=src/helperscripts/n2-manifest.json --os=ubuntu20 --instance=inf1 --ami=non-dlami

         .. tab-item:: Amazon Linux 2 DLAMI Base

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=update --mode=deploy --category=compiler_framework --framework=pytorch --framework-version=1.12.1 --file=src/helperscripts/n2-manifest.json --os=amazonlinux2 --instance=inf1 --ami=non-dlami

   .. tab-item:: PyTorch 1.11.0

      .. tab-set::

         .. tab-item:: Ubuntu 20 DLAMI Base

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=update --mode=deploy --category=compiler_framework --framework=pytorch --framework-version=1.11.0 --file=src/helperscripts/n2-manifest.json --os=ubuntu20 --instance=inf1 --ami=non-dlami

         .. tab-item:: Amazon Linux 2 DLAMI Base

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=update --mode=deploy --category=compiler_framework --framework=pytorch --framework-version=1.11.0 --file=src/helperscripts/n2-manifest.json --os=amazonlinux2 --instance=inf1 --ami=non-dlami

   .. tab-item:: PyTorch 1.10.2

      .. tab-set::

         .. tab-item:: Ubuntu 20 DLAMI Base

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=update --mode=deploy --category=compiler_framework --framework=pytorch --framework-version=1.10.2 --file=src/helperscripts/n2-manifest.json --os=ubuntu20 --instance=inf1 --ami=non-dlami

         .. tab-item:: Amazon Linux 2 DLAMI Base

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=update --mode=deploy --category=compiler_framework --framework=pytorch --framework-version=1.10.2 --file=src/helperscripts/n2-manifest.json --os=amazonlinux2 --instance=inf1 --ami=non-dlami

   .. tab-item:: PyTorch 1.9.1

      .. tab-set::

         .. tab-item:: Ubuntu 20 DLAMI Base

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=update --mode=deploy --category=compiler_framework --framework=pytorch --framework-version=1.9.1 --file=src/helperscripts/n2-manifest.json --os=ubuntu20 --instance=inf1 --ami=non-dlami

         .. tab-item:: Amazon Linux 2 DLAMI Base

            .. include :: /general/setup/install-templates/inf1/note-setup-general.rst

            .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=update --mode=deploy --category=compiler_framework --framework=pytorch --framework-version=1.9.1 --file=src/helperscripts/n2-manifest.json --os=amazonlinux2 --instance=inf1 --ami=non-dlami