.. meta::
   :description: Getting started guide for Neuron Explorer — capture profiles, launch the UI, start performance investigation, organize profiles into folders, and delete profiles.
   :date_updated: 10/26/2026

.. _new-neuron-profiler-setup:

Get started with Neuron Explorer
=================================

Overview
--------

In this guide, you capture a profile of your Neuron workload, start Neuron Explorer, and upload the profile to view it in your browser or VS Code.

Neuron Explorer has three parts:

* **Capture** runs your workload on a Neuron device with profiling turned on and writes raw profile files (``.neff``, ``.ntff``, ``.pb``) to disk.
* **The Neuron Explorer server** runs on the instance and is started with ``neuron-explorer view``. It processes and stores profiles under a **data path** and serves them to the UI.
* **The UI** runs in your local browser or VS Code and reaches the server through an SSH tunnel.

Steps 1 to 3 run on the instance. Step 4 connects your laptop to the server. Steps 5 and 6 add your profile to the server and analyze it.

Prerequisites
-------------

* A Trainium or Inferentia EC2 instance (e.g., trn2.48xlarge, inf2.xlarge) with the :ref:`AWS Neuron DLAMI <setup-guide-index>`
* SSH key pair (``.pem`` file) for connecting to your instance
* Local machine with SSH client and a web browser (or VS Code)

If you're moving from Neuron Profiler or Profiler 2.0, see the :doc:`migration-faq`.

Step 1: Connect to your instance and verify installation
----------------------------------------------------------

Launch an EC2 instance with a Neuron DLAMI. See the :doc:`setup guide </setup/index>` for details. Then, SSH into your EC2 instance and verify Neuron Explorer is installed:

.. code-block:: bash

   neuron-explorer --version

If not installed:

.. code-block:: bash

   sudo apt install aws-neuronx-tools

.. _neuron-explorer-gs-capture:

Step 2: Capture your profile
----------------------------

Capture a **system profile** (runtime events, API calls, and host CPU and memory), a **device profile** (hardware-level NeuronCore traces), or both. For the differences, see :ref:`capture-profile-types`.

1. Before compiling, set the debug-info variables in :ref:`Setup (all methods) <capture-setup>`.
2. Capture with the method that fits your workload:

   * PyTorch: :doc:`how-to-profile-pytorch`
   * JAX: :ref:`JAX profiling <neuron-explorer-jax-profiling>`
   * Any framework, containers, or EKS: :ref:`Environment variable profiling <neuron-explorer-non-framework-user-experience>`
   * A compiled NEFF or NKI kernel: :ref:`neuron-explorer capture <neuron-explorer-capture-cli>`, or follow :doc:`Profile a NKI Kernel </nki/guides/use-neuron-profile>`
   * A quick system profile with no code changes: :ref:`neuron-explorer inspect <neuron-explorer-inspect-cli>`

   For what each environment variable does, see :doc:`environment-variables`.
3. Check the output as described in :ref:`neuron-explorer-verify-capture`.

A device profile is the ``.ntff`` traces plus the ``.neff`` files they executed, so keep them together when you upload. For what each file contains, see :doc:`profile-artifacts`.

.. _neuron-explorer-step4-launch-server:

Step 3: Start the Neuron Explorer server
----------------------------------------

On the EC2 instance, run:

.. code-block:: bash

   neuron-explorer view

   # Expected Output:
   View a list of profiles at http://localhost:3001/
   ctrl-c to exit

This starts the **UI server** (port 3001) and the **API server** (port 3002). Both listen on ``localhost`` only, so you reach them from your laptop through an SSH tunnel in the next step.

The server stores processed profiles in its **data path**, ``~/.local/share/neuron-explorer`` by default. Use ``--data-path <dir>`` to choose a different directory. Profiles in the data path are still there after the server restarts.

``neuron-explorer view`` runs in the foreground and stops when you press Ctrl+C or close the SSH session. To keep the server running after you disconnect, run it in a terminal multiplexer such as ``tmux`` or ``screen``.

.. _neuron-explorer-step2-ssh-tunneling:

Step 4: Open Neuron Explorer from your laptop
---------------------------------------------

From your **local machine**, open SSH tunnels to both server ports. To use VS Code instead, see :ref:`get-started-vscode`.

.. code-block:: bash

   ssh -i ~/path/to/your-key.pem \
       -L 3001:localhost:3001 \
       -L 3002:localhost:3002 \
       ubuntu@<instance-ip> -fN

.. list-table::
   :widths: 40 60

   * - ``-i ~/path/to/your-key.pem``
     - Path to your EC2 key pair
   * - ``-L 3001:localhost:3001``
     - Forwards the UI port
   * - ``-L 3002:localhost:3002``
     - Forwards the API port
   * - ``ubuntu@<instance-ip>``
     - Instance login (use ``ec2-user`` for Amazon Linux)
   * - ``-fN``
     - Runs tunnel in background (no shell)

.. important::
   You must forward **both** ports. The UI on 3001 calls the API on 3002. If you only forward one, the page loads but shows no data. See :doc:`Troubleshooting </tools/neuron-explorer/troubleshooting>` if you run into issues.

In your local browser, navigate to ``http://localhost:3001``:


.. image:: /tools/images/neuron-explorer-browser-landing.png

.. _get-started-vscode:

Using VS Code instead of the browser
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Install: Search for **AWS Neuron Explorer** (publisher: Amazon Web Services) in VS Code Extensions (Ctrl+Shift+X), or install from the `VS Code Marketplace <https://marketplace.visualstudio.com/items?itemName=AmazonWebServices.neuron-explorer>`_.

  .. image:: /tools/images/VSCode_marketplace.png

**Option A — Local Binary (no tunnel needed)**
     
From the Neuron Explorer view in VS Code, and after you have connected to the Neuron EC2 instance with Remote-SSH:

1. Configure the endpoint: click the extension in the left activity bar, select **Local Binary** on the bottom bar.

  .. image:: /tools/images/VSCode_marketplace2.png

2. Access Profile Manager from the extension sidebar.

  .. image:: /tools/images/neuron-explorer-profile-manager-page.png

3. Click **Upload Profile** and paste the path to the profile directory on the instance.

.. note::
   No SSH tunnel is required when running Neuron Explorer on the remote device. The extension starts the ``neuron-explorer`` server for you automatically when you select **Local Binary**.

**Option B — Custom endpoint (SSH tunnel)**

1. Ensure SSH tunnels are active (see :ref:`neuron-explorer-step2-ssh-tunneling`).
2. Configure the endpoint: click the extension in the left activity bar, select **Endpoint** on the bottom bar, choose **Custom endpoint**, and enter ``localhost:3002``.

  .. image:: /tools/images/VSCode_marketplace2.png

  .. image:: /tools/images/VSCode_marketplace3.png

3. Access the **Profile Manager** from the Neuron Explorer extension sidebar.

  .. image:: /tools/images/neuron-explorer-profile-manager-page.png

.. note::
   The VS Code extension uses the same API server. All upload methods (CLI, web UI) work interchangeably — once a profile is uploaded, it's visible in both interfaces.

.. _neuron-explorer-step5-upload-profile:

Step 5: Upload your profile
---------------------------

The server from :ref:`neuron-explorer-step4-launch-server` must be running. Choose the method that fits your workflow:

Option A: CLI upload
^^^^^^^^^^^^^^^^^^^^^

If you're already SSH'd into the instance, this is the quickest path:

.. code-block:: bash

   neuron-explorer view \
       -d ./profile_output \
       --ingest-only \
       --display-name "my-profile-run"

If you started the server with ``--data-path``, pass the same ``--data-path`` here. If the paths don't match, the command stops with an error that names the server's path.

**Expected outcome:** After processing, the CLI outputs a direct link to your profile. Open it in your browser (via the tunnel) to view.

.. image:: /tools/images/neuron-explorer-cli-upload-output.png

**Useful processing flags:**

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Flag
     - Description
   * - ``--ignore-device-profile``
     - Skip device-level traces (faster processing)
   * - ``--system-trace-secondary-group worker_gid,lnc_idx``
     - Reduce tracks for a cleaner view

Option B: Web UI upload
^^^^^^^^^^^^^^^^^^^^^^^^

1. If your profile output is on the EC2 instance and you want to use the browser uploader locally, transfer the files first:

.. code-block:: bash

   # Compress for faster transfer (recommended for large profiles)
   # Run on EC2:
   tar -czf profile_output.tar.gz ./profile_output

   # Transfer to local machine:
   scp -i ~/your-key.pem ubuntu@<instance-ip>:./profile_output.tar.gz .
   tar -xzf profile_output.tar.gz

2. Use your local browser or VSCode to open ``http://localhost:3001``. The **Profile Manager** page is displayed.

   .. image:: /tools/images/profile-workload.png

   Click the **Upload Profile** button on the top right to open the upload dialog:

   .. image:: /tools/images/upload-profile.png
      :width: 65%

3. Enter a **Profile Name** (required).

4. Choose your upload method based on profile type:

   * **For system profiles (or system + device):** click **Upload Profile** and select **Directory Upload**. Then select the directory containing your ``.pb`` files (must include ``trace_info.pb``).

   .. image:: /tools/images/system-only-profile.png
      :width: 65%

   * **For device-only profiles:** Use **Directory Upload** or click **Upload Profile** and select **Individual Files**. Upload your ``.neff`` and ``.ntff`` files in the designated boxes. 

   .. image:: /tools/images/device-only-profile.png
      :width: 65%

5. (Optional) To use the Source Code Viewer, add your source files or folders in the **Source Code** section below the profile section. This section appears for both upload methods and is separate from the profile upload: source files inside the profile directory are **not** picked up by **Directory Upload**. See :ref:`neuron-explorer-source-code`.

**Expected outcome:** The profile appears on the **My Uploads** tab. Click **Refresh** to check processing status. Once complete, click the profile name to open the interactive timeline.

   .. image:: /tools/images/profile-expected-outcome.png

.. _neuron-explorer-step6-analyze:

Step 6: Analyze your profile
----------------------------

Open your profile from the **Profile Manager**. Start with the :doc:`UI overview <overview-ui-tour>`, a guided tour of every viewer grouped by use case. Then go deeper with the viewer that fits your investigation:

* :doc:`System Trace Viewer <overview-system-profiles>`: framework, runtime, and host activity over time
* :doc:`Device Trace Viewer <overview-device-profiles>`: NeuronCore engine, DMA, and instruction timeline
* :doc:`Hierarchy Viewer <overview-hierarchy-view>`: model layers mapped down to hardware execution
* :doc:`Source Code Viewer <how-to-link-view-source-code>`: link instructions to NKI and framework source
* :doc:`Summary Viewer <overview-summary-page>`: high-level metrics for the profile
* :doc:`Tensor Viewer <overview-tensor-viewer>` and :doc:`Memory Viewer <overview-memory-viewer>`: tensor and memory usage
* :doc:`Region Highlighter <overview-region-highlighter>`: regions of interest such as collectives and NKI regions

For terms used in the UI, see the :doc:`glossary`.

.. note::
   **Share your view with teammates.** After you open a profile, Neuron Explorer encodes your current time range, selected event, and active annotation in the page URL. Copy the URL and share it to give others the same view with no additional setup. The shared URL only works for users connected to the same Neuron Explorer instance.

Analyze programmatically
^^^^^^^^^^^^^^^^^^^^^^^^

For programmatic analysis (custom scripts, coding agents), export the profile as Parquet tables. Parquet is the default output format and holds the complete profile data, and the UI reads the same tables, so what you query matches what you see on screen. With ``--output-file``, the Parquet files are written to the directory you name.

.. code-block:: bash

   neuron-explorer view \
       --session-dir ./profile_output \
       --output-format parquet \
       --output-file ./profile_parquet

For the table layout and Python examples, see :doc:`how-to-analyze-profile-output`. For the fields in each table, see :doc:`profile-schema-reference`.

.. note::
   We intend to end support for the full-profile JSON output (``--output-format json``) in Neuron 2.34. Use the Parquet output instead.

**Quick text summary (no UI needed):**

.. code-block:: bash

   neuron-explorer view -d ./profile_output --output-format summary-text

Organize profiles into folders
------------------------------

Folders group related profiles — for example, every run from one experiment. Folders are private to
you. A profile can belong to more than one folder. Folders support only one level, so you cannot
nest one folder inside another.

To create a folder, open the **Profile Manager** page, select the **My Folders** tab, and choose
**Create Folder** — or **Create your first folder** if you have none yet. Names are limited to 100
characters and cannot contain spaces; use underscores, hyphens, or dots instead.

To file profiles, go to the **My Uploads** tab and either choose the folder icon in a profile's
**Actions** column, or select one or more profiles and choose **Actions** > **Add to folder**. In
the **Add to Folder** dialog, select an existing folder — or choose **Create new folder...** — and
then choose **Add**. The upload form also has a **Folder** field, which defaults to
**None (no folder)**, so you can file a profile as you upload it.

.. image:: /tools/neuron-explorer/images/add-to-folder-actions-menu.png
   :alt: A selected profile in the My Uploaded Profiles table with the Actions menu open on Add to folder, and the folder icon visible in the Actions column of the row.

The **My Folders** tab lists each folder with the number of profiles it holds. Choose a folder name
to open it and see those profiles in the same table the **My Uploads** tab uses, then use the
**My Folders** breadcrumb to go back. From the **My Folders** tab you can also rename a folder with
the edit icon beside its name, or delete folders by selecting them, choosing **Delete**, and
confirming. To take a profile out of a folder, open the folder and choose the remove icon in the
profile's row, or select profiles and choose **Actions** > **Remove from folder**.

.. image:: /tools/neuron-explorer/images/my-folders-tab.png
   :alt: The My Folders tab listing two folders with the number of profiles in each, an edit icon beside each folder name, and the Delete and Create Folder buttons.

.. note::
   Removing a profile from a folder, or deleting the folder itself, leaves the profile in place on
   the **My Uploads** tab. Deleting a folder takes its profiles out of that folder only; they remain
   in any other folders that contain them. Deleting a profile is a separate, permanent action, and
   it removes that profile from every folder that contains it. See
   :ref:`Delete a profile <neuron-explorer-get-started-delete-profile>`.

.. _neuron-explorer-get-started-delete-profile:

Delete a profile
----------------

When you no longer need a profile, delete it from the **Profile Manager** page to free up storage.

**Before you start:** The delete action is available only on the **My Uploads** tab. You can delete
only the profiles that you uploaded. If you try to delete a profile that someone else uploaded,
Neuron Explorer reports an error and leaves the profile in place.

To delete one or more profiles:

1. Open the **Profile Manager** page and select the **My Uploads** tab.

2. Start the deletion in one of two ways:

   * Choose the delete icon in the **Actions** column of a profile's row to delete only that
     profile.

   * Select the checkbox for one or more profiles, choose **Actions**, and then choose
     **Delete profile**.

   .. image:: /tools/neuron-explorer/images/delete-profile-actions-menu.png
      :alt: The My Uploaded Profiles table with a profile selected, the delete icon in the Actions column, and the Actions menu open on Delete profile.

3. In the confirmation dialog, review the profiles to be deleted, type ``delete`` in the
   confirmation field, and then choose **Delete**. The **Delete** button stays disabled until you
   type the confirmation text exactly.

   .. image:: /tools/neuron-explorer/images/delete-profile-confirmation.png
      :alt: The Delete profile confirmation dialog, showing the permanence warning and the field for typing the word delete.

**Expected outcome:** A success notification confirms the deletion, and the profiles no longer
appear on the **My Uploads** tab.

.. warning::
   Deleting a profile is permanent and cannot be undone. Neuron Explorer removes the profile's
   stored files and metadata, along with its tags, annotations, and folder membership, and makes
   any persistent link to the profile inaccessible, including the view links described earlier in
   this guide. To recover a deleted profile, upload it again as described in
   :ref:`neuron-explorer-step5-upload-profile`, and then recreate any tags and
   annotations.

.. note::
   When you delete several profiles at once, each profile is deleted independently. If some
   deletions fail, Neuron Explorer deletes the rest and reports the names of the profiles it could
   not delete. Select those profiles and retry.

Next steps
----------

* :doc:`Neuron Explorer documentation </tools/neuron-explorer/index>`: all viewers, tutorials, and references
* :doc:`Troubleshooting & FAQs </tools/neuron-explorer/troubleshooting>`: connection, upload, and capture issues
