.. meta::
   :description: Getting started guide for Neuron Explorer — capture profiles, launch the UI, and start performance investigation.
   :date_updated: 06/02/2026

.. _new-neuron-profiler-setup:

Get started with Neuron Explorer
=================================

Overview
--------

In this guide, you'll capture a profile of your Neuron workload, launch Neuron Explorer, and upload the profile for interactive analysis.

By the end you will have:

* Captured a system or device profile
* Launched Neuron Explorer (browser or VS Code)
* Uploaded and viewed your profile in the interactive timeline

Prerequisites
-------------

* A Trainium or Inferentia EC2 instance (e.g., trn2.48xlarge, inf2.xlarge) with the :ref:`AWS Neuron DLAMI <setup-guide-index>`
* SSH key pair (``.pem`` file) for connecting to your instance
* Local machine with SSH client and a web browser (or VS Code)

Step 1: Connect and verify installation
-----------------------------------------

Launch an EC2 instance with a Neuron DLAMI. See the :doc:`setup guide </setup/index>` for details. Then, SSH into your EC2 instance and verify Neuron Explorer is installed:

.. code-block:: bash

   neuron-explorer --version

If not installed:

.. code-block:: bash

   sudo apt install aws-neuronx-tools

.. _neuron-explorer-step2-ssh-tunneling:

Step 2: Set up SSH tunneling
------------------------------

Neuron Explorer serves a web UI on port 3001 and an API backend on port 3002. You access both from your local machine through SSH tunnels.

From your **local machine**, open the tunnels:

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
   You must forward **both** ports. The UI on 3001 calls the API on 3002. If you only forward one, the page loads but shows no data. See :ref:`Troubleshooting <neuron-explorer-get-started-troubleshooting>` if you run into issues.

Step 3: Capture your profile
------------------------------

Profile types at a glance
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 15 40 25 20

   * - Type
     - What it captures
     - When to use
     - Output files
   * - **System**
     - Runtime events, API calls, model loads, CPU/memory
     - End-to-end execution flow
     - ``ntrace.pb``, ``trace_info.pb``, ``cpu_util.pb``, ``host_mem.pb``
   * - **Device**
     - Hardware-level NeuronCore instruction traces
     - On-device compute bottlenecks
     - Matched ``.neff`` + ``.ntff`` pair
   * - **Both**
     - Combined system + device view
     - Full optimization picture
     - All of the above

.. note::
   Device profiles require a matched pair: the ``.neff`` and ``.ntff`` share a numeric hash in their filename (e.g., ``neff_395760075800974.neff`` pairs with ``395760075800974_instid_0_vnc_0.ntff``).

For instructions on how to capture a profile, see :doc:`Capture Profiles in Neuron Explorer </tools/neuron-explorer/how-to-profile-workload>` or :doc:`Profile a NKI Kernel </nki/guides/use-neuron-profile>`.

After profiling, ``./profile_output`` will contain trace artifacts organized per process. Verify the output matches the :ref:`Expected Output <neuron-explorer-profile-expected-output>` section.

Step 4: Launch Neuron Explorer
--------------------------------

On the EC2 instance, run:

.. code-block:: bash

   neuron-explorer view

   # Expected Output:
   View a list of profiles at http://localhost:3001/
   ctrl-c to exit

This starts the **UI server** on port 3001 (web interface) and the **API server** on port 3002 (data backend).

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

1. Ensure SSH tunnels are active (see :ref:`Step 2: Set up SSH tunneling <neuron-explorer-step2-ssh-tunneling>`).
2. Configure the endpoint: click the extension in the left activity bar, select **Endpoint** on the bottom bar, choose **Custom endpoint**, and enter ``localhost:3002``.

  .. image:: /tools/images/VSCode_marketplace2.png

  .. image:: /tools/images/VSCode_marketplace3.png

3. Access the **Profile Manager** from the Neuron Explorer extension sidebar.

  .. image:: /tools/images/neuron-explorer-profile-manager-page.png

.. note::
   The VS Code extension uses the same API server. All upload methods (CLI, web UI) work interchangeably — once a profile is uploaded, it's visible in both interfaces.

Step 5: Upload your profile
-----------------------------

Choose the method that fits your workflow:

Option A: CLI upload
^^^^^^^^^^^^^^^^^^^^^

If you're already SSH'd into the instance, this is the quickest path:

.. code-block:: bash

   neuron-explorer view \
       -d ./profile_output \
       --ingest-only \
       --display-name "my-profile-run"

**Expected outcome:** After processing, the CLI outputs a direct link to your profile. Open it in your browser (via the tunnel) to view.

.. image:: /tools/images/neuron-explorer-cli-upload-output.png

**Useful flags:**

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

   * **For system profiles (or system + device):** click **Upload Profile** and select **Directory Upload For System Profile**. Then select the directory containing your ``.pb`` files (must include ``trace_info.pb``).

   .. image:: /tools/images/system-only-profile.png
      :width: 65%

   * **For device-only profiles:** click **Upload Profile** and select **Individual Files**. Upload your ``.neff`` and ``.ntff`` files in the designated boxes.

   .. image:: /tools/images/device-only-profile.png
      :width: 65%

**Expected outcome:** The profile appears in the **User Uploaded** table. Click **Refresh** to check processing status. Once complete, click the profile name to open the interactive timeline.

   .. image:: /tools/images/profile-expected-outcome.png

.. note::
   **Why two upload methods?** Directory Upload requires a system profile (``ntrace.pb`` + ``trace_info.pb``). It does not work with device-only profiles. For device-only profiles (just ``.neff`` + ``.ntff``), use **Individual Files**. If you have both system + device files, use **Directory Upload** as it picks up everything.

Option C: Export to JSON
^^^^^^^^^^^^^^^^^^^^^^^^^

For programmatic analysis (custom scripts, coding agents), export to JSON. This generates ``system_profile.json`` and ``device_profile_model_<model_id>.json`` per compiled model.

.. code-block:: bash

   neuron-explorer view \
       --session-dir ./profile_output \
       --output-format json \
       --output-file ./integrated_trace.json

**Quick text summary (no UI needed):**

.. code-block:: bash

   neuron-explorer view -d ./profile_output --output-format summary-text

**JSON schema (system_profile.json):**

The file contains event objects. It also includes ``mem_usage`` (sampled host memory) and ``cpu_util`` (CPU utilization per core).

.. code-block:: json

   {
       "Neuron_Runtime_API_Event": {
           "duration": 27094,
           "group": "nrt-nc-000",
           "id": 1,
           "instance_id": "i-0f207fb2a99bd2d08",
           "name": "nrt_tensor_write",
           "timestamp": 1729888371056597613,
           "type": 11
       },
       "Framework_Event": {
           "duration": 3758079,
           "group": "framework-80375131",
           "instance_id": "i-0f207fb2a99bd2d08",
           "name": "PjitFunction(matmul_allgather)",
           "timestamp": 1729888382798557372
       }
   }


.. _neuron-explorer-get-started-troubleshooting:

Troubleshooting
---------------

Connection issues
^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 5 25 25 45
   :header-rows: 1

   * - #
     - Symptom
     - Cause
     - Fix
   * - 1
     - UI does not load
     - SSH tunnel misconfigured
     - Verify both ports: ``ssh -L 3001:localhost:3001 -L 3002:localhost:3002 ...``
   * - 2
     - "Connection refused" on 3001/3002
     - Servers not running
     - Run ``neuron-explorer view`` on the instance first, then tunnel from local.
   * - 3
     - UI loads but shows no data
     - Only port 3001 forwarded
     - Add ``-L 3002:localhost:3002`` to your SSH command.
   * - 4
     - neuron-explorer not found
     - Tools not installed
     - ``sudo apt install aws-neuronx-tools`` or use the Neuron DLAMI.


Upload and viewing issues
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 5 35 30 30
   :header-rows: 1

   * - #
     - Symptom
     - Cause
     - Fix
   * - 1
     - Upload "successful" but Profile Manager shows "error process incomplete"
     - Missing required files
     - System profiles need ntrace.pb + trace_info.pb. Device profiles need matched .neff + .ntff.
   * - 2
     - Profile hangs in "Uploaded" state indefinitely
     - Processing failed silently
     - Try uploading without source code. If that works, check source is .tar.gz format.
   * - 3
     - Directory upload returns 500
     - Directory upload requires a system profile
     - For device-only profiles, use Individual Files instead.
   * - 4
     - "No profiling data"
     - Wrong directory
     - Use ``neuron-explorer view`` without ``--data-path``. Use ``-d <dir>`` for profile output.


Profiling results issues
^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 5 30 30 35
   :header-rows: 1

   * - #
     - Symptom
     - Cause
     - Fix
   * - 1
     - "DMA results may not be accurate"
     - 1. Compiler mismatch.
       2. DGE notifications are not collected by default, and this can also result in this warning.
     - 1. Update SDK and recompile. Safe to ignore for system profiling.
       2. To enable more accurate DMA data, re-capture with ``--enable-dge-notifs`` flag. Warning, this can result in timeout errors for large NEFFs. If an error occurs you can run with the flag off.
   * - 2
     - Out-of-memory during profiling
     - ``ProfileMode.DEVICE`` reserves ~5 GB HBM on Trn2
     - Remove it from your modes list if you don't need instruction-level device traces.
   * - 3
     - No CPU Neuron traffic in timeline
     - Framework trace not in correct subdirectory
     - This is likely because framework trace JSONs are not in per-process directories. Move ``neuron_framework_trace_rank_<N>.json`` into the matching ``<instance-id>_pid_<pid>/`` subdirectory.
   * - 4
     - Profile shows compilation, not execution
     - Didn't warm up
     - Run 3+ forwards before starting profiler.
   * - 5
     - Compiled model shows 0.2 ms (impossibly fast)
     - Async timing
     - Async dispatch — torch.compile queues work and returns immediately. Add explicit synchronization before timing:

       .. code-block:: python

          torch.neuron.synchronize()  # drain queue before timing
          t0 = time.time()
          for _ in range(50):
              compiled_model(x)
          torch.neuron.synchronize()  # wait for all work to complete
          avg_ms = (time.time() - t0) / 50 * 1000

   * - 6
     - Dropped events in system profile
     - ``WARN[0000] Warning: 1001 trace events were dropped during capture (stored 530560 out of 531561 total events).`` The trace buffers filled and oldest events were overwritten.
     -
       1. Increase buffer: set ``NEURON_RT_INSPECT_SYS_TRACE_MAX_EVENTS_PER_NC`` (default: 1,000,000). Uses more host memory.
       2. Apply capture-time filters (NeuronCore or event type).
       3. Shorten the profiled code region.
   * - 7
     - Incomplete JAX profiles
     - If your JAX profile has fewer events than expected, check:
     -
       1. Is ``jax.profiler.stop_trace`` called inside a ``with jax.profiler.trace`` block? Use ``stop_trace`` only with ``start_trace``.
       2. Is ``NEURON_RT_INSPECT_ENABLE`` set to 1? It should NOT be set when using ``jax.profiler``.
       3. Is ``NEURON_RT_INSPECT_OUTPUT_DIR`` set to the same directory passed to ``jax.profiler.trace``?


Next steps
----------

* :doc:`Capture Profiles in Neuron Explorer </tools/neuron-explorer/how-to-profile-workload>` — Full capturing and profiling reference (PyTorch, JAX, environment variables, CLI, filtering)
* :doc:`Neuron Explorer Full Documentation </tools/neuron-explorer/index>` — Complete viewer and feature reference
