.. _trouble-shooting-inf1-faq:

Troubleshooting for Inf1 - FAQ
==============================

.. contents:: Table of contents
   :local:
   :depth: 1


Performance is not what I expect it to be, what's the next step?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Please check our :ref:`performance-optimization` section on performance
tuning and other notes on how to use pipelining and batching to improve
performance.

Do I need to worry about size of model and size of inferentia memory? what problems can I expect to have?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Errors like this will be logged and can be found as shown
:ref:`neuron_gatherinfo`.

How can I debug / profile my inference request?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See :ref:`neuron-plugin-tensorboard`


How to report Bug/Feature Requests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We welcome you to use the Neuron GitHub issue tracker to report bugs or suggest
features.

When filing an issue, please check existing open, or recently closed,
issues to make sure somebody else hasn't already reported the issue.
Please try to include as much information as you can. Details like these
are incredibly useful:

-  A reproducible test case or series of steps
-  The version of our code being used
-  Any modifications you've made relevant to the bug
-  Anything unusual about your environment or deployment
