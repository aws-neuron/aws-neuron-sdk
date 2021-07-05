Troubleshooting FAQs
====================

.. contents::
   :local:
   :depth: 1


Performance is not what I expect it to be, what's the next step?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Please check our :ref:`performance-optimization` section on performance
tuning and other notes on how to use pipelining and batching to improve
performance!

Do I need to worry about size of model and size of inferentia memory? what problems can I expect to have?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Errors like this will be logged and can be found as shown
:ref:`neuron_gatherinfo`.

How can I debug / profile my inference request?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See :ref:`neuron-plugin-tensorboard`

Contributing Guidelines FAQs
----------------------------

Whether it's
a bug report, new feature, correction, or additional documentation, we
greatly value feedback and contributions from our community.

Please read through this document before submitting any issues or pull
requests to ensure we have all the necessary information to effectively
respond to your bug report or contribution.

How to reporting Bugs/Feature Requests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We welcome you to use the GitHub issue tracker to report bugs or suggest
features.

When filing an issue, please check existing open, or recently closed,
issues to make sure somebody else hasn't already reported the issue.
Please try to include as much information as you can. Details like these
are incredibly useful:

-  A reproducible test case or series of steps
-  The version of our code being used
-  Any modifications you've made relevant to the bug
-  Anything unusual about your environment or deployment

Contributing via Pull Requests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Contributions via pull requests are much appreciated. Before sending us
a pull request, please ensure that:

1. You are working against the latest source on the *master* branch.
2. You check existing open, and recently merged, pull requests to make
   sure someone else hasn't addressed the problem already.
3. You open an issue to discuss any significant work - we would hate for
   your time to be wasted.

To send us a pull request, please:

1. Fork the repository.
2. Modify the source; please focus on the specific change you are
   contributing. If you also reformat all the code, it will be hard for
   us to focus on your change.
3. Ensure local tests pass.
4. Commit to your fork using clear commit messages.
5. Send us a pull request, answering any default questions in the pull
   request interface.
6. Pay attention to any automated CI failures reported in the pull
   request, and stay involved in the conversation.

GitHub provides additional document on `forking a
repository <https://help.github.com/articles/fork-a-repo/>`__ and
`creating a pull
request <https://help.github.com/articles/creating-a-pull-request/>`__.

How to find contributions to work on
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Looking at the existing issues is a great way to find something to
contribute on. As our projects, by default, use the default GitHub issue
labels (enhancement/bug/duplicate/help wanted/invalid/question/wontfix),
looking at any 'help wanted' issues is a great place to start.

What is the code of conduct
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This project has adopted the `Amazon Open Source Code of
Conduct <https://aws.github.io/code-of-conduct>`__. For more information
see the `Code of Conduct
FAQ <https://aws.github.io/code-of-conduct-faq>`__ or contact
opensource-codeofconduct@amazon.com with any additional questions or
comments.

How to notify for a security issue
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you discover a potential security issue in this project we ask that
you notify AWS/Amazon Security via our `vulnerability reporting
page <http://aws.amazon.com/security/vulnerability-reporting/>`__.
Please do **not** create a public github issue.

What is the licensing
~~~~~~~~~~~~~~~~~~~~~~~~

See the :ref:`license-documentation` and :ref:`license-summary-docs-samples` files
for our project's licensing. We will ask you to confirm the licensing of
your contribution.

We may ask you to sign a `Contributor License Agreement
(CLA) <http://en.wikipedia.org/wiki/Contributor_License_Agreement>`__
for larger changes.
