# Contributing Guidelines

Thank you for your interest in contributing to our project. Whether it's a bug report, new feature, correction, or additional
documentation, we greatly value feedback and contributions from our community.

Please read through this document before submitting any issues or pull requests to ensure we have all the necessary
information to effectively respond to your bug report or contribution.

## Reporting Bugs/Feature Requests

We welcome you to use the GitHub issue tracker to report bugs or suggest features.

When filing an issue, please check existing open, or recently closed, issues to make sure somebody else hasn't already
reported the issue. Please try to include as much information as you can. Details like these are incredibly useful:

* A reproducible test case or series of steps
* The version of our code being used
* Any modifications you've made relevant to the bug
* Anything unusual about your environment or deployment

## Contributing Workflow (via Pull Requests)

Contributions via pull requests are much appreciated. Before sending us a pull request, please ensure that:

1. You are working against the latest source on the *master* branch.
2. You check existing open, and recently merged, pull requests to make sure someone else hasn't addressed the problem already.
3. You open an issue to discuss any significant work - we would hate for your time to be wasted.

**Important**: Currently, local doc builds require a Python 3.9 environment. If you are on MacOS, you can install it from the terminal with `brew install python@3.9`. Add it to your working path with `brew link python@3.9` and confirm it works by running `python3.9 --version`.

### Docker Build

If you don't have Python 3.9/3.10 or a compatible gcc toolchain, use the Docker workflow:

```bash
./build.sh build   # Build Docker image (first time only)
./build.sh html    # Build HTML docs to _build/html/
./build.sh shell   # Interactive shell for debugging
./build.sh clean   # Remove _build/ directory
```

### Manual Build

To send us a pull request, please:

1. Clone the repository locally:

    ```bash
    git clone git@github.com:YOUR-USERNAME/private-aws-neuron-sdk-staging.git
    ```

2. Install the build dependencies. This requires a Python 3.9 installation and venv:

    ```bash
    cd .. # The root folder where you have your cloned Git repos; don't run this in the repo folder but one level up or you'll have venv files in your repo folder
    python3.9 -m venv venv && . venv/bin/activate
    pip install -U pip
    cd private-aws-neuron-sdk-staging
    pip install -r requirements.txt
    ```

3. Build the documentation into HTML. This command will allow you to view the
   rendered documentation by opening the generated `_build/html/index.html`. On first run, this will take about 15 mins. Subsequent html generations are incremental and will take less time.

   Run:

   ```bash
   sphinx-build -b html . _build/html
   ```

   Or leverage the make file and run:

   ```bash
   make html
   ```

   If this doesn't work, try this command:

   ```bash
   sphinx-build -C -b html . _build/html
   ```

   **IMPORTANT**: RTD runs LaTeX (latex.py) to generate the PDFs linked from each page. If you are getting build errors on RTD but your local builds are green, confirm that LaTeX builds are not failing locally by running: `sphinx-build -b latex . _build/html`. The most common cause of LaTeX build failures happens when an undefined ref ID (this is, not declared on any page in the form `.. _ref-id-here:`) is referenced in a link or toctree.

   **NOTE**: If you get an error for the spelling extension, like `Extension error: Could not import extension sphinxcontrib.spelling (exception: The 'enchant' C library was not found and maybe needs to be installed. See  https://pyenchant.github.io/pyenchant/install.html`, run `brew install enchant`.

4. Modify the source; please focus on the specific change you are contributing. If you also reformat all the code, it will be hard for us to focus on your change.
5. Rebuild the documentation with `sphinx-build -b html . _build/html`. Always ensure that the docs build without errors and that your changes look correct before pushing your changes to remote.
    * If you encounter errors that are unclear, run the build in verbose mode with `sphinx-build -vv -b html . _build/html`.
6. Commit your changes to your branch with a clear, scoped commit messages. Bad: "fixed stuff". Good: "Updated ref IDs in all containers topics".
7. Push your changes to remote (`git push origin`) and create a PR from your branch into `master` or the standing release branch (example: `release-2.27.0`). Answer any default questions in the pull request interface.
    * See: [pull request guide](https://help.github.com/articles/creating-a-pull-request/)).
8. Pay attention to any automated CI failures reported in the pull request, and stay involved in the conversation.

Updated process documentation can be found here: [Runbook: Authoring a topic for the Neuron documentation](https://quip-amazon.com/e9B9AM7Npb17/Runbook-Authoring-a-topic-for-the-Neuron-documentation).

## Updating the sitemap

If you add or remove a topic, you must recreate the sitemap. To do so:

1. From a shell, `cd` to the root of this repo (`private-aws-neuron-sdk-staging`) on your local machine.
2. Run the following command: `python3 ./_utilities/create_sitemap.py`. This will generate the sitemap as `sitemap.xml` in the root folder of the repo.
3. Rename the `sitemap.xml` file to `sitemap1.xml`.
4. Move the `sitemap1.xml` file to the `/static` folder, copying over the previous version.
5. Delete the generated `sitemap.xml` file from the root (**not** from `/static`) if you did a copy instead of a move.
6. Push a PR with the updated sitemap to remote and request DougEric review/approve it.

## Finding contributions to work on

Looking at the existing issues is a great way to find something to contribute on. As our projects, by default, use the default GitHub issue labels (enhancement/bug/duplicate/help wanted/invalid/question/wontfix), looking at any 'help wanted' issues is a great place to start.
    * Or, if you're so inclined, get on DougEric's Christmas card list by fixing broken links, formatting errors, removing stale topics, and fixing spelling/grammar errors.

## Code of Conduct

This project has adopted the [Amazon Open Source Code of Conduct](https://aws.github.io/code-of-conduct).
For more information see the [Code of Conduct FAQ](https://aws.github.io/code-of-conduct-faq) or contact
opensource-codeofconduct@amazon.com with any additional questions or comments.

## Security issue notifications

If you discover a potential security issue in this project we ask that you notify AWS/Amazon Security via our [vulnerability reporting page](http://aws.amazon.com/security/vulnerability-reporting/). Please do **not** create a public github issue.

## Licensing

See the [LICENSE-DOCUMENTATION](./LICENSE-DOCUMENTATION), [LICENSE-SAMPLECODE](./LICENSE-SAMPLECODE) and [LICENSE-SUMMARY-DOCS-SAMPLES](./LICENSE-SUMMARY-DOCS-SAMPLES) files for our project's licensing. We will ask you to confirm the licensing of your contribution.

We may ask you to sign a [Contributor License Agreement (CLA)](http://en.wikipedia.org/wiki/Contributor_License_Agreement) for larger chan
