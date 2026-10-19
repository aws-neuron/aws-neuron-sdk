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

**Important**: Currently, local doc builds require a Python 3.10.* environment. If you are on MacOS, you can install it from the terminal with `brew install python@3.10`. Add it to your working path with `brew link python@3.10` and confirm it works by running `python3.10 --version`.

### Docker Build

If you don't have Python 3.10 or a compatible gcc toolchain, use the Docker workflow:

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

2. Install the build dependencies. This requires a Python 3.10 installation and venv:

    ```bash
    cd .. # The root folder where you have your cloned Git repos; don't run this in the repo folder but one level up or you'll have venv files in your repo folder
    python3.10 -m venv venv && . venv/bin/activate
    pip install -U pip
    cd private-aws-neuron-sdk-staging
    pip install -r requirements.txt --extra-index-url=https://pypi.org/simple
    ```

3. Build the documentation into HTML. This command will allow you to build and view the
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

   For speedier builds in multiprocessor environments, run:

   ```bash
   sphinx-build -b html . _build/html -j auto
   ```

   **NOTE**: If you get an error for the spelling extension, like `Extension error: Could not import extension sphinxcontrib.spelling (exception: The 'enchant' C library was not found and maybe needs to be installed. See  https://pyenchant.github.io/pyenchant/install.html`, run `brew install enchant`.

4. Modify the source; please focus on the specific change you are contributing. If you also reformat all the code, it will be hard for us to focus on your change.
5. Rebuild the documentation with `sphinx-build -b html . _build/html`. Always ensure that the docs build without errors and that your changes look correct before pushing your changes to remote.
    * If you encounter errors that are unclear, run the build in verbose mode with `sphinx-build -vv -b html . _build/html`.
6. Commit your changes to your branch with a clear, scoped commit messages. Bad: "fixed stuff". Good: "Updated ref IDs in all containers topics".
7. Push your changes to remote (`git push origin`) and create a PR from your branch into `master` or the standing release branch (example: `release-2.27.0`). Answer any default questions in the pull request interface. Before you push, check your branch name against the protected-name rule below.
    * See: [pull request guide](https://help.github.com/articles/creating-a-pull-request/)).
8. Pay attention to any automated CI failures reported in the pull request, and stay involved in the conversation.

> [!IMPORTANT]
> **Do not use protected words in your branch name.** An aws-neuron organization ruleset (`neuron-ip-scan-all-private-repos-active-mode`) requires the `internal-scanner/scan` status check on every branch whose name matches a protected pattern. The rule also applies when you create a new branch. If your branch name matches, the push fails with this error:
>
> ```
> Required status check "internal-scanner/scan" is expected.
> ```
>
> The patterns match **anywhere in the branch name**, in these forms:
>
> * `release`, `Release`, or `RELEASE` (for example, `fix-release-notes` or `2.33-release-docs`)
> * `staging` (for example, `staging-cleanup`)
> * Names that start with `prod`
> * The exact names `main`, `mainline`, `master`, `dev`, `develop`, `development`, and `neuron-staging`
>
> To fix the error, push under a different branch name. You do not have to rename your local branch:
>
> ```bash
> git push -u origin HEAD:<your-alias>/<short-topic>
> ```
>
> Standing release branches such as `release-2.33.0` are created by repository admins, who can bypass the rule. Do not push to those branches directly. Open a PR into them instead.

Internal Amazon contributors: see the internal authoring runbook ("Runbook: Authoring a topic for the Neuron documentation") for updated process documentation.

## Build the NKI docs

The NKI documentation under `/nki` is sourced from the private
`private-nki-staging` repo, pinned here as a git submodule at `docs-src/nki`
(all content pinned from other repos lives under `/docs-src`). The submodule is
sparse-checked-out to `docs/public/nki`, and `/docs-src` is **excluded** from
the Sphinx build. The site builds `/nki`, so before building you copy the
pinned content into `/nki` with a manual step. This copy is intentionally
**not** run from `conf.py`, so a build can never fail on it — in particular the
public repo has no `/docs-src` submodule.

One tool drives all of this: `_utilities/nki_docs.py`, with three subcommands
(`sync`, `advance`, `refresh`). The tracked branch lives only in `.gitmodules`
(`submodule.docs-src/nki.branch`) — there is no separate pin file.

### One-time, after cloning

Initialize the submodule (a fresh clone can instead use
`git clone --recurse-submodules`):

```bash
git submodule update --init docs-src/nki
```

`git submodule update --init` does **not** by itself apply the sparse-checkout,
so on a fresh clone the mount contains the whole provider repo until it is
scoped. You don't need to run the sparse commands by hand: the `sync` step below
initializes the submodule (if needed) and re-applies the `docs/public/nki`
sparse scope every time before copying.

### Refresh `/nki` from the pinned content, then build

From the repo root, with the Python 3.10 venv active:

```bash
python _utilities/nki_docs.py sync     # copy docs-src/nki/docs/public/nki -> /nki
make html                              # or: sphinx-build -b html . _build/html
```

`sync` copies `docs-src/nki/docs/public/nki/**` into `/nki`, leaves
`/nki/library/` untouched, and never writes into `/docs-src`. It is
**delete-aware**: a git-tracked file under `/nki` that no longer exists in the
pinned source is removed, so an upstream delete or rename doesn't leave a stale
page behind. Untracked files (gitignored build output such as
`nki/api/generated`, or `__pycache__`) and `/nki/library` are never deleted. It
is safe to re-run; add `-v` for a per-file report of what was added, updated, or
removed (and any untracked drift), or `--dry-run` to preview without writing.
The copied files land in your working tree — commit them when you intend to
publish (see below).

### Change which NKI commit is pinned

To move the pin to the latest commit of the tracked branch and materialize it
into `/nki` in one step:

```bash
python _utilities/nki_docs.py refresh                      # tracked branch (.gitmodules)
python _utilities/nki_docs.py refresh --branch release/0.8.0   # switch branches
```

`refresh` advances the pin and then runs `sync`, so you can't advance the pin
but forget to copy the new content. Passing `--branch` updates the tracked
branch in `.gitmodules` (the single source of truth). It validates the branch
on the remote, advances the pin, re-applies the sparse-checkout, and **stages**
the new gitlink + `.gitmodules` (it does not commit). Use `advance` if you ever
need to move the pin without copying:

```bash
python _utilities/nki_docs.py advance                      # pin only, no /nki copy
```

### Publishing (cherry-pick) — keep the pin out of public

Publishing to the public `aws-neuron-sdk` repo stays a cherry-pick. Cherry-pick
the commit(s) that change the materialized `/nki` files. **Never** cherry-pick
the submodule machinery — the `docs-src/nki` gitlink, `.gitmodules`, or the
`.readthedocs.yml` `submodules:` block (all fenced with `# publish:staging-only`
markers). The public repo has no `/docs-src` and no access to the private
provider, so carrying any of those over would break the public build.

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

We may ask you to sign a [Contributor License Agreement (CLA)](http://en.wikipedia.org/wiki/Contributor_License_Agreement) for larger changes.

## Troubleshooting

If you can't seem to launch a build, chances are your venv is not using Python 3.10. From a terminal, `cd` into your repo clone and perform the following steps:

### Configure Python 3.10 and a supporting venv

1. Install Python 3.10 via `brew`: brew install python@3.10

2. Create the Python 3.10 venv:

  ```
  python3.10 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
  ```

3. Build the docs with `make html` or `sphinx-build -b html . _build/html -j auto`


### Activate a Python 3.10 venv

1. If you are still in the wrong venv, drop it from your shell.
`deactivate`

2. Activate the new 3.10 venv.
`source .venv/bin/activate`

#. Confirm.
`python -V `             # should print 3.10.20
`which sphinx-build`     # should print .../private-aws-neuron-sdk-staging/.venv/bin/sphinx-build
                         # if it still prints `/opt/homebrew/...`, the venv didn't take

4. Install the pinned Sphinx + extensions.
`pip install --upgrade pip`
`pip install -r requirements.txt`

5. Build the docs with `make html` or `sphinx-build -b html . _build/html -j auto`
