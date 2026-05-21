# Onboarding

Welcome. This guide gets a new contributor from zero to a working local build of the AWS Neuron SDK documentation.

If you already have access and a working environment, skip to [`CONTRIBUTING.md`](./CONTRIBUTING.md) for the pull request workflow.

## What this repository is

This repository is the source for the AWS Neuron SDK documentation. Content is authored in reStructuredText (`.rst`) and Jupyter notebooks (`.ipynb`), rendered by [Sphinx](https://www.sphinx-doc.org/), and hosted on [Read the Docs](https://readthedocs.org/) at <https://awsdocs-neuron.readthedocs-hosted.com/>.

- **Source format:** reStructuredText (`.rst`) and Jupyter (`.ipynb`).
- **Build system:** Sphinx 5.3 driven by `conf.py` and `Makefile`.
- **Python version:** 3.10 (matches the Read the Docs build environment).
- **Hosting:** Read the Docs, configured via `.readthedocs.yml`.
- **Public docs URL:** <https://awsdocs-neuron.readthedocs-hosted.com/>
- **Staging docs URL:** <https://awsdocs-neuron-staging.readthedocs-hosted.com/> (per-branch previews at `/en/<branch-name>/`).

> **Do not push to `aws-neuron-sdk`.** That repository is the published output. All drafting, review, and staged previews happen in `private-aws-neuron-sdk-staging`.

## Before you start

Two things a new contributor should know before writing anything.

**A content plan is required to publish.** Whatever you are documenting — release notes, feature docs, a refactor, a deep dive — Doug Erickson captures the scope in a content plan and files Jira tickets for each topic. Without a plan and matching tickets, docs do not ship. If you need to start writing before a plan is finalized, do so with the understanding that scope may change.

**Authoring is template-driven, not free-form.** Every new or rewritten page starts from a content-type template in `_content-types/`. The content plan identifies which template maps to which topic and where the topic lives in the site. Consistency across pages is how customers (and AI agents) learn where information lives.

Doug runs **Neuron Docs Office Hours** if you want to walk through any of this live. When in doubt, ping him on Slack.

## Step 1: Get a GitHub account in `aws-neuron`

If you already have an Amazon-linked GitHub account and are a member of `aws-neuron`, skip to Step 2.

1. Take the GitHub training on A2Z.
2. Create a GitHub account if you do not have one. Include `aws` in the login name so customers can tell when they are interacting with an Amazon employee. General guidance: <https://w.amazon.com/bin/view/Open_Source/GitHub#How_do_I_sign_up_for_GitHub.3F>
3. Install `git secrets` and Code Defender. Instructions: <https://codedefender.proserve.aws.dev/installation>
4. Link your Amazon alias to your GitHub account through the Open Sourcerer tool: <https://console.harmony.a2z.com/open-sourcerer/connect-account>
5. Self-invite to the `aws-neuron` organization: <https://console.harmony.a2z.com/open-sourcerer/self-invite>. When Open Sourcerer prompts for the organization, enter `aws-neuron`.
6. Accept the invitation at <https://github.com/orgs/aws-neuron>.

## Step 2: Join the correct team

Team membership gates access to this repository. Request the team that matches your role:

- Neuron SDK team: <https://github.com/orgs/aws-neuron/teams/neuron-sdk>
- Product management team: <https://github.com/orgs/aws-neuron/teams/neuron-product-management>
- External collaborators: <https://github.com/orgs/aws-neuron/teams/neuron-external-contributors>

Click **Request to join** on the relevant team page. A maintainer must approve the request manually. Until approval lands, this repository returns 404 when you visit <https://github.com/aws-neuron/private-aws-neuron-sdk-staging>; once approved, it returns 200.

Approvals come from the [`aws-neuron/neuron-sdk` maintainers](https://github.com/orgs/aws-neuron/teams/neuron-sdk/members?query=role%3Amaintainer). If your request has been pending for more than a business day, ping Doug Erickson.

## Step 3: Get Read the Docs staging access

Branch previews on the staging site are private. You need a Read the Docs account linked to your GitHub account, plus membership in `neuron-team-ro` on the `awsdocs` RTD organization.

1. Sign up at <https://readthedocs.com/accounts/signup/> using your GitHub username.
2. Log in at <https://readthedocs.com/accounts/login/> using your GitHub account.
3. Ask Doug Erickson, Michael Wade, or Maen Suleiman to add you to the [`neuron-team-ro`](https://readthedocs.com/organizations/awsdocs/teams/neuron-team-ro/) team.
4. Accept the emailed invite from Read the Docs once access is provisioned. Without that acceptance, the staging URLs return 403 or 404.

## Step 4: Clone the repository

Use SSH. Make sure you have added your SSH public key to your GitHub account.

```bash
git clone git@github.com:aws-neuron/private-aws-neuron-sdk-staging.git
cd private-aws-neuron-sdk-staging
```

The default branch is `master`. Most PRs target `master` or a standing release branch (for example, `release-2.30.0`). Do not name a branch with the `release` prefix — release branches are protected and you cannot edit them. Do not include `/` in branch names; Read the Docs uses the branch name verbatim in the preview URL.

## Step 5: Install the build toolchain

You have two options: install Python locally, or use the provided Docker image. The Docker path is easier on macOS machines that do not already have a matching Python toolchain.

### Option A: Local Python 3.10

Create the virtual environment **outside** the repo directory. If you create it inside the repo, Sphinx will try to parse the notebooks in `.venv/` and the build will fail.

```bash
cd ..
python3.10 -m venv venv && source venv/bin/activate
pip install -U pip
cd private-aws-neuron-sdk-staging
pip install -r requirements.txt --extra-index-url=https://pypi.org/simple
```

Install Pandoc (used for Markdown ingestion):

```bash
brew install pandoc          # macOS
# or: apt-get install pandoc  (Linux)
```

Install the `enchant` C library for the Sphinx spelling extension:

```bash
brew install enchant         # macOS
# or: apt-get install enchant-2  (Linux)
```

### Option B: Docker

From the repo root:

```bash
./build.sh build    # build the image (first time only)
./build.sh html     # build HTML into _build/html/
./build.sh shell    # interactive shell for debugging
./build.sh clean    # remove _build/
```

Use this path if you can't install Python 3.10 or `enchant` on your host.

## Step 6: Build the documentation locally

From the repo root:

```bash
make html
```

Or equivalently:

```bash
sphinx-build -b html . _build/html
```

The first build takes around 15 minutes. Subsequent builds are incremental and faster.

Open the output:

```bash
open _build/html/index.html    # macOS
# or point any browser at _build/html/index.html
```

Other useful build targets:

```bash
make clean                                 # wipe _build/
sphinx-build -E -b html . _build/html       # clean build without wiping
sphinx-build -b linkcheck . _build/html     # find broken links
sphinx-build -b spelling . _build/html      # spell check
sphinx-build -b html . _build/html -j auto  # parallel build
sphinx-build -vv -b html . _build/html      # verbose for debugging
```

If you hit a build error you don't understand, run with `-vv` first. The most common root causes are:

- A virtual environment living inside the repo directory (move it out, or add it to `exclude_patterns` in `conf.py`).
- `enchant` not installed (the spelling extension fails to load).
- Broken links causing the build to stop. Add the URL to `linkcheck_ignore` in `conf.py` if the upstream link is intermittently flaky.

## Step 7: Make your first change

**Content plan first.** Before you open an editor, make sure Doug has a content plan covering your work and that Jira tickets exist for every page you plan to create or update. For bigger efforts, Doug will prepare a working branch with template stubs already in place. Small fixes (typos, broken links) do not need a plan, but anything with real scope does.

The PR workflow itself lives in [`CONTRIBUTING.md`](./CONTRIBUTING.md). The short version:

1. Start your branch from the right base. Feature docs and release notes tied to a release train branch from `release-X.XX.X`. Out-of-band work branches from `master`.
2. Author from a content-type template in `_content-types/`. Match the template for the intent of your page.
3. Build locally and confirm the page renders cleanly.
4. Push. Read the Docs picks up the push via webhook and builds a branch preview at `https://awsdocs-neuron-staging.readthedocs-hosted.com/en/<your-branch>/`. The build takes around 20 minutes.
5. Open a PR against `master` or the release branch. Link the Jira tickets and include the staging preview URL in the PR description.
6. Have the reviewers on standby (see the next section).

If you are adding or removing topics, rebuild the sitemap as described in [`CONTRIBUTING.md`](./CONTRIBUTING.md#updating-the-sitemap).

## How docs get published

Neuron ships on a 4-week release train with an 8-week planning cadence. Publication happens on the Thursday of week 4 of each train.

**Publish-day math:**

- Monday (EoD) of week 4: last day to merge your PR for this train.
- Monday–Wednesday: the docs team monitors the PR and Slack for late change requests from stakeholder or leadership reviews.
- Thursday: docs are published to the public site.

**Reviewer requirements.** Every docs PR needs three reviewers identified in the content plan:

- The PM for the component area (required).
- At least one SME — Lead or Sr+ Engineer for the component, or a qualified SA.
- A third, which can be another SME or Doug (especially for major changes).

Add the reviewers to the GitHub PR, create a Slack group chat with the same people, and share the PR link, content-plan link, and staging preview link.

**Missed cutoff?** The PR is redirected to the next release's branch. For a feature tied to a release, missing the docs cutoff can delay the feature itself. Plan to have your draft in PR form one week before the publish date.

**Squash and Merge** is the default. When approvals are complete, merge yourself or ask Doug to merge. Doug files a SIM ticket with Neuron DevOps for publication.

## Repository layout at a glance

| Path | Purpose |
|---|---|
| `conf.py` | Sphinx configuration |
| `index.rst` | Documentation homepage and root `toctree` |
| `Makefile` | Build automation |
| `.readthedocs.yml` | Read the Docs hosting configuration |
| `requirements.txt` | Python dependencies |
| `build.sh` | Docker build wrapper |
| `llms.txt` | Top-level summary for AI agents and LLM crawlers |
| `/about-neuron/`, `/frameworks/`, `/libraries/`, `/nki/`, `/tools/`, `/compiler/`, `/neuron-runtime/`, `/neuron-customops/`, `/deploy/`, `/setup/`, `/release-notes/` | Content directories |
| `/deploy/` | Workload orchestration and deployment — replaces the legacy `devflows/`, `containers/`, and `dlami/` trees. Subdirs: `ec2/`, `eks/`, `ecs/`, `batch/`, `sagemaker/`, `parallelcluster/`, `infrastructure/`, `environments/` (DLAMIs and DLCs), `images/` (container images), `tutorials/`, `third-party/`, `docker-examples/` |
| `_content-types/` | Templates for new pages (pick one before authoring) |
| `_ext/` | Custom Sphinx extensions |
| `_static/`, `_templates/` | Theme assets and overrides |
| `_utilities/` | Build and maintenance scripts |
| `archive/` | Retired docs kept for reference |
| `images/`, `includes/` | Shared assets and reusable RST snippets pulled in via `.. include::` |
| `info/` | Build-time excludes and metadata (not user-facing content) |
| `src/` | Source-code samples referenced by tutorials and how-tos |
| `static/` | SEO assets (`robots.txt`, sitemap, site-verification files) |

## Content conventions

- Every new or rewritten page starts from a template in `_content-types/`. The content plan identifies which one.
- Every page has a `.. meta::` block (`:description:`, `:date-modified:`) and a stable `.. _label:` reference anchor above its H1.
- Every page is added to the nearest parent `toctree`. No orphans.
- Use sentence-case headings, active voice, second person. See the AWS Technical Style Guide for the full rules.

**AI polish is expected.** Before you open the PR, run a Claude-based GenAI tool over your draft with the matching prompt from Doug's prompt library to clean up spelling, grammar, terminology, and AWS technical style. Review the output before accepting it. Options include Cedric, Diya (use "Balanced" with Claude 3.5 or later), or VS Code Q CLI or Cline. At a minimum, run a spelling and grammar check until automation is in place.

## Troubleshooting

**`404 Not Found` at the staging repo URL.** Your team-join request has not been approved yet. Give it a business day, then ping Doug Erickson.

**`403 Forbidden` or `404 Not Found` on staging preview URLs.** You have not accepted the Read the Docs email invite, or you are not yet in `neuron-team-ro`. See Step 3.

**`sphinx-build` fails with a notebook validation error.** Your virtual environment is inside the repo. Recreate it one directory up, or add its directory name to `exclude_patterns` in `conf.py`.

**`Extension error: Could not import extension sphinxcontrib.spelling … enchant C library`.** Install `enchant` with `brew install enchant` (macOS) or `apt-get install enchant-2` (Linux).

**Read the Docs build fails on a broken link.** Search the build log for `broken` to find the URL. Add the URL or its domain to `linkcheck_ignore` in `conf.py` if the upstream is unstable.

## Getting support

**Docs Office Hours.** Doug runs regular office hours for first-time contributors and anyone working through a content plan. Good for walking through the workflow, picking templates, or pressure-testing a draft.

**Doc Hack Day.** If your content plan has a lot of topics in a short window, ask Doug to schedule a Doc Hack Day (full or half). Everyone works drafts together, with live help on authoring, templates, Git, and AI prompts. These sessions consistently deliver more in a day than async work does in a week.

**Emergency publishing path.** If you need to publish urgently and Doug is unavailable, file a SIM ticket to CTI `AWS > Neuron > DevOps` with a title like `{Date}: Neuron docs publishing request for {component or effort}`. Describe the change and the required publish date. The Neuron DevOps oncall picks it up.

**Who to contact for what:**

- Repo access or team approvals: Doug Erickson, or any [`aws-neuron/neuron-sdk` maintainer](https://github.com/orgs/aws-neuron/teams/neuron-sdk/members?query=role%3Amaintainer).
- Read the Docs access: Doug Erickson, Michael Wade, Maen Suleiman.
- Content plans, templates, reviewers, publishing: Doug Erickson.
- Build or tooling issues: open an issue against this repo tagged `build`.

## References

- [`CONTRIBUTING.md`](./CONTRIBUTING.md) — PR workflow and build details
- [`README.md`](./README.md) — top-level product overview
- Runbook: [Authoring a topic for the Neuron documentation]()
- Internal Confluence: [AWS Neuron SDK Documentation Workflow, 2025 Edition]()
- Internal Confluence: [Joining the new GitHub organization and team]()
- Internal Confluence (legacy): [Neuron Sphinx/Read the Docs FAQ]()
