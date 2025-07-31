**IMPORTANT!** _If this is a documentation PR for a specific release, this PR must go the corresponding release branch_ (`release-X.XX.X`). _If it is an "out-of-band" doc update, the PR must go to the_ `master` _branch_.


## Required PR information

To expedite approvals and merges for releases, provide the following information (select the `...` button to the right at the top of your PR message to edit it):

> **AWS email alias**: {_your-name_}@amazon.com

>**Description**: {_What this documentation change is and why you made it. If you have a corresponding Jira ticket or content plan, link it here. The more details you provide around any decisions you made when preparing the docs, the less annoying comments you'll get preparing to release it._}

> **Date this must be published by**: {_If empty, we will assume the release date for the branch you're merging into._}

> **Link to ReadTheDocs staging for this branch's doc changes**: https://awsdocs-neuron-staging.readthedocs-hosted.com/en/{YOUR_BRANCH_NAME_HERE}/

> **Set the `docs-review-needed` label on the PR for tracking.**

## Before you request approvals

> Run a spelling and grammar check over your prose and make the changes it suggests. VSCode has a number of extensions (cSpell, LTeX) that you can use. You can also provide the rendered HTML for (or a cut-and-paste of) your pages to an AI and have it correct your spelling, grammar, and formatting issues. If you need an advanced prompt, contact @erickson-doug.

## Approvers

We require 3-4 approvers to merge for non-trivial content changes (where a "trivial" change is a typo/grammar fix or a minor update to the format syntax):

1. A senior+ engineer who will review your documentation for technical accuracy and clarity in communicating the technical concepts in your work
2. A product manager for your Neuron component area who will review it for customer relevance and product/component/feature messaging
3. The lead tech writer (@erickson-doug) who will review your work for overall doc design and quality, and perform the merge when all approvals are met
4. (For PRs with code/notebook submissions) A QA/test engineer who can run your code and confirm the results.

Make sure you get a commitment from these reviewers in advance! It's hard to get good quality doc reviews in order in the 11th hour of a release.

**Note**: For trivial changes, you only need @erickson-doug's approval. He will merge your content once he's confirmed the fixes on staging.

## Doc review checklist

### Engineering reviewer checklist

- [ ] I've confirmed that the contributions in this PR meet the current  [AWS Neuron writing guidelines](https://quip-amazon.com/m97CAO0kQFEU/Writing-for-AWS-Neuron).
- [ ] I've confirmed that the documentation submitted is technically correct to the best of my knowledge.
- [ ] I've confirmed that the documentation submitted has no spelling or grammar errors or use of internal jargon/terminology.
- [ ] I've verified the changes render correctly on RTD (link above).
- [ ] (If code is included) I've run tests to verify the contents of the change.

---

## For PRs that include code or notebook examples

**MANDATORY: PR must include test run output**

Provide this information for the QA reviewer in order to expedite their review.

**Test run output:**
Specify the release version, instance size and type, OS type and test output.

**For Training tutorials:**

{Convergence graph for training tutorials}

{Performance metrics `average_throughput`, `latency_p50`, `latency_p99` and MFU% if available}

Make sure this PR contains correct classification terms (Alpha, Beta, and Stable).

If possible, provide your results or a link to them for the reviewer to check your work.

## Code example/notebook content PR checklist

- [ ] (If applicable) I've automated a test to safeguard my changes from regression.
- [ ] (If applicable) I've posted test collateral to prove my change was effective and not harmful.
- [ ] (If applicable) I've added someone from QA to the list of reviewers.  Do this if you didn't make an automated test or feel it's appropriate for another reason.
- [ ] (If applicable) I've reviewed the licenses of updated and new binaries and their dependencies to make sure all licenses are on the pre-approved Amazon license list.  See https://inside.amazon.com/en/services/legal/us/OpenSource/Pages/BlessedOpenSourceLicenses.aspx.

By submitting this pull request, I confirm that you can use, modify, copy, and redistribute this contribution, under the terms of your choice.