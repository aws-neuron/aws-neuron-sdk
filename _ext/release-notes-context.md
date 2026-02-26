# Release Notes Writing Guidelines

## Core Principles

### Answer Three Questions for Every Item

- **What?** — What feature/API is affected?
- **When?** — Under what conditions does this occur?
- **So what?** — What is the impact on the user?

### All Content Must Be:

- **Customer-visible** - Written from the customer's perspective about capabilities they can use
- **Documented** - If documentation doesn't exist, exclude the feature. All new features must include documentation URLs.
- **Actionable** - Include workarounds, timelines, or how to check if affected

## DO:

- **Write in customer-visible terms** - Describe what customers can now do, not how it was implemented
- **State the impact clearly** - Use concrete language about what happens to users
- **Be specific about conditions** - Replace vague phrases with precise conditions
- **Quantify performance improvements** - Provide specific before/after metrics (e.g., "improved from 2.164x to 3.654x speedup") and state the conditions that trigger these improvements (e.g., "for batch I/O operations with 1024 ops at 10KB")
- **Explain the impact of wrong defaults** - When fixing incorrect default values, state what the wrong default was and what impact it had on users
- **Specify what was missing** - When fixing "missing" items, list what was missing and confirm they are now documented
- **Describe previous behavior for bugs** - Always explain what the incorrect behavior was before the fix
- **Categorize breaking changes correctly** - If a bug fix changes API behavior (e.g., renaming a parameter), list it under Breaking Changes, not Bug Fixes
- **Provide actionable information** - Include workarounds if available, fix timelines if known, or how users can check if they're affected
- **Provide migration guidance for breaking changes** - Tell users what they should do when behavior changes, with before/after examples
- **Link to documentation** - Every feature must have corresponding documentation with URL
- **Include documentation URLs for all new features** - If no URL exists, either create documentation first or remove the feature from release notes
- **Use standard terminology** - Use terms your audience already knows
- **Use clear, descriptive sentences** - Transform technical phrases into customer-understandable language
- **Focus on customer-visible results** - Describe what customers will see, not internal mechanics
- **Drop unnecessary words** - Remove "when specified," "may," "is in progress" when they add no value
- **Remove empty sections** - Don't include placeholder text like "None in this release"
- **Verify accuracy** - Check version numbers, dates, and technical details
- **Run IP scanner** - Catch any internal code name leaks before publishing
- **Use active voice** - Write "The system ignores the parameter" instead of "The parameter is ignored"
- **Define abbreviations on first use** - Write "time to first token (TTFT)" before using "TTFT"
- **Remove temporal qualifiers** - Replace "for now" with specific timelines or remove entirely
- **Provide concrete examples** - Include calculation examples for complex parameters

## DO NOT:

- **Include internal code names** - Remove references like "TRN3PDS", "Mariana", "Penguin"
- **Document undocumented features** - If documentation doesn't exist, exclude the feature
- **Include features without documentation URLs** - Every new feature must have a documentation link
- **List unreleased features** - Only include features available to customers
- **Include internal-only metrics** - Remove metrics useful only internally
- **Document bugs never released** - Only include fixes for publicly released issues
- **Use internal API names** - Unless they're part of the public API
- **Include debug variables** - Remove environment variables meant only for internal use
- **Use vague language** - Avoid "in certain cases," "some patterns," "may sometimes"
- **Use ambiguous phrasing** - Avoid phrases like "Fixed dynamic for loop" that could mean multiple things
- **Leave impacts unexplained** - Don't just say "fixed wrong default" without explaining what the impact was
- **Mix breaking changes with bug fixes** - Parameter renames or behavior changes belong in Breaking Changes, not Bug Fixes
- **Create heavy noun chains** - Break up complex phrases (e.g., "dtype override was ignored during reshape" not "reshape dtype override not being applied")
- **Write without context** - Every change needs metrics, conditions, or migration guidance
- **Use hedging language** - Replace "may result in" with "results in" when deterministic
- **Focus on internal implementation** - Avoid phrases like "internally uses" or internal platform identifiers
- **Use passive voice without clear subject** - Avoid constructions where the actor is unclear
- **Reference undefined versions** - Don't use "V0" or "V1" without defining them

## Impact Statements

| Avoid | Prefer |
|-------|--------|
| "incorrectly interpret" | "produces incorrect results" |
| "not being applied" | "is ignored" |
| "failing check" | "crashes with validation error" |
| "may incorrectly interpret tensor shapes" | "can produce incorrect results when transposing tensors" |

## Conditions - Be Specific

| Avoid | Prefer |
|-------|--------|
| "in certain cases" | "when reduction axis is not the last dimension" |
| "some patterns" | "multi-dimensional transposes with more than 2 axes" |
| "may sometimes" | "consistently occurs when..." |
| "for now" | "Support is planned for version X.X.X" or remove entirely |
| "small inputs" | "inputs under 512 tokens" |
| "low batch sizes" | "batch sizes of 4 or less" |

## Phrasing Examples

### Bug Fixes:

| Avoid | Prefer |
|-------|--------|
| "Fixed bug in nrt_vnc_usage_find_internal" | "Improved error handling to return a clear error instead of asserting during nrt_init" |
| "Fixed dynamic for loop incorrectly incrementing the loop induction variable" | "Fixed: dynamic for loops now correctly increment the loop counter. Previously, the counter incremented incorrectly, causing [specific impact]" |
| "Fixed reshape dtype override not being applied when specified" | "Fixed a bug where specifying a data type override during a reshape operation was ignored" |
| "Fixed reshape of shared/private HBM tensors failing partition size check" | "Fixed a bug where reshaping tensors stored in shared or private HBM incorrectly failed the partition size check" |
| "Fixed incorrect default value for on_false_value" | "Fixed incorrect default value for on_false_value in nki.isa.range_select. Previously defaulted to [X], now correctly defaults to [Y], which [impact]" |

### Performance Improvements:

| Avoid | Prefer |
|-------|--------|
| "Optimized zero-copy operations by enabling descriptor merging" | "Enhanced zero-copy operation performance: Write performance improved from 2.164x to 3.654x speedup for batch I/O operations(1_Batch_1024_Ops_10_KBs)" |
| "Optimized mesh AllGather on TP8 configurations using destination routing" | "Optimized mesh AllGather: [X]% performance improvement on TP8 configurations when [specific conditions]" |

### New Features:

| Avoid | Prefer |
|-------|--------|
| "Added support for TRN3PDS platform" | "Added support for [public instance type name] with optimized topology configurations for distributed training. See [documentation URL]" |
| "Added IOCTL to lookup Neuron device/HBM for a given virtual address" | "Added capability to lookup Neuron device for a given virtual address, enabling frameworks to identify which device holds a tensor. See [documentation link] for API details" |

### Known Issues:

| Avoid | Prefer |
|-------|--------|
| "may incorrectly interpret tensor shapes in certain multi-dimensional transpose patterns" | "can produce incorrect results when transposing tensors with certain multi-dimensional shapes" |
| "Training, Inference, and Penguin kernels compilation and execution validation is in progress" | Remove entirely (internal project name and not customer-actionable) |
| "Chunked prefill is not supported on Neuron for now" | "Chunked prefill is not supported. If you attempt to enable it with DISABLE_NEURON_CUSTOM_SCHEDULER='1', the system will fail to start with an error. Use standard prefill mode instead." |

## Breaking Changes Checklist

When documenting breaking changes, always include:

1. **What changed** - The specific API, parameter, or behavior
2. **Why it's breaking** - What will stop working
3. **Migration path** - What users should do instead
4. **Example (if helpful)** - Show old vs. new usage

### Example:

**Breaking:** NumPy synonyms (e.g., `np.add` for `nl.add`) are no longer accepted in NKI API calls.

**Migration:** Replace all NumPy function calls with their NKI equivalents:
- Replace `np.add(x, y)` with `nl.add(x, y)`
- Replace `np.multiply(x, y)` with `nl.multiply(x, y)`

Always explain:
- Why is this breaking?
- What was the previous behavior?
- What is the workaround or migration effort?

## Quick Template

```
[Fixed/Known Issue]: [API/Feature] [impact] when [specific conditions]. [Optional: Workaround or timeline.]
```

### Example:

```
Fixed: nki.isa.dma_copy causes a runtime timeout when copying FP32 from SBUF to BF16 in HBM with indirect addressing. Workaround: cast to BF16 in SBUF before copying.
```

## Quality Checks Before Publishing

1. **No internal names** - Run IP scanner to catch code name leaks
2. **Customer value** - Each item explains why customers should care
3. **Documentation links** - New features link to relevant docs with URLs
4. **Documentation exists** - Verify all features are documented before including; if no documentation URL exists, remove the feature from release notes
5. **Accuracy** - Technical details are correct and verifiable
6. **Clarity** - Phrasing is clear and professional
7. **Completeness** - Previous behavior and migration paths explained
8. **Impact explained** - Bug fixes describe what was broken and what the impact was
9. **Active voice** - Sentences use active voice with clear subjects
10. **Abbreviations defined** - All abbreviations spelled out on first use
11. **No vague language** - All conditions and impacts are specific and quantified
12. **Examples provided** - Complex parameters include calculation examples

## Key Principles

### All content must be:

- **Customer-visible** (not internal implementation details)
- **Documented with URLs** (if docs don't exist, exclude it)
- **Impactful** (explain value, not just what changed)

### Every bug fix must answer:

- What was broken?
- What was the impact?
- What works now?

### Every new feature must include:

- Documentation URL
- Customer benefit
- Usage guidance or examples

## How to Review Release Notes

When reviewing release notes against these guidelines, provide feedback in the following format:

### Issue [Number]: [Brief Issue Title]

**Original Text:**
```
[Exact text from the release notes]
```

**Problem:**
[Description of the content/completeness issue]

**Phrasing Problem:**
[Description of the language/clarity issue, if applicable]

**Example Rewrite:**
```
[Suggested improved version showing correct phrasing and content]
```

**Action:**
- [Specific action item 1]
- [Specific action item 2]

## Review Process:

1. **Extract original text** - Include the exact text being reviewed
2. **Identify problems** - Separate content issues from phrasing issues
3. **Provide examples** - Show how to rewrite the text correctly
4. **List actions** - Give specific, actionable steps to fix each issue
5. **Check documentation** - Verify URLs exist for all new features; if not, recommend removal
6. **Verify completeness** - Ensure all three questions (What? When? So what?) are answered
7. **Check phrasing** - Identify vague language, passive voice, undefined terms, internal references
8. **Validate breaking changes** - Ensure migration guidance and before/after examples are included
