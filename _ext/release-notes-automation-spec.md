# Release Notes Review Automation Specification

## Overview

This specification defines a GitHub Action that automatically reviews release notes files in pull requests using Amazon Q CLI to ensure they meet quality standards defined in the release notes writing guidelines.

## Purpose

Automate the review of release notes changes to:
- Ensure consistency and quality across all release notes
- Catch common issues before human review
- Provide immediate feedback to PR authors
- Reduce manual review burden on documentation team

## Scope

### In Scope
- PRs labeled with "release-notes"
- RST files under `/release-notes/components/` directory
- Files that have been modified in the PR (not just added to context)
- Automated review using Q CLI with release notes guidelines
- Posting review feedback as PR comments

### Out of Scope
- Release notes files outside `/release-notes/components/`
- Non-RST files
- PRs without the "release-notes" label
- Manual approval/rejection of PRs (action only provides feedback)

## Requirements

### Functional Requirements

#### FR1: PR Detection and Filtering
- **FR1.1**: Action triggers on pull request events (opened, synchronize, labeled)
- **FR1.2**: Action only runs when PR has "release-notes" label
- **FR1.3**: Action identifies all changed RST files in `/release-notes/components/` directory

#### FR2: File Analysis
- **FR2.1**: Action reads content of each changed RST file
- **FR2.2**: Action loads release notes guidelines from `_ext/release-notes-context.md`
- **FR2.3**: Action processes files individually to provide file-specific feedback

#### FR3: Q CLI Integration
- **FR3.1**: Action invokes Amazon Q CLI with appropriate context
- **FR3.2**: Action provides Q CLI with:
  - Release notes guidelines from `_ext/release-notes-context.md`
  - Content of the changed RST file
  - Instruction to review against guidelines
- **FR3.3**: Action captures Q CLI output for each file

#### FR4: Review Feedback
- **FR4.1**: Action formats Q CLI feedback into readable PR comment
- **FR4.2**: Action posts comment to PR with review results
- **FR4.3**: Comment includes:
  - List of files reviewed
  - Issues found per file (using format from guidelines)
  - Suggested improvements
  - Link to full guidelines document
- **FR4.4**: If no issues found, action posts positive confirmation

#### FR5: Error Handling
- **FR5.1**: Action handles Q CLI failures gracefully
- **FR5.2**: Action reports when no RST files are found in scope
- **FR5.3**: Action logs errors for debugging without failing the PR

### Non-Functional Requirements

#### NFR1: Performance
- Action completes review within 5 minutes for typical PRs (1-5 files)
- Action processes files in parallel when possible

#### NFR2: Security
- Action uses GitHub secrets for Q CLI credentials
- Action has read-only access to repository
- Action has write access only to PR comments

#### NFR3: Maintainability
- Action configuration is version controlled in `.github/workflows/`
- Action uses official Q CLI container/action when available
- Action logic is simple and well-documented

## User Stories

### US1: Automatic Review Trigger
**As a** documentation contributor  
**I want** the review action to run automatically when I label my PR  
**So that** I get immediate feedback without manual intervention

**Acceptance Criteria:**
- Action triggers when "release-notes" label is added
- Action runs on subsequent commits to labeled PR
- Action does not run on PRs without the label

### US2: Targeted File Review
**As a** documentation contributor  
**I want** only my changed release notes files to be reviewed  
**So that** I get relevant feedback without noise from unchanged files

**Acceptance Criteria:**
- Only files in `/release-notes/components/*.rst` are reviewed
- Only files modified in the PR are analyzed
- Files in other directories are ignored

### US3: Clear Feedback
**As a** documentation contributor  
**I want** clear, actionable feedback on my release notes  
**So that** I know exactly what to improve

**Acceptance Criteria:**
- Feedback follows the format specified in guidelines
- Each issue includes: original text, problem, example rewrite, action items
- Feedback is posted as a PR comment
- Comment includes link to full guidelines

### US4: No False Failures
**As a** documentation contributor  
**I want** the action to provide feedback without blocking my PR  
**So that** I can address issues without being blocked by automation

**Acceptance Criteria:**
- Action never fails the PR check
- Action always succeeds even if issues are found
- Issues are reported as comments, not check failures

## Technical Design

### GitHub Action Workflow

**File Location:** `.github/workflows/release-notes-review.yml`

**Trigger Events:**
```yaml
on:
  pull_request:
    types: [opened, synchronize, labeled]
    paths:
      - 'release-notes/components/**/*.rst'
```

**Workflow Steps:**

1. **Check Label**
   - Verify PR has "release-notes" label
   - Exit gracefully if label not present

2. **Get Changed Files**
   - Use GitHub API to get list of changed files
   - Filter for `release-notes/components/**/*.rst`
   - Exit if no matching files found

3. **Setup Q CLI**
   - Install/configure Amazon Q CLI
   - Authenticate using GitHub secrets

4. **Load Guidelines**
   - Read `_ext/release-notes-context.md`
   - Prepare as context for Q CLI

5. **Review Each File**
   - For each changed RST file:
     - Read file content
     - Invoke Q CLI with prompt:
       ```
       Review the following release notes file against the guidelines provided.
       
       Guidelines: [content from release-notes-context.md]
       
       File: [filename]
       Content: [file content]
       
       Provide feedback using the review format specified in the guidelines.
       Focus on: customer visibility, documentation links, impact clarity, 
       specific conditions, and actionable information.
       ```
     - Capture Q CLI response

6. **Format Feedback**
   - Combine all file reviews into single comment
   - Format as markdown with sections per file
   - Include summary at top

7. **Post Comment**
   - Post formatted feedback as PR comment
   - Include link to guidelines
   - Tag PR author

### Q CLI Prompt Template

```markdown
You are reviewing release notes for the AWS Neuron SDK. Review the following 
file against the release notes writing guidelines.

GUIDELINES:
[Full content of _ext/release-notes-context.md]

FILE TO REVIEW: {filename}

CONTENT:
{file_content}

INSTRUCTIONS:
1. Review the content against all guidelines
2. Identify issues using the review format from the guidelines
3. For each issue, provide:
   - Issue number and title
   - Original text
   - Problem description
   - Phrasing problem (if applicable)
   - Example rewrite
   - Specific action items
4. If no issues found, state "No issues found - release notes meet guidelines"

Focus especially on:
- Customer-visible language (no internal code names)
- Documentation URLs for all new features
- Specific conditions (not vague language)
- Clear impact statements
- Proper categorization (breaking changes vs bug fixes)
- Migration guidance for breaking changes
```

### Comment Format Template

```markdown
## 🤖 Release Notes Review

This PR modifies {count} release notes file(s). Here's the automated review:

### Files Reviewed
- ✅ `release-notes/components/file1.rst` - {issue_count} issue(s)
- ✅ `release-notes/components/file2.rst` - No issues found

---

### 📝 Review Feedback

#### File: `release-notes/components/file1.rst`

[Q CLI feedback for file1]

---

#### File: `release-notes/components/file2.rst`

[Q CLI feedback for file2]

---

### 📚 Resources

- [Release Notes Writing Guidelines](_ext/release-notes-context.md)
- Need help? Tag @documentation-team

---

*This is an automated review. Please address the feedback and request human 
review when ready.*
```

## Implementation Notes

### GitHub Action Configuration

**Required Secrets:**
- `Q_CLI_TOKEN` or equivalent for Q CLI authentication

**Required Permissions:**
```yaml
permissions:
  contents: read
  pull-requests: write
```

**Environment:**
- Ubuntu latest runner
- Node.js 18+ (if using JavaScript action)
- Python 3.9+ (if using Python script)

### Q CLI Integration Options

**Option 1: Direct CLI Invocation**
```bash
q chat --prompt-file prompt.txt --context-file guidelines.md
```

**Option 2: Q CLI GitHub Action** (if available)
```yaml
- uses: aws/q-cli-action@v1
  with:
    prompt: ${{ steps.prepare.outputs.prompt }}
    context: ${{ steps.prepare.outputs.context }}
```

**Option 3: API Integration** (if Q provides API)
```python
import q_cli
response = q_cli.chat(prompt=prompt, context=guidelines)
```

## Testing Strategy

### Unit Tests
- Test file filtering logic
- Test prompt generation
- Test comment formatting

### Integration Tests
- Test with sample PR containing valid release notes
- Test with sample PR containing issues
- Test with PR without "release-notes" label
- Test with PR modifying non-component files

### Manual Testing
- Create test PR with intentional issues
- Verify action triggers correctly
- Verify feedback is accurate and helpful
- Verify comment formatting is readable

## Success Criteria

1. **Automation Works**: Action runs on 100% of labeled PRs
2. **Accurate Detection**: Action correctly identifies changed RST files
3. **Useful Feedback**: 80%+ of PR authors find feedback helpful
4. **No False Blocks**: Action never blocks valid PRs
5. **Performance**: Action completes within 5 minutes
6. **Reliability**: Action succeeds 95%+ of the time

## Future Enhancements

### Phase 2 (Optional)
- Support for reviewing other release notes files (not just components)
- Severity levels for issues (critical, warning, suggestion)
- Auto-fix suggestions as code suggestions
- Integration with PR review status
- Metrics dashboard for common issues

### Phase 3 (Optional)
- Pre-commit hook for local review
- VS Code extension for real-time feedback
- Training mode to help new contributors learn guidelines
- Historical analysis of release notes quality trends

## Dependencies

- GitHub Actions infrastructure
- Amazon Q CLI availability and access
- Repository write access for bot account
- `_ext/release-notes-context.md` guidelines file

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Q CLI unavailable | High | Graceful failure with manual review fallback |
| Q CLI rate limits | Medium | Implement retry logic and rate limiting |
| False positives | Medium | Continuous refinement of guidelines and prompts |
| Action performance | Low | Parallel processing and caching |
| Cost of Q CLI usage | Low | Monitor usage and set budget alerts |

## Rollout Plan

1. **Phase 1**: Implement basic action with manual trigger
2. **Phase 2**: Enable automatic trigger on label
3. **Phase 3**: Gather feedback and refine prompts
4. **Phase 4**: Expand to other release notes files if successful

## Maintenance

- **Owner**: Documentation team
- **Review Frequency**: Quarterly
- **Update Triggers**: 
  - Changes to release notes guidelines
  - Q CLI updates
  - User feedback on accuracy
  - GitHub Actions platform changes
