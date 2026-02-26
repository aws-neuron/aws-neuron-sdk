# Jira Integration Design for News & Blogs

## Overview

This document describes a design for populating the `news-and-blogs.yaml` file from Jira tickets, allowing contributors to submit article links via Jira instead of direct pull requests.

## Design Goals

1. **Simple for contributors**: Submit a Jira ticket with article metadata
2. **Automated**: Minimal manual intervention to add articles to YAML
3. **Quality control**: Review process before articles appear on the site
4. **Compatible**: Works with existing Sphinx build process
5. **No backend required**: Leverages existing CI/CD infrastructure

## Architecture

### Option 1: GitHub Actions + Jira API (Recommended)

```
Jira Ticket Created → GitHub Action Triggered → Parse Ticket → Update YAML → Create PR
```

**Components:**

1. **Jira Ticket Template**: Custom issue type "News Article Submission"
2. **GitHub Action**: Runs on schedule (e.g., hourly) or webhook
3. **Python Script**: Fetches approved tickets, generates YAML entries
4. **Automated PR**: Creates pull request with new articles

**Workflow:**

```yaml
# .github/workflows/sync-jira-articles.yml
name: Sync Jira Articles to YAML

on:
  schedule:
    - cron: '0 */6 * * *'  # Every 6 hours
  workflow_dispatch:  # Manual trigger

jobs:
  sync-articles:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: pip install jira pyyaml
      
      - name: Fetch and process Jira tickets
        env:
          JIRA_URL: ${{ secrets.JIRA_URL }}
          JIRA_USER: ${{ secrets.JIRA_USER }}
          JIRA_TOKEN: ${{ secrets.JIRA_TOKEN }}
        run: python scripts/sync_jira_articles.py
      
      - name: Create Pull Request
        uses: peter-evans/create-pull-request@v5
        with:
          commit-message: 'Add articles from Jira'
          title: 'Add news articles from Jira submissions'
          body: 'Automated PR from Jira article submissions'
          branch: jira-articles-sync
```

### Option 2: Jira Automation + Webhook

```
Jira Ticket Approved → Webhook to GitHub → GitHub Action → Update YAML → Create PR
```

**Advantages:**
- Real-time updates when tickets are approved
- No polling required
- More efficient

**Setup:**
1. Configure Jira Automation rule
2. Trigger on status change to "Approved"
3. Send webhook to GitHub repository dispatch endpoint

### Option 3: Manual Script (Simplest)

```
Developer runs script → Fetches approved tickets → Updates YAML → Commits changes
```

**Use case:** Lower volume, manual review preferred

## Jira Ticket Structure

### Custom Fields Required

```
Issue Type: News Article Submission

Fields:
- Article Title (text, required)
- Article URL (URL, required)
- Description (text area, required)
- Author Name (text, required)
- Author URL (URL, optional)
- Publication Date (date, required)
- Category (dropdown: blog|news|tutorial|case-study|benchmark)
- Locale (dropdown: en-US|ja-JP|zh-CN|ko-KR|de-DE|fr-FR|es-ES|pt-BR)
- Keywords (labels or multi-select)
- Featured (checkbox)
- Icon (text, optional, for featured articles)

Status Workflow:
- Submitted → Under Review → Approved → Published → Rejected
```

### Example Jira Ticket

```
Title: Add Karakuri AWS Trainium Tutorial

Fields:
- Article Title: AWS Trainium: 50 Exercises
- Article URL: https://zenn.dev/karakuri_blog/articles/5ccedeee1beb08
- Description: Learn how to build LLMs for Trainium accelerators...
- Author Name: Karakuri
- Author URL: https://about.karakuri.ai/
- Publication Date: 2026-02-19
- Category: tutorial
- Locale: ja-JP
- Keywords: trainium, llm, training, tutorial
- Featured: Yes
- Icon: 🚀
```

## Implementation Script

### `scripts/sync_jira_articles.py`

```python
#!/usr/bin/env python3
"""
Sync approved Jira article submissions to news-and-blogs.yaml
"""

import os
import yaml
from jira import JIRA
from datetime import datetime

# Configuration
JIRA_URL = os.environ.get('JIRA_URL')
JIRA_USER = os.environ.get('JIRA_USER')
JIRA_TOKEN = os.environ.get('JIRA_TOKEN')
YAML_FILE = 'about-neuron/news-and-blogs/news-and-blogs.yaml'

# JQL to find approved, unpublished articles
JQL_QUERY = 'project = NEURON AND issuetype = "News Article Submission" AND status = "Approved" AND labels != "published"'

def connect_jira():
    """Connect to Jira instance"""
    return JIRA(server=JIRA_URL, basic_auth=(JIRA_USER, JIRA_TOKEN))

def fetch_approved_articles(jira):
    """Fetch approved article submissions from Jira"""
    issues = jira.search_issues(JQL_QUERY, maxResults=100)
    articles = []
    
    for issue in issues:
        article = {
            'title': issue.fields.customfield_10001,  # Article Title
            'url': issue.fields.customfield_10002,     # Article URL
            'description': issue.fields.customfield_10003,  # Description
            'author': issue.fields.customfield_10004,  # Author Name
            'date': issue.fields.customfield_10006,    # Publication Date
            'category': issue.fields.customfield_10007.value,  # Category
            'locale': issue.fields.customfield_10008.value,    # Locale
            'keywords': [label.name for label in issue.fields.labels if label.name != 'published'],
            'featured': bool(issue.fields.customfield_10009),  # Featured checkbox
        }
        
        # Optional fields
        if issue.fields.customfield_10005:  # Author URL
            article['author_url'] = issue.fields.customfield_10005
        
        if issue.fields.customfield_10010:  # Icon (for featured)
            article['icon'] = issue.fields.customfield_10010
        
        articles.append({
            'article': article,
            'issue_key': issue.key
        })
    
    return articles

def load_yaml():
    """Load existing YAML file"""
    with open(YAML_FILE, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def article_exists(data, url):
    """Check if article URL already exists in YAML"""
    all_urls = [a['url'] for a in data.get('featured_articles', [])]
    all_urls.extend([a['url'] for a in data.get('all_articles', [])])
    return url in all_urls

def add_articles_to_yaml(data, new_articles):
    """Add new articles to appropriate sections"""
    added_keys = []
    
    for item in new_articles:
        article = item['article']
        
        # Skip if already exists
        if article_exists(data, article['url']):
            print(f"Skipping duplicate: {article['title']}")
            continue
        
        # Add to appropriate section
        if article.get('featured', False):
            data['featured_articles'].append(article)
        else:
            data['all_articles'].append(article)
        
        added_keys.append(item['issue_key'])
        print(f"Added: {article['title']} ({item['issue_key']})")
    
    return added_keys

def save_yaml(data):
    """Save updated YAML file"""
    with open(YAML_FILE, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

def mark_as_published(jira, issue_keys):
    """Add 'published' label to Jira tickets and transition to Published status"""
    for key in issue_keys:
        issue = jira.issue(key)
        
        # Add published label
        labels = issue.fields.labels
        if 'published' not in labels:
            labels.append('published')
            issue.update(fields={'labels': labels})
        
        # Transition to Published status (adjust transition ID as needed)
        try:
            jira.transition_issue(issue, 'Published')
        except Exception as e:
            print(f"Could not transition {key}: {e}")

def main():
    print("Connecting to Jira...")
    jira = connect_jira()
    
    print("Fetching approved articles...")
    new_articles = fetch_approved_articles(jira)
    
    if not new_articles:
        print("No new articles to add.")
        return
    
    print(f"Found {len(new_articles)} approved articles")
    
    print("Loading existing YAML...")
    data = load_yaml()
    
    print("Adding articles to YAML...")
    added_keys = add_articles_to_yaml(data, new_articles)
    
    if added_keys:
        print("Saving YAML...")
        save_yaml(data)
        
        print("Marking Jira tickets as published...")
        mark_as_published(jira, added_keys)
        
        print(f"Successfully added {len(added_keys)} articles!")
    else:
        print("No new articles added (all were duplicates)")

if __name__ == '__main__':
    main()
```

## Setup Instructions

### 1. Configure Jira

1. Create custom issue type "News Article Submission"
2. Add custom fields (see structure above)
3. Configure workflow: Submitted → Under Review → Approved → Published
4. Create Jira API token for automation user

### 2. Configure GitHub Secrets

Add these secrets to your GitHub repository:

```
JIRA_URL: https://your-company.atlassian.net
JIRA_USER: automation@your-company.com
JIRA_TOKEN: <api-token>
```

### 3. Add GitHub Action

Create `.github/workflows/sync-jira-articles.yml` with the workflow above.

### 4. Install Dependencies

Add to `requirements.txt`:
```
jira==3.5.0
PyYAML==6.0
```

### 5. Test

1. Create a test Jira ticket
2. Approve it
3. Run workflow manually: Actions → Sync Jira Articles → Run workflow
4. Verify PR is created with new article

## Alternative: Simpler Webhook Approach

If you want something lighter without Jira API polling:

### Jira Automation Rule

```
Trigger: Issue transitioned to "Approved"
Condition: Issue type = "News Article Submission"
Action: Send web request

URL: https://api.github.com/repos/aws-neuron/aws-neuron-sdk/dispatches
Method: POST
Headers:
  Authorization: Bearer ${GITHUB_TOKEN}
  Accept: application/vnd.github.v3+json
Body:
{
  "event_type": "jira-article-approved",
  "client_payload": {
    "issue_key": "{{issue.key}}",
    "title": "{{issue.customfield_10001}}",
    "url": "{{issue.customfield_10002}}",
    "description": "{{issue.customfield_10003}}",
    "author": "{{issue.customfield_10004}}",
    "date": "{{issue.customfield_10006}}",
    "category": "{{issue.customfield_10007}}",
    "locale": "{{issue.customfield_10008}}"
  }
}
```

Then GitHub Action receives webhook and processes directly without Jira API calls.

## Maintenance

### Regular Tasks

1. **Monitor failed syncs**: Check GitHub Action logs
2. **Review PRs**: Automated PRs should still be reviewed before merge
3. **Clean up Jira**: Archive old Published tickets
4. **Update mappings**: If custom field IDs change, update script

### Troubleshooting

**Articles not syncing:**
- Check Jira API credentials
- Verify custom field IDs match
- Check JQL query returns expected tickets

**Duplicate articles:**
- Script checks URL before adding
- Manually remove duplicates from YAML if needed

**Formatting issues:**
- Validate YAML after sync: `python -m yaml about-neuron/news-and-blogs/news-and-blogs.yaml`
- Check for special characters in descriptions

## Security Considerations

1. **API Tokens**: Store in GitHub Secrets, never commit
2. **Permissions**: Use dedicated Jira service account with minimal permissions
3. **Validation**: Sanitize all input from Jira before adding to YAML
4. **Review**: Always review automated PRs before merging

## Cost & Complexity

| Approach | Setup Time | Maintenance | Cost |
|----------|-----------|-------------|------|
| GitHub Actions + Jira API | 4-6 hours | Low | Free (GitHub Actions) |
| Webhook + GitHub Actions | 2-3 hours | Very Low | Free |
| Manual Script | 1-2 hours | Medium | Free |

## Recommendation

**For production use**: Start with **Option 3 (Manual Script)** to validate the workflow, then upgrade to **Option 1 (GitHub Actions)** once the process is proven and volume increases.

**For high volume**: Use **Option 2 (Webhook)** for real-time updates.

## Future Enhancements

1. **Validation**: Add URL validation, duplicate detection in Jira
2. **Preview**: Generate preview of how article will appear
3. **Scheduling**: Support future publication dates
4. **Analytics**: Track article submissions and approval rates
5. **Notifications**: Notify submitters when articles are published
6. **Bulk import**: Support CSV upload for multiple articles
