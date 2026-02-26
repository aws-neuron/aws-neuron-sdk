#!/usr/bin/env python3
"""
Validation script for news-and-blogs.yaml

This script validates the structure and content of article entries
to ensure they meet the required format before submission.

Usage:
    python validate_articles.py
"""

import sys
from pathlib import Path
from datetime import datetime
import re

try:
    import yaml
except ImportError:
    print("Error: PyYAML is required. Install with: pip install pyyaml")
    sys.exit(1)


VALID_CATEGORIES = {'blog', 'news', 'tutorial', 'case-study', 'benchmark'}
REQUIRED_FIELDS = {'title', 'url', 'description', 'author', 'date', 'category', 'locale', 'keywords'}
OPTIONAL_FIELDS = {'featured', 'author_url', 'icon'}
ALL_FIELDS = REQUIRED_FIELDS | OPTIONAL_FIELDS

# Valid locale codes
VALID_LOCALES = {
    'en-US', 'en-GB', 'en-CA', 'en-AU', 'en-NZ', 'en-IE', 'en-IN', 'en-SG', 'en-ZA',
    'ja-JP', 'zh-CN', 'zh-TW', 'zh-HK', 'ko-KR', 'th-TH', 'vi-VN', 'id-ID', 'ms-MY', 'fil-PH',
    'de-DE', 'fr-FR', 'es-ES', 'es-MX', 'es-AR', 'pt-BR', 'pt-PT', 'it-IT', 'nl-NL', 'pl-PL',
    'ru-RU', 'tr-TR', 'sv-SE', 'da-DK', 'no-NO', 'fi-FI', 'cs-CZ', 'hu-HU', 'ro-RO', 'el-GR',
    'uk-UA', 'ar-SA', 'ar-AE', 'ar-EG', 'he-IL', 'fa-IR', 'hi-IN', 'bn-BD', 'ur-PK', 'sw-KE'
}


def validate_url(url):
    """Validate URL format"""
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
        r'localhost|'  # localhost
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # or IP
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return url_pattern.match(url) is not None


def validate_date(date_str):
    """Validate date format (YYYY-MM-DD)"""
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False


def validate_article(article, index, section):
    """Validate a single article entry"""
    errors = []
    warnings = []
    
    # Check for required fields
    missing_fields = REQUIRED_FIELDS - set(article.keys())
    if missing_fields:
        errors.append(f"Missing required fields: {', '.join(missing_fields)}")
    
    # Check for unknown fields
    unknown_fields = set(article.keys()) - ALL_FIELDS
    if unknown_fields:
        warnings.append(f"Unknown fields (will be ignored): {', '.join(unknown_fields)}")
    
    # Validate title
    if 'title' in article:
        if not article['title'] or not isinstance(article['title'], str):
            errors.append("Title must be a non-empty string")
        elif len(article['title']) > 200:
            warnings.append(f"Title is very long ({len(article['title'])} chars). Consider shortening.")
    
    # Validate URL
    if 'url' in article:
        if not validate_url(article['url']):
            errors.append(f"Invalid URL format: {article['url']}")
    
    # Validate description
    if 'description' in article:
        if not article['description'] or not isinstance(article['description'], str):
            errors.append("Description must be a non-empty string")
        elif len(article['description']) < 20:
            warnings.append("Description is very short. Consider adding more detail.")
        elif len(article['description']) > 500:
            warnings.append(f"Description is very long ({len(article['description'])} chars). Consider shortening.")
    
    # Validate author
    if 'author' in article:
        if not article['author'] or not isinstance(article['author'], str):
            errors.append("Author must be a non-empty string")
    
    # Validate author_url (optional)
    if 'author_url' in article:
        if article['author_url'] and not validate_url(article['author_url']):
            errors.append(f"Invalid author_url format: {article['author_url']}")
    
    # Validate date
    if 'date' in article:
        if not validate_date(str(article['date'])):
            errors.append(f"Invalid date format: {article['date']}. Use YYYY-MM-DD")
        else:
            article_date = datetime.strptime(str(article['date']), '%Y-%m-%d')
            if article_date > datetime.now():
                warnings.append(f"Date is in the future: {article['date']}")
    
    # Validate category
    if 'category' in article:
        if article['category'] not in VALID_CATEGORIES:
            errors.append(f"Invalid category: {article['category']}. Must be one of: {', '.join(VALID_CATEGORIES)}")
    
    # Validate locale
    if 'locale' in article:
        if not isinstance(article['locale'], str):
            errors.append("Locale must be a string")
        elif article['locale'] not in VALID_LOCALES:
            warnings.append(f"Locale '{article['locale']}' not in standard list. Will display with 🌐 globe icon. Common locales: en-US, ja-JP, zh-CN, de-DE, fr-FR, es-ES, pt-BR, ko-KR")
    
    # Validate keywords
    if 'keywords' in article:
        if not isinstance(article['keywords'], list):
            errors.append("Keywords must be a list")
        elif len(article['keywords']) == 0:
            warnings.append("Keywords list is empty. Consider adding relevant keywords for better filtering")
        else:
            for i, keyword in enumerate(article['keywords']):
                if not isinstance(keyword, str):
                    errors.append(f"Keyword at index {i} must be a string")
                elif len(keyword.strip()) == 0:
                    warnings.append(f"Keyword at index {i} is empty or whitespace")
            if len(article['keywords']) > 10:
                warnings.append(f"Article has {len(article['keywords'])} keywords. Consider limiting to 5-10 most relevant keywords")
    
    # Validate featured
    if 'featured' in article:
        if not isinstance(article['featured'], bool):
            errors.append("Featured must be true or false (boolean)")
        if section == 'all_articles' and article['featured']:
            warnings.append("Article marked as featured but in all_articles section")
    
    # Validate icon (optional)
    if 'icon' in article:
        if not isinstance(article['icon'], str) or len(article['icon']) > 10:
            warnings.append("Icon should be a short string (emoji recommended)")
    
    return errors, warnings


def main():
    """Main validation function"""
    yaml_file = Path(__file__).parent / 'news-and-blogs.yaml'
    
    if not yaml_file.exists():
        print(f"❌ Error: {yaml_file} not found")
        return 1
    
    print(f"Validating {yaml_file}...\n")
    
    try:
        with open(yaml_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"❌ YAML Parse Error: {e}")
        return 1
    
    if not isinstance(data, dict):
        print("❌ Error: YAML file must contain a dictionary")
        return 1
    
    total_errors = 0
    total_warnings = 0
    
    # Validate featured_articles section
    if 'featured_articles' in data:
        print("📌 Validating featured_articles section...")
        if not isinstance(data['featured_articles'], list):
            print("❌ Error: featured_articles must be a list")
            total_errors += 1
        else:
            for i, article in enumerate(data['featured_articles'], 1):
                errors, warnings = validate_article(article, i, 'featured_articles')
                if errors or warnings:
                    print(f"\n  Article #{i}: {article.get('title', 'NO TITLE')}")
                    for error in errors:
                        print(f"    ❌ Error: {error}")
                        total_errors += 1
                    for warning in warnings:
                        print(f"    ⚠️  Warning: {warning}")
                        total_warnings += 1
        print()
    
    # Validate all_articles section
    if 'all_articles' in data:
        print("📚 Validating all_articles section...")
        if not isinstance(data['all_articles'], list):
            print("❌ Error: all_articles must be a list")
            total_errors += 1
        else:
            for i, article in enumerate(data['all_articles'], 1):
                errors, warnings = validate_article(article, i, 'all_articles')
                if errors or warnings:
                    print(f"\n  Article #{i}: {article.get('title', 'NO TITLE')}")
                    for error in errors:
                        print(f"    ❌ Error: {error}")
                        total_errors += 1
                    for warning in warnings:
                        print(f"    ⚠️  Warning: {warning}")
                        total_warnings += 1
        print()
    
    # Summary
    print("=" * 60)
    if total_errors == 0 and total_warnings == 0:
        print("✅ Validation passed! No errors or warnings found.")
        return 0
    else:
        print(f"Validation complete:")
        if total_errors > 0:
            print(f"  ❌ {total_errors} error(s) found - must be fixed")
        if total_warnings > 0:
            print(f"  ⚠️  {total_warnings} warning(s) found - should be reviewed")
        
        if total_errors > 0:
            print("\n❌ Validation FAILED - please fix errors before submitting")
            return 1
        else:
            print("\n✅ Validation PASSED - warnings are optional to fix")
            return 0


if __name__ == '__main__':
    sys.exit(main())
