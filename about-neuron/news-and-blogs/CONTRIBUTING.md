# Contributing to AWS Neuron News and Blogs

Thank you for your interest in sharing content about AWS Neuron, Trainium, and Inferentia! This page collects external articles, blog posts, tutorials, and news to help the community discover valuable content.

## How to Add Your Article

### Quick Steps

1. **Fork the repository** on GitHub
2. **Edit the data file**: `about-neuron/news-and-blogs/news-and-blogs.yaml`
3. **Add your article** following the format below
4. **Submit a pull request** with your changes

### Article Entry Format

Add your article to the appropriate section in `news-and-blogs.yaml`:

```yaml
- title: "Your Article Title"
  url: "https://example.com/your-article"
  description: "A brief 1-2 sentence description of your article content."
  author: "Your Name or Organization"
  author_url: "https://your-website.com"  # Optional for featured articles
  date: "YYYY-MM-DD"  # Publication date
  category: "blog"  # Options: blog, news, tutorial, case-study, benchmark
  locale: "en-US"  # Language/region code (e.g., en-US, ja-JP, zh-CN, de-DE, fr-FR)
  featured: false  # Set to true only if approved by AWS Neuron team
  icon: "📝"  # Optional emoji icon for featured articles
```

### Sections

- **`featured_articles`**: Highlighted content (requires AWS Neuron team approval)
- **`all_articles`**: All community and official content

### Categories

Choose the most appropriate category for your content:

- **`blog`**: Technical blog posts and articles
- **`news`**: News announcements and press releases
- **`tutorial`**: Step-by-step guides and how-tos
- **`case-study`**: Customer success stories and use cases
- **`benchmark`**: Performance benchmarks and comparisons

### Locale Codes

Specify the language and region of your article using standard locale codes:

**Common Locales:**
- `en-US` - English (United States) 🇺🇸
- `en-GB` - English (United Kingdom) 🇬🇧
- `ja-JP` - Japanese 🇯🇵
- `zh-CN` - Chinese (Simplified) 🇨🇳
- `zh-TW` - Chinese (Traditional) 🇹🇼
- `ko-KR` - Korean 🇰🇷
- `de-DE` - German 🇩🇪
- `fr-FR` - French 🇫🇷
- `es-ES` - Spanish (Spain) 🇪🇸
- `es-MX` - Spanish (Mexico) 🇲🇽
- `pt-BR` - Portuguese (Brazil) 🇧🇷
- `it-IT` - Italian 🇮🇹
- `nl-NL` - Dutch 🇳🇱
- `ru-RU` - Russian 🇷🇺
- `ar-SA` - Arabic 🇸🇦
- `hi-IN` - Hindi 🇮🇳

A flag emoji will be automatically displayed next to your article based on the locale. If your locale isn't in the list, a 🌐 globe icon will be shown.

### Example Entry

```yaml
all_articles:
  - title: "Building Large Language Models on AWS Trainium"
    url: "https://example.com/llm-trainium-guide"
    description: "A comprehensive guide to training and deploying LLMs using AWS Trainium instances with practical code examples."
    author: "Jane Developer"
    date: "2026-01-15"
    category: "tutorial"
    locale: "en-US"
    featured: false
```

### Guidelines

1. **Content must be relevant** to AWS Neuron, Trainium, or Inferentia
2. **Provide accurate information** - ensure URLs work and descriptions are clear
3. **Use proper formatting** - follow YAML syntax exactly
4. **One article per pull request** - makes review easier
5. **Include context** in your PR description about why this content is valuable

### Featured Articles

To request your article be featured:

1. Add it to `all_articles` first with `featured: false`
2. In your pull request, explain why it should be featured
3. AWS Neuron team will review and may promote it to `featured_articles`

Featured articles should be:
- High-quality, in-depth content
- Particularly valuable to the community
- Recent (typically within the last 6 months)

### Review Process

1. Submit your pull request
2. AWS Neuron team will review within 5-7 business days
3. May request changes or clarifications
4. Once approved, your article will appear on the next documentation build

### Questions?

- Open an issue in the repository
- Contact your AWS Neuron support representative
- Email: aws-neuron-support@amazon.com

## Content Guidelines

### What to Include

✅ Technical tutorials and guides  
✅ Performance benchmarks and analysis  
✅ Customer success stories  
✅ Integration guides with other tools  
✅ Best practices and optimization tips  
✅ Conference talks and presentations  
✅ Research papers using Neuron/Trainium/Inferentia  

### What Not to Include

❌ Marketing content without technical substance  
❌ Broken or paywalled links  
❌ Content unrelated to AWS Neuron ecosystem  
❌ Duplicate submissions  
❌ Self-promotional content without value to community  

## Technical Details

This page uses:
- **Sphinx** with `sphinxcontrib.datatemplates` extension
- **YAML** for data storage
- **Jinja2** templates for rendering
- **sphinx-design** for grid layouts

The system is fully static - no backend required. All content is rendered at build time.

## License

By contributing, you agree that your contributions will be licensed under the same license as this project. See the repository LICENSE files for details.
