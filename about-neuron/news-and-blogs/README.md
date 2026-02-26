# AWS Neuron News and Blogs System

This directory contains a dynamic, community-driven news and blogs page for AWS Neuron, Trainium, and Inferentia content.

## Overview

The system allows external contributors to add links to relevant articles, blog posts, and news through a simple YAML data file, without requiring any backend infrastructure.

## Architecture

```
about-neuron/news-and-blogs/
├── index.rst                    # Main page (uses datatemplate directives)
├── news-and-blogs.yaml          # Data file with all article metadata
├── featured-articles.tmpl       # Jinja2 template for featured section
├── all-articles.tmpl            # Jinja2 template for all articles section
├── CONTRIBUTING.md              # Contribution guidelines
└── README.md                    # This file
```

## How It Works

1. **Data Storage**: Article metadata is stored in `news-and-blogs.yaml`
2. **Templating**: Jinja2 templates (`*.tmpl`) define how articles are rendered
3. **Rendering**: Sphinx's `datatemplates` extension processes the YAML and templates at build time
4. **Output**: Static HTML with grid cards using `sphinx-design`

## Key Features

- ✅ **No backend required** - fully static site generation
- ✅ **Easy contributions** - edit a YAML file and submit a PR
- ✅ **Version controlled** - all changes tracked in Git
- ✅ **Automated rendering** - Sphinx handles everything at build time
- ✅ **Responsive design** - uses sphinx-design grid system
- ✅ **Maintainable** - clear separation of data, templates, and content

## Adding New Articles

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed instructions.

Quick example:

```yaml
all_articles:
  - title: "My Article Title"
    url: "https://example.com/article"
    description: "Brief description"
    author: "Author Name"
    date: "2026-01-15"
    category: "blog"
    featured: false
```

## Modifying Templates

Templates use Jinja2 syntax and have access to the YAML data structure.

### Featured Articles Template (`featured-articles.tmpl`)

Renders articles from the `featured_articles` section with:
- Large cards with borders
- Icons and bold titles
- Author attribution with links
- Publication dates

### All Articles Template (`all-articles.tmpl`)

Renders articles from the `all_articles` section with:
- 2-column grid on desktop, 1-column on mobile
- Simple card layout
- Title and description

## Customization

### Adding New Fields

1. Add field to YAML entries:
   ```yaml
   - title: "Article"
     new_field: "value"
   ```

2. Update template to use it:
   ```jinja
   {{ article.new_field }}
   ```

### Changing Layout

Edit the grid directive in templates:
```rst
.. grid:: 1 1 2 3  # 1 col mobile, 1 tablet, 2 desktop, 3 wide
   :gutter: 2
```

### Adding Filters/Sorting

You can add Jinja2 filters in templates:

```jinja
{% for article in all_articles | sort(attribute='date', reverse=True) %}
  {# Sorted by date, newest first #}
{% endfor %}
```

## Dependencies

Required Sphinx extensions (already in `conf.py`):
- `sphinxcontrib.datatemplates` - YAML data processing
- `sphinx_design` - Grid card layouts

## Testing Locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Build documentation:
   ```bash
   sphinx-build -b html . _build/html
   ```

3. View the page:
   ```bash
   open _build/html/about-neuron/news-and-blogs/index.html
   ```

## Troubleshooting

### Template Not Found Error

Ensure templates are in the same directory as `index.rst` or add the directory to `templates_path` in `conf.py`.

### YAML Parse Error

Validate your YAML:
```bash
python -c "import yaml; yaml.safe_load(open('news-and-blogs.yaml'))"
```

### Articles Not Rendering

Check that:
1. YAML file is in the same directory as `index.rst`
2. Template files exist and have correct names
3. YAML structure matches template expectations

## Future Enhancements

Possible improvements:
- Add category filtering/grouping
- Add search functionality
- Add RSS feed generation
- Add automatic link checking
- Add article metadata validation
- Sort by date automatically
- Add pagination for large lists

## Support

For questions or issues:
- Open a GitHub issue
- Contact AWS Neuron support team
- See main repository CONTRIBUTING.md
