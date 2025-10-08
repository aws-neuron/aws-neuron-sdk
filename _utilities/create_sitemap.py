# v1.0 by dougeric 2025-09-30
# Script to create sitemap.xml for Sphinx-generated docs; must be run at the root of the docs repo with venv

import os
from pathlib import Path
from datetime import datetime

def create_sitemap(root_dir, base_url):
    """
    This function generates a sitemap.xml file for the given root directory and base URL.
    It recursively scans all .rst files in the root directory, excluding those in directories
    starting with "_". For each .rst file, it calculates the last modification time, converts
    the .rst path to the corresponding HTML path, and adds a <url> entry to the sitemap in the
    format required by Google Search Console.
    """
    sitemap = ['<?xml version="1.0" encoding="UTF-8"?>',
               '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">']
    
    for path in Path(root_dir).rglob('*.rst'):
        # Skip directories starting with "_"
        if any(part.startswith('_') for part in path.parts):
            continue
            
        # Convert .rst path to expected html path
        rel_path = path.relative_to(root_dir)
        html_path = str(rel_path).replace('.rst', '.html')
        
        # Get file modification time
        mod_time = datetime.fromtimestamp(os.path.getmtime(path))
        
        sitemap.append(f'  <url>')
        sitemap.append(f'    <loc>{base_url}/{html_path}</loc>')
        sitemap.append(f'    <lastmod>{mod_time.strftime("%Y-%m-%d")}</lastmod>')
        sitemap.append(f'  </url>')
    
    sitemap.append('</urlset>')
    return '\n'.join(sitemap)

# Call the function and write the result to sitemap.xml
sitemap_content = create_sitemap('./', 'https://awsdocs-neuron.readthedocs-hosted.com/en/latest')
with open('sitemap.xml', 'w') as f:
    f.write(sitemap_content)
print("\nsitemap.xml has been created.\n")