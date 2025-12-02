"""Sphinx configuration."""

import datetime
import os
import shutil

from amazon_doc_utils import brazil_info

# Get metadata from brazil
brazil_version, intersphinx_factory = brazil_info.get(
    [brazil_info.PackageVersion, brazil_info.IntersphinxFactory]
)


def run_apidoc(app):
    """Generate doc stubs using sphinx-apidoc."""
    module_dir = os.path.join(app.srcdir, "../src/")
    output_dir = os.path.join(app.srcdir, "_apidoc")
    excludes = []

    # Ensure that any stale apidoc files are cleaned up first.
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    cmd = [
        "--separate",
        "--module-first",
        "--doc-project=API Reference",
        "-o",
        output_dir,
        module_dir,
    ]
    cmd.extend(excludes)

    try:
        from sphinx.ext import apidoc  # Sphinx >= 1.7

        apidoc.main(cmd)
    except ImportError:
        from sphinx import apidoc  # Sphinx < 1.7

        cmd.insert(0, apidoc.__file__)
        apidoc.main(cmd)


def setup(app):
    """Register our sphinx-apidoc hook."""
    app.connect("builder-inited", run_apidoc)


# Sphinx configuration below.
project = brazil_version.name
version = brazil_version.mv
release = brazil_version.full_version
copyright = "{}, Amazon.com".format(datetime.datetime.now().year)

intersphinx_mapping = intersphinx_factory.get_mapping()

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
]

source_suffix = ".rst"
master_doc = "index"

autoclass_content = "class"
autodoc_member_order = "bysource"
default_role = "py:obj"

html_theme = "haiku"
htmlhelp_basename = "{}doc".format(project)

napoleon_use_rtype = False
