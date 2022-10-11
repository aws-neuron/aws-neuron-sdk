# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
import datetime

sys.path.append(os.path.abspath("./_ext"))
#sys.path.append(os.path.abspath("./_static"))

# get environment variables

project_name = ""
branch_name = ""

if os.environ.get('READTHEDOCS') == "True":
    env_branch_name = os.environ.get('READTHEDOCS_VERSION_NAME')
    branch_name = env_branch_name
    if branch_name == "latest":
        branch_name = "master"       
    if os.environ.get('READTHEDOCS_PROJECT') == "awsdocs-neuron":
        env_project_name = "aws-neuron-sdk"
        project_name = env_project_name
    elif os.environ.get('READTHEDOCS_PROJECT') == "awsdocs-neuron-staging":
        env_project_name = "private-aws-neuron-sdk-staging"
        project_name = env_project_name
else:
    env_project_name = os.environ.get('GIT_PROJECT_NAME')
    env_branch_name = os.environ.get('GIT_BRANCH_NAME')

    # set project name
    if env_project_name != None:
        project_name = env_project_name
    else:
        project_name = "aws-neuron-sdk"

    # set branch name
    if env_branch_name != None:
        branch_name = env_branch_name
        if branch_name == "latest":
            branch_name = "master"
    else:
        branch_name = "master"

# -- Project information -----------------------------------------------------

project = 'AWS Neuron'
copyright = "{}, Amazon.com".format(datetime.datetime.now().year)
author = 'AWS'
master_doc = 'index'
html_title = 'AWS Neuron Documentation'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinxcontrib.contentui',
    'nbsphinx',
    'sphinx.ext.extlinks',
    'sphinx.ext.intersphinx',
    'sphinx_plotly_directive',
    'df_tables',
    'sphinxcontrib.programoutput',
    'neuron_tag',
    'sphinx_design',
    'ablog',
    'sphinx.ext.viewcode',
]


html_sidebars = {
   'general/announcements/index': ["recentposts.html"]
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build','**.ipynb_checkpoints','.venv']
html_extra_path = ['static']

# nbsphinx_allow_errors = True
nbsphinx_execute = 'never'

html_logo = 'images/Site-Merch_Neuron-ML-SDK_Editorial.png'

# -- more options -------------------------------------------------


projectblob = project_name+'/blob/'+branch_name
projecttree = project_name+'/tree/'+branch_name

extlinks = {
            'mxnet-neuron': ('https://github.com/aws/'+projectblob+'/neuron-guide/neuron-frameworks/mxnet-neuron/%s', '')
            ,'pytorch-neuron': ('https://github.com/aws/'+projectblob+'/neuron-guide/neuron-frameworks/pytorch-neuron/%s', '')
            ,'tensorflow-neuron': ('https://github.com/aws/'+projectblob+'/neuron-guide/neuron-frameworks/tensorflow-neuron/%s', '')
            ,'neuron-deploy': ('https://github.com/aws/'+projectblob+'/neuron-deploy/%s', '')
            ,'neuron-tools-tree': ('https://github.com/aws/'+projecttree+'/neuron-guide/neuron-tools/%s', '')
            ,'mxnet-neuron-src': ('https://github.com/aws/'+projectblob+'/src/examples/mxnet/%s', '')
            ,'pytorch-neuron-src': ('https://github.com/aws/'+projectblob+'/src/examples/pytorch/%s', '')
            ,'tensorflow-neuron-src': ('https://github.com/aws/'+projectblob+'/src/examples/tensorflow/%s', '')
            ,'neuron-gatherinfor-src': ('https://github.com/aws/'+projectblob+'/src/examples/neuron-gatherinfo/%s', '')
            ,'neuron-monitor-src': ('https://github.com/aws/'+projectblob+'/src/examples/neuron-monitor/%s', '')
            ,'compile-pt': ('https://github.com/aws/'+projectblob+'/src/benchmark/pytorch/%s_compile.py', '')
            ,'benchmark-pt': ('https://github.com/aws/'+projectblob+'/src/benchmark/pytorch/%s_benchmark.py', '')
            }


intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'torch': ('https://pytorch.org/docs/master/', None),
}


# -- Options for Theme  -------------------------------------------------


html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/aws-neuron/" + project_name ,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button" : True,
    "use_fullscreen_button" : True,
    "use_edit_page_button": True,
    "home_page_in_toc": False,
    "repository_branch" : branch_name



}


# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
#html_theme = 'sphinx_rtd_theme'

#html_theme_options = {
#       
#    'navigation_depth': 3
#}



#html_theme = "pydata_sphinx_theme"
#html_theme_options = {
#   "use_edit_page_button": True,
#}

#html_context = {
#    "github_url": "https://github.com", 
#    "github_user": "aws-neuron",
#    "github_repo": "private-aws-neuron-sdk-staging",
#    "github_version": "master",
#    "doc_path": "/",
#}

# -- Options for HTML output -------------------------------------------------

html_css_files = ['css/custom.css','styles/sphinx-book-theme.css']

#def setup(app):
#   app.add_css_file('css/custom.css')

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

plotly_include_source = False
plotly_html_show_source_link = False
plotly_html_show_formats = False
plotly_include_directive_source = False



# -- ABlog config -------------------------------------------------
blog_path = "general/announcements/index"
blog_post_pattern = "general/appnotes/*.rst"
blog_feed_length = 5
fontawesome_included = True
post_show_prev_next = False
post_auto_image = 1
post_auto_excerpt = 2
execution_show_tb = "READTHEDOCS" in os.environ


# --- for neuron-tag directive ---

rst_prolog = """

.. neuron-tag::


"""

rst_epilog = """

.. neuron-tag::

"""
