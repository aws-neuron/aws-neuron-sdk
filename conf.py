# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.append(os.path.abspath("./_ext"))

# get environment variables

env_project_name = os.environ.get('GIT_PROJECT_NAME')
env_branch_name = os.environ.get('GIT_BRANCH_NAME')



# -- Project information -----------------------------------------------------

project = 'AWS Neuron'
copyright = '2021, Amazon Web Services'
author = 'AWS'
master_doc = 'index'

# The full version, including alpha/beta/rc tags
#release = '1.8.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinxcontrib.contentui','nbsphinx','sphinx.ext.extlinks','archive']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build','**.ipynb_checkpoints']


# nbsphinx_allow_errors = True
nbsphinx_execute = 'never'

html_logo = 'images/NeuronStandalone_white_xsmall.png'

# -- more options -------------------------------------------------

# set project name
if env_project_name != None:
    project_name = env_project_name
else:
    project_name = "aws-neuron-sdk"

# set branch name
if env_branch_name != None:
    branch_name = env_branch_name
else:
    branch_name = "master"


projectblob = project_name+'/blob/'+branch_name
projecttree = project_name+'/tree/'+branch_name

#projectblob = "private-aws-neuron-sdk-staging/blob/aws-maen-rtd"
#projecttree = "private-aws-neuron-sdk-staging/tree/aws-maen-rtd"

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
            }


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'navigation_depth': 3
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
#html_static_path = ['_static']

