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
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'local_documenter',
    'archive',
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

napoleon_google_docstring = True

# -- more options -------------------------------------------------


projectblob = project_name+'/blob/'+branch_name
projecttree = project_name+'/tree/'+branch_name

extlinks = {
            'mxnet-neuron': ('https://github.com/aws-neuron/'+projectblob+'/neuron-guide/neuron-frameworks/mxnet-neuron/%s', '')
            ,'pytorch-neuron': ('https://github.com/aws-neuron/'+projectblob+'/neuron-guide/neuron-frameworks/pytorch-neuron/%s', '')
            ,'tensorflow-neuron': ('https://github.com/aws-neuron/'+projectblob+'/neuron-guide/neuron-frameworks/tensorflow-neuron/%s', '')
            ,'neuron-deploy': ('https://github.com/aws-neuron/'+projectblob+'/neuron-deploy/%s', '')
            ,'neuron-tools-tree': ('https://github.com/aws-neuron/'+projecttree+'/neuron-guide/neuron-tools/%s', '')
            ,'mxnet-neuron-src': ('https://github.com/aws-neuron/'+projectblob+'/src/examples/mxnet/%s', '')
            ,'pytorch-neuron-src': ('https://github.com/aws-neuron/'+projectblob+'/src/examples/pytorch/%s', '')
            ,'tensorflow-neuron-src': ('https://github.com/aws-neuron/'+projectblob+'/src/examples/tensorflow/%s', '')
            ,'neuron-gatherinfor-src': ('https://github.com/aws-neuron/'+projectblob+'/src/examples/neuron-gatherinfo/%s', '')
            ,'neuron-monitor-src': ('https://github.com/aws-neuron/'+projectblob+'/src/examples/neuron-monitor/%s', '')
            ,'compile-pt': ('https://github.com/aws-neuron/'+projectblob+'/src/benchmark/pytorch/%s_compile.py', '')
            ,'benchmark-pt': ('https://github.com/aws-neuron/'+projectblob+'/src/benchmark/pytorch/%s_benchmark.py', '')
            ,'llama-sample': ('https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/transformers-neuronx/inference/%s.ipynb', '')
            ,'github':(f'https://github.com/aws-neuron/{projectblob}', '')
            }


intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'torch': ('https://pytorch.org/docs/master/', None),
    'transformers': ('https://huggingface.co/docs/transformers/master/en/', None),
}


# -- Options for Theme  -------------------------------------------------

#top_banner_message="<a class='reference internal' style='color:white;' href='https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/announcements/neuron2.x/dlami-pytorch-introduce.html'>  Deep Learning AMI Neuron PyTorch is now available! </a> <br>  <a class='reference internal' style='color:white;'  href='https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/announcements/neuron2.x/sm-training-trn1-introduce.html'> Amazon Sagemaker now supports training jobs on Trn1! </a>"

#top_banner_message="<span>&#9888;</span><a class='reference internal' style='color:white;' href='https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/setup/setup-troubleshooting.html#gpg-key-update'>  Neuron repository GPG key for Ubuntu installation has expired, see instructions how to update! </a>"


top_banner_message="Neuron 2.13.2 is released! check <a class='reference internal' style='color:white;' href='https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/index.html#latest-neuron-release'> What's New  </a> and <a class='reference internal' style='color:white;' href='https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/announcements/index.html'> Announcements  </a>"


html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/aws-neuron/" + project_name ,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button" : True,
    "use_fullscreen_button" : True,
    "use_edit_page_button": True,
    "home_page_in_toc": False,
    "repository_branch" : branch_name,
    "announcement": top_banner_message,
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

# Exclude private github from linkcheck. Readthedocs only exposes the ssh-agent to the 'checkout' build step, which is too early for the linkchecker to run.
linkcheck_ignore = [r'http://localhost:\d+/',r'https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/announcements/neuron2.x/dlami-pytorch-introduce.html' ,r'https://github\.com/aws-neuron/private-aws-neuron-sdk-staging/',r'https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/announcements/neuron2.x/dlami-pytorch-introduce.html', r'https://awsdocs-neuron-staging.readthedocs-hosted.com/en/latest/frameworks/tensorflow/tensorflow-neuronx/setup/tensorflow-neuronx-install.html#install-tensorflow-neuronx',r'https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx#inference',r'https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx#training', r'https://github.com/aws/deep-learning-containers/blob/master/available_images.md#neuron-containers',r'https://github.com/aws-neuron/aws-neuron-sagemaker-samples/tree/master/inference/inf2-bert-on-sagemaker'
,r'https://github.com/awslabs/multi-model-server/blob/master/docs/management_api.md',r'https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/training/dp_bert_hf_pretrain/run_dp_bert_large_hf_pretrain_bf16_s128.sh',r' https://github.com/pytorch/xla/blob/master/test/test_train_mp_mnist.py',r'https://github.com/pytorch/xla/blob/v1.10.0/TROUBLESHOOTING.md'
,r'https://github.com/tensorflow/docs/blob/master/site/en/r1/guide/saved_model.md',r'https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/g3doc/index.md',r'https://github.com/pytorch/xla/blob/master/test/test_train_mp_mnist.py',r'https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/transformers-neuronx/inference/meta-llama-2-13b-sampling.ipynb',r'https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/examples/pytorch/torch-neuronx/t5-inference-tutorial.ipynb',r'https://github.com/aws-neuron/aws-neuron-parallelcluster-samples/blob/master/examples/jobs/neuronx-nemo-megatron-llamav2-job.md']
linkcheck_exclude_documents = [r'src/examples/.*', 'general/announcements/neuron1.x/announcements', r'release-notes/.*',r'containers/.*',r'general/.*']
nitpicky = True
