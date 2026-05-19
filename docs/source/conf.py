# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import re
import sys
sys.path.insert(0, os.path.abspath('../../src'))
from docutils import nodes

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'HyFI'
copyright = '2025, Sandro Truttmann'
author = 'Sandro Truttmann'
release = '1.0.0'
version = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.imgconverter',
    'myst_parser',
]

# Support for both .rst and .md files
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Logo and theme options
html_logo = '_static/hyfi_logo.svg'
latex_logo = '_static/hyfi_logo.png'
html_theme_options = {
    'logo_only': False,
    'display_version': True,
    'style_nav_header_background': '#2c3e50',
}

# -- Extension configuration -------------------------------------------------

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}

# MyST parser configuration
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "amsmath",
]

# -- LaTeX / PDF output -------------------------------------------------------
# xelatex handles Unicode (Greek letters, special chars) natively
latex_engine = 'xelatex'
# Use standard makeindex instead of xindy (xindy may not be installed)
latex_use_xindy = False

latex_elements = {
    'papersize': 'a4paper',
    'fontpkg': r'''
\setmainfont{Latin Modern Roman}
\setsansfont{Latin Modern Sans}
\setmonofont{Latin Modern Mono}
''',
}

# -- LaTeX-only emoji stripping -----------------------------------------------
# Keeps emojis in HTML; silently removes them from the PDF build.
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F9FF"
    "\U00002702-\U000027B0"
    "\U0000231A-\U0000231B"
    "]+",
    flags=re.UNICODE,
)


def _strip_emoji_for_latex(app, doctree, docname):
    if app.builder.name not in ('latex',):
        return
    for node in doctree.traverse(nodes.Text):
        cleaned = _EMOJI_RE.sub('', str(node))
        if cleaned != str(node):
            node.parent.replace(node, nodes.Text(cleaned))


def setup(app):
    app.connect('doctree-resolved', _strip_emoji_for_latex)
