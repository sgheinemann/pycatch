# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

sys.path.insert(0, os.path.abspath('..'))

project = 'pyCATCH'
copyright = '2023, Stephan G. Heinemann'
author = 'Stephan G. Heinemann'

# Add your package version here:
version = '1.0.0'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx_simplepdf',]

#autodoc_mock_imports = ['../pycatch']

# Include the module and class in your documentation
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'inherited-members': True,
    'show-inheritance': True,
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = []

simplepdf_vars = {
    'primary': '#1a365d',
    'cover-bg': '#1a365d',
    'cover-text': '#ffffff',
}

# Inject cover text directly via Sphinx theme options (No extra files!)
simplepdf_theme_options = {
    'extra_css': r"""
        /* Subtitle under title */
        .simplepdf-cover .title::after {
            content: "Collection of Analysis Tools for Coronal Holes";
            display: block;
            font-size: 0.4em;
            font-weight: 300;
            margin-top: 15px;
            color: #cbd5e1;
        }

        /* Author and Date under version */
        .simplepdf-cover .version::after {
            content: "\A Author: Stephan G. Heinemann \A Date: October 2026";
            white-space: pre-wrap;
            display: block;
            font-size: 0.6em;
            font-weight: 400;
            margin-top: 20px;
            color: #e2e8f0;
            line-height: 1.5;
        }
    """
}
