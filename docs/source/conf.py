# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'HODLR'
copyright = '2025, Rastislav Turanyi, Hussam Al Daas'
author = 'Rastislav Turanyi, Hussam Al Daas'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'hawkmoth',
    'hawkmoth.ext.napoleon',
]

templates_path = ['_templates']
exclude_patterns = []


# HAWKMOTH
DOCS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(DOCS)
INCLUDE = os.path.join(ROOT, 'include')

hawkmoth_root = os.path.join(os.path.dirname(DOCS), 'src')
hawkmoth_clang = [f'-I{INCLUDE}']

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_custom_sections = [
    ('Errors', 'Raises'),
    ('Allocations')
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
