"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html

-- Path setup --------------------------------------------------------------

If extensions (or modules to document with autodoc) are in another directory,
add these directories to sys.path here. If the directory is relative to the
documentation root, use os.path.abspath to make it absolute, like shown here.
"""
import os
import sys

sys.path.insert(0, os.path.abspath("../"))

from virtual_rainforest import version as vr_version  # noqa: E402

version = vr_version.__version__
release = version

# -- Project information -----------------------------------------------------

project = "Virtual Rainforest"
copyright = (
    "2022, Rob Ewers, David Orme, Olivia Daniels, Jacob Cook, "
    "Jaideep Joshi, Taran Rallings, Vivienne Groner"
)
author = (
    "Rob Ewers, David Orme, Olivia Daniels, Jacob Cook, Jaideep Joshi, "
    "Taran Rallings, Vivienne Groner"
)


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "autodocsumm",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "sphinxcontrib.bibtex",
    "myst_nb",
    # "sphinx_astrorefs",  # Gives author year references
    "sphinx_rtd_theme",
]
autodoc_default_flags = ["members"]
autosummary_generate = True

myst_enable_extensions = ["dollarmath", "deflist"]

# Enable mhchem for chemical formulae
mathjax3_config = {
    "tex": {
        "extensions": ["mhchem.js"],
        # 'inlineMath': [['$', '$']]
    }
}

# Turn off ugly rendering of class attributes
napoleon_use_ivar = True

# Suppress signature expansion of arguments
autodoc_preserve_defaults = True

bibtex_bibfiles = ["refs.bib"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "development/training/.pytest_cache/*",
]

master_doc = "index"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_theme = "sphinx_rtd_theme"  # 'sphinx_material'

html_theme_options = {
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "top",
    "style_external_links": False,
    "style_nav_header_background": "grey",
    # Toc options
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_sidebars = {
    "**": ["logo-text.html", "globaltoc.html", "localtoc.html", "searchbox.html"]
}
