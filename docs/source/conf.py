# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import datetime
import os
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as packageVersion

# make the package importable when building the docs from a source checkout
# without installing it (on Read the Docs, OptiCommPy is installed with pip)
sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "OptiCommPy"
copyright = (
    f"2023-{datetime.date.today().year}, Edson P. da Silva, Adolfo F. Herbster"
)
author = "Edson P. da Silva, Adolfo F. Herbster, Carlos D. F. da Silva, Joaquin Matres"

# The full version (release) and the short X.Y version are taken from the
# installed package, so that they always match setup.py.
try:
    release = packageVersion("OptiCommPy")
except PackageNotFoundError:
    release = "unknown"
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
    "nbsphinx",
    "nbsphinx_link",
]

templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

language = "en"

# The root toctree document.
root_doc = "index"

# The reST default role (used for this markup: `text`) to use for all documents.
default_role = "autolink"

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = False

# -- Extensions configuration ------------------------------------------------

autosummary_generate = True  # Turn on sphinx.ext.autosummary

# The GPU modules depend on CuPy, which requires CUDA and is not available on
# the Read the Docs build servers: mock it so that their API is documented.
autodoc_mock_imports = ["cupy", "cupyx"]

# Links to the documentation of the packages OptiCommPy builds upon
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# The example notebooks are stored with their outputs and some of them need a
# GPU, so they are rendered as they are and never executed during the build.
nbsphinx_execute = "never"

# The ePub format cannot package the notebook (.ipynb) copies that nbsphinx
# provides for download in the HTML output: they are skipped silently.
suppress_warnings = ["epub.unknown_project_files"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Canonical URL of the docs (set by Read the Docs for the version being built)
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "")
