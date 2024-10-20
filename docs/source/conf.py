import importlib.metadata
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath("../../src"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "ovrlpy"
copyright = f"""
{datetime.now():%Y}, Sebastian Tiesmeyer, Naveed Ishaque, Roland Eils,
Berlin Institute of Health @ Charité"""
author = "Sebastian Tiesmeyer"
version = importlib.metadata.version("ovrlpy")
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    "sphinx_copybutton",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "autoapi.extension",
    "sphinx.ext.mathjax",
    "myst_nb",
]

nb_execution_mode = "off"


autodoc_typehints = "none"
autodoc_typehints_format = "short"

autoapi_dirs = ["../../ovrlpy"]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]
autoapi_own_page_level = "function"
# autoapi_python_class_content = "both"
# autoapi_template_dir = "_templates"
# autoapi_member_order = "groupwise"

python_use_unqualified_type_names = True  # still experimental

autosummary_generate = True
autosummary_imported_members = True

nitpicky = True
nitpick_ignore = [
    ("py:class", "optional"),
]

intersphinx_mapping = dict(
    anndata=("https://anndata.readthedocs.io/en/stable/", None),
    matplotlib=("https://matplotlib.org/stable/", None),
    numpy=("https://numpy.org/doc/stable/", None),
    pandas=("https://pandas.pydata.org/pandas-docs/stable/", None),
    python=("https://docs.python.org/3", None),
    scanpy=("https://scanpy.readthedocs.io/en/stable/", None),
    scipy=("https://docs.scipy.org/doc/scipy/", None),
)

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_logo = "../resources/ovrlpy-logo.png"
html_theme_options = {
    "logo_only": True,
    # 'display_version': False,
}


# def skip_submodules(app, what, name, obj, skip, options):
#     if what == "module":
#         skip = True
#     return skip


# def setup(sphinx):
#     sphinx.connect("autoapi-skip-member", skip_submodules)
