"""Sphinx configuration for the sentropy documentation."""

import importlib
import os
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version as _version

# -- Project information -----------------------------------------------------
project = "sentropy"
author = "Arnaout Lab"
copyright = f"{datetime.now().year}, {author}"

try:
    release = _version("sentropy")
except PackageNotFoundError:  # package not installed
    release = "dev"
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_parser",             # write pages in Markdown
    "sphinx.ext.autodoc",      # API docs from docstrings
    "sphinx.ext.napoleon",     # NumPy-style docstrings
    "sphinx.ext.viewcode",     # [source] links
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",  # links to numpy/pandas/... docs
    "sphinx.ext.doctest",      # run the examples in the docs
    "sphinx_copybutton",
    "sphinx_design",           # cards, tabs, grids
    "sphinxarg.ext",           # CLI reference from argparse
]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "amsmath"]
myst_heading_anchors = 3

# -- autodoc -----------------------------------------------------------------
autodoc_member_order = "bysource"
autodoc_typehints = "description"
add_module_names = False
napoleon_numpy_docstring = True
napoleon_google_docstring = False

# Heavy optional dependencies are mocked if they are not installed, so the HTML
# can be built quickly without them. (`make doctest` needs the real thing.)
autodoc_mock_imports = []
for _mod in ("torch", "ray"):
    try:
        importlib.import_module(_mod)
    except ImportError:
        autodoc_mock_imports.append(_mod)

# -- doctest -----------------------------------------------------------------
doctest_global_setup = "import numpy as np\nfrom sentropy import sentropy"

# -- Cross-references to other projects ---------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}
if os.environ.get("SENTROPY_DOCS_OFFLINE"):  # build without internet access
    intersphinx_mapping = {}
nitpicky = False
suppress_warnings = ["myst.domains"]

# -- HTML output ---------------------------------------------------------------
html_theme = "furo"
html_title = f"sentropy {release}"
html_baseurl = "https://arnaoutlab.github.io/sentropy/"
html_theme_options = {
    "source_repository": "https://github.com/ArnaoutLab/sentropy/",
    "source_branch": "main",
    "source_directory": "docs/",
}
_logo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "images", "diversity_logo.png"))
if os.path.exists(_logo):
    html_logo = _logo
