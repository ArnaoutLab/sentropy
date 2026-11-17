# Minimal makefile for Sphinx documentation
SPHINXOPTS  ?=
SPHINXBUILD ?= sphinx-build
SOURCEDIR   = .
BUILDDIR    = _build

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# `make html` builds the site; `make doctest` runs the examples in the docs
# (doctest needs the full install, including torch).
.PHONY: help Makefile
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
