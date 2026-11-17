# Installation

## From PyPI

```bash
pip install sentropy
```

Requires Python 3.9 or later. NumPy, pandas, SciPy and PyTorch are installed as
dependencies.

### Optional: parallel similarity functions with Ray

When you pass a similarity *function* (rather than a matrix), `sentropy` can
compute the similarity matrix in parallel with [Ray](https://www.ray.io/):

```bash
pip install "sentropy[ray]"
```

Ray does not always support the newest Python release; check the
[Ray installation page](https://docs.ray.io/en/latest/ray-overview/installation.html)
if installation fails. Without Ray, pass `parallelize=False`.

## From source (development)

```bash
git clone https://github.com/ArnaoutLab/sentropy.git
cd sentropy
pip install -e ".[tests]"
pytest --pyargs sentropy --cov sentropy
```

## Building this documentation locally

```bash
pip install -r docs/requirements.txt
pip install -e .
sphinx-build -b html docs docs/_build/html
```

Then open `docs/_build/html/index.html`. To also run every example in these
pages as a test, use `sphinx-build -b doctest docs docs/_build/doctest`
(this needs the full install, including PyTorch).
