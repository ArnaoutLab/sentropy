# Contributing and support

## Getting help

- **Questions and usage problems:** open an issue at
  <https://github.com/ArnaoutLab/sentropy/issues>. Please include your
  `sentropy` version (`sentropy --version`), Python version, operating system,
  and a minimal example that reproduces the problem.
- **Bug reports:** same place; please include the full traceback.
- **Feature requests:** open an issue describing the use case before writing
  code, so we can agree on the design.

## Contributing code or documentation

1. Fork the repository and create a branch.
2. Install in development mode: `pip install -e ".[tests]"`.
3. Make your change and add or update tests; run
   `pytest --pyargs sentropy --cov sentropy`.
4. If you changed behaviour, update the docstrings and these docs
   (`docs/`), and add a line to the {doc}`changelog`.
5. Open a pull request describing what changed and why.

By contributing you agree that your contributions are released under the
project's MIT license.

## Citing

If you use `sentropy` in published work, please cite the paper listed in the
[repository README](https://github.com/ArnaoutLab/sentropy#how-to-cite-this-work).
