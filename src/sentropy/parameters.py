"""Module for sentropy's argument-parser configuration.

Constants
---------
LCR_MEASURES
    LCR diversity measures accepted with one input file.
ONE_INPUT_MEASURES
    All measures accepted with one input file (LCR + vendi).
TWO_INPUT_MEASURES
    Measures accepted with two input files (sre, sce).

Classes
-------
ValidateViewpoint
    Validator for the -q parameter.

Functions
---------
configure_arguments
    Creates argument parser for sentropy.
"""

from argparse import Action, ArgumentParser
from warnings import warn

from numpy import inf

from sentropy.exceptions import ArgumentWarning

LCR_MEASURES = (
    "alpha",
    "rho",
    "beta",
    "gamma",
    "normalized_alpha",
    "normalized_rho",
    "normalized_beta",
    "rho_hat",
    "beta_hat",
)
ONE_INPUT_MEASURES = LCR_MEASURES + ("vendi",)
TWO_INPUT_MEASURES = ("sre", "sce")


class ValidateViewpoint(Action):
    """Validator for the -q (viewpoint) parameter.

    Warns if viewpoints larger than 100 are passed, reminding that
    they are treated as infinity in the diversity calculation.
    """

    def __call__(self, parser, args, values, option_string=None):
        if any(viewpoint > 100 and viewpoint != inf for viewpoint in values):
            warn(
                "viewpoints > 100.0 defaults to the analytical formula "
                "for viewpoint = infinity.",
                category=ArgumentWarning,
            )
        setattr(args, self.dest, values)


def configure_arguments():
    """Creates argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured to handle command-line arguments for executing
        sentropy as a module (python -m sentropy) or via the
        `sentropy` console script.
    """
    parser = ArgumentParser(
        prog="sentropy",
        description=(
            "Calculate similarity-sensitive diversity measures from "
            "species counts and pairwise similarities, following "
            "https://arxiv.org/abs/1404.6520."
        ),
        epilog=(
            "Flag names mirror the arguments of the sentropy.sentropy() "
            "function (-similarity, -q, -measure, -level, ...). "
            "For more information, see https://github.com/ArnaoutLab/sentropy"
        ),
    )

    parser.add_argument(
        "-i",
        "--input_filepath",
        nargs="+",
        required=True,
        type=str,
        help=(
            "One or two csv or tsv file(s) with one column per subset, "
            "one row per species, where each element contains the "
            "count of each species in the corresponding subsets. "
            "One input file: calculate one-community diversity "
            "measures. Two input files: calculate relative sentropy "
            "(sre) or cross-sentropy (sce) between the two "
            "communities."
        ),
    )
    parser.add_argument(
        "-l",
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="WARNING",
        help=(
            "Logging verbosity level (listed in decreasing verbosity)."
        ),
    )
    parser.add_argument(
        "-o",
        "--output_filepath",
        default=None,
        help=(
            "A filepath to where the program's output will be saved. "
            "Tab-delimited csv for diversity tables; pickled object "
            "for sre/sce results."
        ),
    )
    parser.add_argument(
        "-similarity",
        dest="similarity",
        help=(
            "The filepath to a csv or tsv file containing the pairwise "
            "similarities for the species in the input file(s). The "
            "file must have a header listing the species names "
            "corresponding to each column, and column and row ordering "
            "must be the same. If omitted, the identity similarity is "
            "used (i.e., ordinary, frequency-only diversity)."
        ),
    )
    parser.add_argument(
        "-q",
        dest="q",
        nargs="+",
        type=float,
        default=[1],
        action=ValidateViewpoint,
        help=(
            "One or more viewpoint parameters. Any non-negative number "
            "(including inf) is valid, but viewpoint parameters greater "
            "than 100 are treated like inf."
        ),
    )
    parser.add_argument(
        "-measure",
        dest="measure",
        nargs="+",
        help=(
            "A list of diversity measures to be computed. With one "
            f"input file: any of {', '.join(ONE_INPUT_MEASURES)} "
            "('vendi' cannot be combined with the others in a single "
            f"call). With two input files: exactly one of "
            f"{', '.join(TWO_INPUT_MEASURES)} per call "
            "(similarity-sensitive relative sentropy or cross-entropy), "
            "matching the sentropy() Python API. Defaults to 'alpha'"
            "(one input file) or 'sre' (two input files)."
        ),
    )
    parser.add_argument(
        "-chunk_size",
        type=int,
        default=10,
        help=(
            "Number of rows to read at a time from the similarity "
            "matrix file. Only relevant when -similarity is a file."
        ),
    )
    parser.add_argument(
        "-level",
        choices=["overall", "subset", "both"],
        default="both",
        help=(
            "Whether to compute diversity at the set level "
            "('overall'), subset level ('subset') or both ('both')."
        ),
    )
    parser.add_argument(
        "-eff_no",
        type=int,
        choices=[0, 1],
        default=1,
        help=(
            "Whether to compute diversity as effective numbers (1) or "
            "as entropies (0)."
        ),
    )
    parser.add_argument(
        "-backend",
        choices=["numpy", "torch"],
        default="numpy",
        help="Whether to use the numpy backend or the torch one.",
    )
    parser.add_argument(
        "-device",
        choices=["cpu", "cuda", "mps"],
        default="cpu",
        help=(
            "Device on which to compute: cpu, or a gpu ('cuda' or "
            "'mps'; requires -backend torch)."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=__get_version__(),
        help="Show the sentropy version and exit.",
    )

    return parser


def __get_version__() -> str:
    from importlib.metadata import version
    return f"sentropy {version('sentropy')}"