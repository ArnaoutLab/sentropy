"""Module for diversity's argument parser configuration.

Classes
-------
ValidateViewpoint
    Validator for -viewpoint parameter.

Functions
---------
configure_arguments
    Creates argument parser for .
"""

from argparse import Action, ArgumentParser
from sys import stdout
from warnings import warn

from numpy import inf

from sentropy.exceptions import ArgumentWarning


class ValidateViewpoint(Action):
    """Validator for -viewpoint parameter."""

    def __call__(self, parser, args, values, option_string=None):
        """Validates -viewpoint parameter.

        Warns if arguments larger than 100 are passed, reminding that
        they are treated as infinity in the diversity calculation.
        """
        if any([viewpoint > 100 and viewpoint != inf for viewpoint in values]):
            warn(
                "viewpoints > 100.0 defaults to the analytical formula"
                " for viewpoint = infinity.",
                category=ArgumentWarning,
            )
        setattr(args, self.dest, values)


def configure_arguments():
    """Creates argument parser.

    Returns
    -------
    argparse.ArgumentParser configured to handle command-line arguments
    for executing diversity as a module.
    """
    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_filepath",
        nargs="+",
        type=str,
        help=(
            "One or two csv or tsv file(s) with one column per subset, one "
            "row per species, where each element contains the count of "
            "each species in the corresponding subsets."
        ),
    )
    parser.add_argument(
        "-l",
        "--log_level",
        help=(
            "Logging verbosity level. Must be one of DEBUG, INFO,"
            " WARNING, ERROR, CRITICAL (listed in decreasing"
            " verbosity)."
        ),
        default="WARNING",
    )
    parser.add_argument(
        "-o",
        "--output_filepath",
        default=None,
        help="A filepath to where the program's output will be saved",
    )
    parser.add_argument(
        "-s",
        "--similarity",
        help=(
            "The filepath to a csv or tsv file containing a similarity"
            " for the species in the input file. The file must have a"
            " header listing the species names corresponding to each"
            " column, and column and row ordering must be the same."
        ),
    )
    parser.add_argument(
        "-qs",
        nargs="+",
        type=float,
        help=(
            "A list of viewpoint parameters. Any non-negative number"
            " (including inf) is valid, but viewpoint parameters"
            " greater than 100 are treated like inf."
        ),
        action=ValidateViewpoint,
    )
    parser.add_argument(
        "-ms",
        nargs="+",
        type=str,
        help=(
            "A list of diversity measures to be computed. Must be in 'alpha', 'rho', 'beta',"
            "'gamma', 'normalized_alpha', 'normalized_rho', 'normalized_beta', 'rho_hat', 'beta_hat'."
        ),
        default= ["alpha", "rho", "beta", "gamma", "normalized_alpha", "normalized_rho", \
        "normalized_beta", "rho_hat", "beta_hat"]
    )

    parser.add_argument(
        "-chunk_size",
        type=int,
        help="Number of rows to read at a time from the similarities matrix.",
        default=1,
    )

    parser.add_argument(
        "-level",
        type=str,
        help="whether to compute diversity at the set level ('overall'), subset level ('subset') or both ('both').",
        default='both',
    )

    parser.add_argument(
        "-eff_no",
        type=int,
        help="whether to compute diversity as effective numbers (1) or as entropies (0).",
        default=1,
    )

    parser.add_argument(
        "-backend",
        type=str,
        help="whether to use the numpy backend ('numpy') or the torch one ('torch').",
        default='numpy',
    )

    parser.add_argument(
        "-device",
        type=str,
        help="whether to compute the diversity indices on the cpu ('cpu') or the gpu ('mps' or 'cuda').",
        default='cpu',
    )


    return parser
