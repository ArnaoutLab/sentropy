"""Main module for executing sentropy on the command line.

Functions
---------
main
    Calculates diversities according to command-line specifications.
cli
    Console-script entry point: parses arguments and calls main().
"""

from sys import argv
from platform import python_version
from logging import captureWarnings, getLogger
import pickle

from numpy import int64
from pandas import read_csv

from sentropy import sentropy
from sentropy.log import LOG_HANDLER, LOGGER
from sentropy.parameters import (
    configure_arguments,
    ONE_INPUT_MEASURES,
    TWO_INPUT_MEASURES,
)

# Ensure warnings are handled properly.
captureWarnings(True)
getLogger("py.warnings").addHandler(LOG_HANDLER)


def main(args):
    """Calculates diversity from species counts and similarities.

    Parameters
    ----------
    args: argparse.Namespace
        Return object of argparse.ArgumentParser object created by
        sentropy.parameters.configure_arguments and applied to command
        line arguments.
    """
    LOGGER.setLevel(args.log_level)
    LOGGER.info(" ".join([f"python{python_version()}", *argv]))
    LOGGER.debug(f"args: {args}")

    measures = args.measure
    num_inputs = len(args.input_filepath)

    if num_inputs == 1:
        if measures is None:
            measures = ['alpha']
        else:
            invalid = [m for m in measures if m not in ONE_INPUT_MEASURES]
            if invalid:
                raise SystemExit(
                    f"Unknown measure(s) with one input file: "
                    f"'{' '.join(invalid)}'. With one input file, -measure "
                    f"must be one or more of: "
                    f"{', '.join(ONE_INPUT_MEASURES)}."
                )
            if "vendi" in measures and len(measures) > 1:
                raise SystemExit(
                    "Measure 'vendi' cannot be combined with other "
                    "measures in a single call."
                )

        counts = read_csv(
            args.input_filepath[0], sep=None, engine="python", dtype=int64
        )

        if "vendi" in measures and args.similarity is not None:
            # Vendi scores require the FULL similarity matrix for the
            # eigendecomposition, so materialize the similarity file into a
            # DataFrame instead of streaming it via a filepath.
            similarity = read_csv(args.similarity, sep=None, engine="python")
            if similarity.shape[0] != similarity.shape[1]:
                # Retry assuming species names in the first column.
                similarity = read_csv(
                    args.similarity, sep=None, engine="python", index_col=0
                )
            if similarity.shape[0] != similarity.shape[1]:
                raise SystemExit(
                    f"Similarity file '{args.similarity}' does not contain a "
                    f"square matrix (got {similarity.shape[0]}x"
                    f"{similarity.shape[1]} after header/label handling)."
                )
        else:
            similarity = args.similarity

        df = sentropy(
            counts,
            similarity=similarity,
            q=args.q,
            measure=measures,
            chunk_size=args.chunk_size,
            return_dataframe=True,
            level=args.level,
            eff_no=args.eff_no,
            backend=args.backend,
            device=args.device,
        )

        print(df)

        if args.output_filepath is not None:
            df.to_csv(
                args.output_filepath,
                sep="\t",
                float_format="%.4f",
                index=False,
            )

    elif num_inputs == 2:
        # Mirror the Python API: with two input files, sentropy() accepts
        # exactly ONE measure ('sre' or 'sce'), as a plain string.
        if measures is None:
            measure = "sre"
        else:
            invalid = [m for m in measures if m not in TWO_INPUT_MEASURES]
            if invalid:
                raise SystemExit(
                    f"Unknown measure(s) with two input files: "
                    f"'{' '.join(invalid)}'. With two input files, "
                    "-measure must be exactly one of: "
                    f"{', '.join(TWO_INPUT_MEASURES)}."
                )
            if len(measures) > 1:
                raise SystemExit(
                    "With two input files, -measure accepts exactly one "
                    f"measure ({', '.join(TWO_INPUT_MEASURES)}), not "
                    "several. Run separate calls to compute both."
                )
            measure = measures[0]

        counts_a = read_csv(
            args.input_filepath[0], sep=None, engine="python", dtype=int64
        )
        counts_b = read_csv(
            args.input_filepath[1], sep=None, engine="python", dtype=int64
        )

        result = sentropy(
            counts_a,
            counts_b,
            similarity=args.similarity,
            q=args.q,
            measure=measure,
            chunk_size=args.chunk_size,
            return_dataframe=True,
            level=args.level,
            eff_no=args.eff_no,
            backend=args.backend,
            device=args.device,
        )

        print("result:", result)

        if args.output_filepath is not None:
            with open(args.output_filepath, "wb") as f:
                pickle.dump(result, f)

    else:
        raise SystemExit(
            f"Expected exactly one or two input files with -i (got "
            f"{num_inputs})."
        )

    LOGGER.info("Done!")


def cli():
    """Console-script entry point.

    Register in pyproject.toml as:

        [project.scripts]
        sentropy = "sentropy.__main__:cli"
    """
    main(configure_arguments().parse_args())


if __name__ == "__main__":  # pragma: no cover
    cli()