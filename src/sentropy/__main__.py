"""Main module for executing diversity on command-line.

Functions
---------
main
    Calculates diversities according to command-line specifications.
"""

from sys import argv
from platform import python_version
from logging import captureWarnings, getLogger
from numpy import int64

from pandas import read_csv

from sentropy.log import LOG_HANDLER, LOGGER
from sentropy import sentropy
from sentropy.parameters import configure_arguments
import json, pickle

# Ensure warnings are handled properly.
captureWarnings(True)
getLogger("py.warnings").addHandler(LOG_HANDLER)


def main(args):
    """Calculates diversity from species counts and similarities.

    Parameters
    ----------
    args: argparse.Namespace
        Return object of argparse.ArgumentParser object created by
        diversity.parameters.configure_arguments and applied to command
        line arguments.
    """
    LOGGER.setLevel(args.log_level)
    LOGGER.info(" ".join([f"python{python_version()}", *argv]))
    LOGGER.debug(f"args: {args}")

    if len(args.input_filepath) == 1:
        counts = read_csv(args.input_filepath[0], sep=None, engine="python", dtype=int64)
        df = sentropy(counts, similarity=args.similarity,\
            qs=args.qs, ms=args.ms, chunk_size=args.chunk_size, \
            return_dataframe=True, level=args.level, eff_no=args.eff_no, \
            backend=args.backend, device=args.device)

        print(df)

        if args.output_filepath is not None:
            df.to_csv(args.output_filepath, sep="\t", float_format="%.4f", index=False)

    else:
        counts_a = read_csv(args.input_filepath[0], sep=None, engine="python", dtype=int64)
        counts_b = read_csv(args.input_filepath[1], sep=None, engine="python", dtype=int64)
        result = sentropy(counts_a, counts_b, similarity=args.similarity,\
            qs=args.qs, ms=args.ms, chunk_size=args.chunk_size, \
            return_dataframe=True, level=args.level, eff_no=args.eff_no, \
            backend=args.backend, device=args.device)

        print("set-level relative entropy:", result[0])
        print("subset-level relative entropies:", result[1])
        
        if args.output_filepath is not None:
            with open(args.output_filepath, 'wb') as f:
                pickle.dump(result, f)

    LOGGER.info("Done!")


if __name__ == "__main__": # pragma: no cover
    parser = configure_arguments()
    args = parser.parse_args()
    main(args)
