import io
import sys
import json
import panflute as pf
from .main import Plait


def main():
    # a Pandoc filter should always be called with one argument: the target format
    # https://pandoc.org/filters.html#arguments
    if len(sys.argv) > 1:
        filter_to = sys.argv[1]
    else:
        print("Warning: no target format specified", file=sys.stderr)
        filter_to = "json"

    # load JSON AST from standard input
    plaiter = Plait(pf.load())

    output_doc = plaiter.plait_ast()

    # "dump" a panflute Doc element
    # defaults to standard output, which is what we want
    pf.dump(output_doc)


if __name__ == "__main__":
    main()
