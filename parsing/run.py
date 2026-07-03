"""CLI entry point for the deterministic parsing pipeline.

Usage:
    python -m parsing.run                  # everything
    python -m parsing.run --ext .pptx .xlsx .docm .doc .gif   # only these
"""
import argparse

from .orchestrator import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext", nargs="*", default=None, help="Restrict to these extensions, e.g. --ext .pptx .xlsx")
    args = ap.parse_args()
    run(only_exts=args.ext)


if __name__ == "__main__":
    main()
