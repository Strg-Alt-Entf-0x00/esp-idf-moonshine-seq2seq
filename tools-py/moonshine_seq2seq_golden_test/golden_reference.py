#!/usr/bin/env python3
"""Command-line entry point for the standalone Moonshine ASR benchmark."""

if __package__:
    from .benchmark import main
else:
    from benchmark import main


if __name__ == "__main__":
    raise SystemExit(main())
