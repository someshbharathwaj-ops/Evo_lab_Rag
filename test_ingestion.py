"""Compatibility entry point for the ingestion command."""

from ingestion.pipeline import main


if __name__ == "__main__":
    raise SystemExit(main())
