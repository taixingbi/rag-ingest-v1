"""CLI and env setup for ingest script. Parse argv, set ENV and CHROMA_DATABASE, load dotenv."""
import os
import sys

from dotenv import load_dotenv


def setup_cli_env() -> None:
    """Parse argv, set ENV and CHROMA_DATABASE, load dotenv. Call when run as script."""
    if len(sys.argv) < 2 or sys.argv[1] not in ("dev", "qa", "prod"):
        sys.exit("Usage: python ingest.py {dev|qa|prod} [save <database_name>]")
    _env = sys.argv[1]
    os.environ["ENV"] = _env
    load_dotenv()
    load_dotenv(f".env.{_env}")
    if len(sys.argv) >= 4 and sys.argv[2] == "save":
        os.environ["CHROMA_DATABASE"] = sys.argv[3]
    else:
        os.environ["CHROMA_DATABASE"] = f"rag_{_env}"
