import os
import sys
from typing import Iterable, Tuple

from dotenv import load_dotenv


def init_env_and_file(
    endpoint_env: str,
    key_env: str,
    argv: Iterable[str] | None = None,
) -> Tuple[str, str, str, bytes]:
    load_dotenv()

    endpoint = os.getenv(endpoint_env)
    key = os.getenv(key_env)

    if not endpoint or not key:
        raise ValueError(
            f"Missing {endpoint_env} or {key_env} environment variables. Check your .env file."
        )

    args = list(argv) if argv is not None else sys.argv

    if len(args) < 2:
        raise ValueError("Usage: python script.py <path-to-file>")

    file_name = args[1]

    if not os.path.exists(file_name):
        raise FileNotFoundError(f"File not found: {file_name}")

    with open(file_name, "rb") as f:
        file_date = f.read()

    os.system("cls" if os.name == "nt" else "clear")

    print(f"Analyzing {file_name} ...\n")

    return endpoint, key, file_name, file_date
