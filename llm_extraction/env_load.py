"""Load YANDEX_API_KEY / YANDEX_FOLDER_ID from .claude/.env without printing them.

.claude/.env lives at the local working-directory root (Nornikel_Hack/), which
is two levels above this repo clone (see CLAUDE.md's "two-tier layout") --
walk up from this file looking for it instead of hardcoding a fixed depth,
since that depth differs between this repo copy and the root test-harness
copy it was promoted from.
"""
import os
from pathlib import Path


def _find_env_path() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / ".claude" / ".env"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find .claude/.env in any parent directory")


ENV_PATH = _find_env_path()


def load_env() -> None:
    content = ENV_PATH.read_text(encoding="utf-8")
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_env()
API_KEY = os.environ["YANDEX_API_KEY"]
FOLDER_ID = os.environ["YANDEX_FOLDER_ID"]
