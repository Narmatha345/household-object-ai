"""Start the A2A agent:  python -m app.a2a   (from the backend/ folder)

Host, port, public URL and token come from A2A_* variables in backend/.env.
"""

from __future__ import annotations

import logging
import sys

import uvicorn

from app.a2a.config import get_a2a_settings


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = get_a2a_settings()
    try:
        settings.validate()
    except ValueError as exc:
        logging.getLogger("household_object_ai.a2a").error("%s", exc)
        sys.exit(2)
    uvicorn.run("app.a2a.server:build_app", factory=True, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
