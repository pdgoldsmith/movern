import requests

from movern._version import __version__
from movern.utils import global_logger


def validate_version():
    current_version = __version__

    package = "movern"
    try:
        response = requests.get(f"https://pypi.org/pypi/{package}/json", timeout=3)
        data = response.json()
        if "info" not in data:
            # Package not yet on PyPI (e.g. pre-release fork)
            return
        latest_version = data["info"]["version"]
    except (requests.ConnectionError, requests.Timeout, ValueError, KeyError):
        global_logger.info(
            "Cannot determine whether Movern version is up-to-date"
        )
        return

    if current_version != latest_version:
        global_logger.warning(
            """
            You are using movern version %s, however a newer version is available.
            Please upgrade via: "python -m pip install --upgrade movern"
            """,
            current_version,
        )
