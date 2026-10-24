from __future__ import annotations

from requests import Response, Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

REQUEST_TIMEOUT = (10, 60)


def make_session() -> Session:
    session = Session()
    retry = Retry(
        total=3,
        connect=3,
        read=2,
        backoff_factor=0.7,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        respect_retry_after_header=True,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({"User-Agent": "FloodSight-GatorHacks/0.1"})
    return session


def checked_json(response: Response) -> dict:
    response.raise_for_status()
    try:
        data = response.json()
    except ValueError as exc:
        raise ValueError(
            f"Expected JSON from {response.url}; received {response.headers.get('Content-Type', 'unknown content type')}."
        ) from exc
    if isinstance(data, dict) and data.get("error"):
        error = data["error"]
        raise RuntimeError(
            f"API error {error.get('code', '')}: {error.get('message', 'unspecified service error')}"
        )
    return data


def get_json(session: Session, url: str, params: dict | None = None) -> dict:
    response = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
    return checked_json(response)
