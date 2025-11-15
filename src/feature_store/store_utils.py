import logging
import os
import time
from typing import Any, Dict, List

log = logging.getLogger(__name__)


def import_store_manager():
    """Attempt to import the project's StoreManager.

    Returns the StoreManager class if available, otherwise None.
    """
    try:
        from src.feature_store.store_manager import StoreManager

        return StoreManager
    except Exception as e:
        log.warning("Could not import StoreManager: %s", e)
        return None


def try_store_features(
    features: Dict[str, Any],
    name: str,
    version: int,
    primary_key: List[str],
    event_time: str,
    api_key: str | None = None,
    project_name: str | None = None,
    max_retries: int = 2,
    backoff_seconds: int = 2,
) -> bool:
    """Try to store features to Hopsworks using StoreManager with retries.

    Returns True if stored to remote feature store, False otherwise.
    This function centralizes import errors, retries, and logging so calling code
    can simply fall back to local CSV when it returns False.
    """
    if not features:
        log.warning("No features provided to try_store_features for %s", name)
        return False

    StoreManager = import_store_manager()
    if StoreManager is None:
        log.warning("StoreManager unavailable; cannot store to Hopsworks")
        return False

    attempt = 0
    last_exc = None
    while attempt <= max_retries:
        try:
            sm = StoreManager(api_key=api_key, project_name=project_name)
            sm.store_features(features=features, name=name, version=version, primary_key=primary_key, event_time=event_time)
            log.info("Successfully stored features to Hopsworks feature group %s v%s", name, version)
            return True
        except Exception as e:
            last_exc = e
            log.warning(
                "Attempt %d to store features to Hopsworks failed: %s",
                attempt + 1,
                str(e)[:200],
            )
            attempt += 1
            if attempt <= max_retries:
                time.sleep(backoff_seconds)

    log.error("All attempts to store to Hopsworks failed. Last error: %s", last_exc)
    return False
