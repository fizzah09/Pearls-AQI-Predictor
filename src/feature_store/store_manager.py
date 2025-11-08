import logging
from typing import Dict, Any, List
import pandas as pd

log = logging.getLogger(__name__)


class StoreManager:
    def __init__(self, api_key: str, project_name: str):
        self.api_key = api_key
        self.project_name = project_name
        self.project = None
        self.fs = None
        self._connect()

    def _connect(self):
        try:
            import hopsworks
        except Exception as e:
            raise ImportError("hopsworks package not installed. Add it to requirements.txt") from e
        self.project = hopsworks.login(api_key_value=self.api_key, project=self.project_name)
        self.fs = self.project.get_feature_store()
        log.info("Connected to Hopsworks project %s", self.project_name)

    def _get_or_create_fg(self, name: str, version: int, primary_key: List[str], event_time: str):
        try:
            fg = self.fs.get_feature_group(name=name, version=version)
            log.info("Retrieved existing feature group %s v%s: %s", name, version, type(fg))
            if fg is None:
                log.warning("get_feature_group returned None, will try to create")
                raise Exception("Feature group returned None")
            return fg
        except Exception as e:
            log.info("Feature group %s v%s not found (%s), creating new one", name, version, str(e)[:100])
            try:
                fg = self.fs.create_feature_group(
                    name=name,
                    version=version,
                    primary_key=primary_key,
                    event_time=event_time,
                    online_enabled=False,
                    description=f"Feature group for {name}",
                )
                log.info("Created feature group %s v%s: %s", name, version, type(fg))
                return fg
            except Exception as create_err:
                log.error("Failed to create feature group %s: %s", name, create_err)
                raise

    def store_features(self, features: Dict[str, Any], name: str, version: int, primary_key: List[str], event_time: str):
        if not features:
            log.warning("No features to store for %s", name)
            return
        try:
            df = pd.DataFrame([features])
            log.info("Preparing to store %d features for %s", len(features), name)
            fg = self._get_or_create_fg(name, version, primary_key, event_time)
            if fg is None:
                log.error("Feature group is None for %s", name)
                return
            fg.insert(df, write_options={"wait_for_job": False, "start_offline_materialization": True})
            log.info("Successfully inserted 1 row into %s v%s", name, version)
        except Exception as e:
            log.error("Failed to store features for %s: %s", name, e)
            raise