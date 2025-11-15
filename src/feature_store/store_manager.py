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
            if self.fs is None:
                log.warning("Feature store client is None; attempting to reconnect")
                try:
                    self._connect()
                except Exception as conn_err:
                    log.error("Reconnection attempt failed: %s", conn_err)
            if self.fs is None:
                # Fail fast with a clear error to avoid AttributeError when calling get_feature_group
                raise RuntimeError("Feature store client (self.fs) is not initialized")
            fg = self.fs.get_feature_group(name=name, version=version)
            log.info("Retrieved existing feature group %s v%s: %s", name, version, type(fg))
            if fg is None:
                log.warning("get_feature_group returned None, will try to create")
                raise Exception("Feature group returned None")
            return fg
        except Exception as e:
            log.info("Feature group %s v%s not found (%s), creating new one", name, version, str(e)[:100])
            # Ensure feature store client is available before attempting to create the feature group.
            if self.fs is None:
                log.warning("Feature store client is None before create; attempting to reconnect")
                try:
                    self._connect()
                except Exception as conn_err:
                    log.error("Reconnection attempt failed: %s", conn_err)
                    raise RuntimeError("Feature store client (self.fs) is not initialized") from conn_err
                if self.fs is None:
                    raise RuntimeError("Feature store client (self.fs) is not initialized after reconnect")

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

            # Align DataFrame to feature group schema to avoid HSFS schema rejections.
            try:
                # hsfs Feature objects usually expose a `name` attribute
                fg_feature_objs = getattr(fg, "features", None)
                if fg_feature_objs is None:
                    # fallback to attribute that may exist on other hsfs versions
                    fg_feature_objs = getattr(fg, "_features", None)

                if fg_feature_objs:
                    expected_cols = [f.name.lower() for f in fg_feature_objs]
                else:
                    # If we cannot introspect, fall back to DataFrame as-is
                    expected_cols = [c.lower() for c in df.columns]

                # Normalize incoming df columns to lowercase for matching
                df.columns = [c.lower() for c in df.columns]

                # Ensure timestamp column is datetime (HSFS expects timestamp types)
                if "timestamp" in df.columns:
                    try:
                        # If timestamp is integer epoch seconds, convert to datetime
                        if pd.api.types.is_integer_dtype(df["timestamp"].dtype) or pd.api.types.is_float_dtype(df["timestamp"].dtype):
                            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
                        else:
                            # attempt a general parse
                            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
                    except Exception:
                        log.exception("Failed to coerce timestamp column to datetime; leaving as-is")

                # Build aligned DataFrame with expected columns in order; add missing cols as NA
                aligned = {}
                for col in expected_cols:
                    if col in df.columns:
                        aligned[col] = df[col]
                    else:
                        # create missing column as NA with a dtype matching expected feature type
                        if fg_feature_objs:
                            # find matching feature obj
                            fobj = next((f for f in fg_feature_objs if getattr(f, "name", "").lower() == col), None)
                            exp_type = None
                            if fobj is not None:
                                if hasattr(fobj, "type"):
                                    exp_type = getattr(fobj, "type")
                                elif hasattr(fobj, "feature_type"):
                                    exp_type = getattr(fobj, "feature_type")

                            if exp_type:
                                tstr = str(exp_type).lower()
                                try:
                                    if "int" in tstr or "bigint" in tstr:
                                        aligned[col] = pd.Series([pd.NA], dtype="Int64")
                                    elif "double" in tstr or "float" in tstr or "real" in tstr:
                                        aligned[col] = pd.Series([float("nan")], dtype=float)
                                    elif "timestamp" in tstr or "date" in tstr:
                                        aligned[col] = pd.Series([pd.NaT], dtype="datetime64[ns]")
                                    else:
                                        # default to object (string-like)
                                        aligned[col] = pd.Series([None], dtype=object)
                                except Exception:
                                    aligned[col] = pd.Series([None])
                            else:
                                aligned[col] = pd.Series([None])
                        else:
                            # no feature metadata; default to object
                            aligned[col] = pd.Series([None])

                aligned_df = pd.DataFrame(aligned)

                # If we can inspect feature types from the FG, coerce columns to the expected types.
                if fg_feature_objs:
                    for f in fg_feature_objs:
                        col = f.name.lower()
                        exp_type = None
                        # hsfs feature object may expose different attributes for type
                        if hasattr(f, "type"):
                            exp_type = getattr(f, "type")
                        elif hasattr(f, "feature_type"):
                            exp_type = getattr(f, "feature_type")

                        if col not in aligned_df.columns:
                            continue

                        try:
                            if exp_type and "timestamp" in str(exp_type).lower():
                                aligned_df[col] = pd.to_datetime(aligned_df[col], utc=True, errors="coerce")
                            elif exp_type and ("bigint" in str(exp_type).lower() or "int" in str(exp_type).lower()):
                                    # use pandas nullable integer dtype so NaNs are preserved
                                    num = pd.to_numeric(aligned_df[col], errors="coerce")
                                    # if there are fractional values, round them to nearest int as a best-effort
                                    try:
                                        if not num.dropna().empty and (num.dropna() % 1 != 0).any():
                                            num = num.round(0)
                                    except Exception:
                                        pass
                                    aligned_df[col] = num.astype("Int64")
                            elif exp_type and ("double" in str(exp_type).lower() or "float" in str(exp_type).lower()):
                                aligned_df[col] = pd.to_numeric(aligned_df[col], errors="coerce").astype(float)
                            else:
                                # leave as-is for other types
                                aligned_df[col] = aligned_df[col]
                        except Exception:
                            log.debug("Could not cast column %s to expected type %s", col, exp_type)
                            continue
                else:
                    # Best-effort: try casting object columns to float where possible
                    for col in aligned_df.columns:
                        if aligned_df[col].dtype == object:
                            try:
                                aligned_df[col] = aligned_df[col].astype(float)
                            except Exception:
                                continue

                # Attempt insert; if insert fails due to null-only columns or unsupported dtypes,
                # drop columns that are entirely null or convert them to object as a last resort.
                def _normalize_for_avro(df_: pd.DataFrame) -> pd.DataFrame:
                    # Convert pandas nullable integers to native Python ints/None
                    for c in df_.columns:
                        try:
                            if pd.api.types.is_integer_dtype(df_[c].dtype) or str(df_[c].dtype) == "Int64":
                                df_[c] = df_[c].apply(lambda x: int(x) if pd.notna(x) else None)
                        except Exception:
                            continue
                    return df_

                try:
                    aligned_df = _normalize_for_avro(aligned_df)
                    fg.insert(aligned_df, write_options={"wait_for_job": False, "start_offline_materialization": True})
                    log.info("Successfully inserted 1 row into %s v%s", name, version)
                except Exception as insert_exc:
                    log.warning("Initial insert failed (%s). Attempting to drop null-only columns and retry.", insert_exc)
                    # drop columns that are entirely NA/None
                    non_null_mask = aligned_df.notna().any(axis=0)
                    cleaned_df = aligned_df.loc[:, non_null_mask]
                    # If cleaned_df is empty (all-null), convert null columns to object dtype with None and retry
                    if cleaned_df.shape[1] == 0:
                        for c in aligned_df.columns:
                            aligned_df[c] = aligned_df[c].astype(object)
                        try:
                            aligned_df = _normalize_for_avro(aligned_df)
                            fg.insert(aligned_df, write_options={"wait_for_job": False, "start_offline_materialization": True})
                            log.info("Inserted fallback row into %s v%s after converting null-only columns to object", name, version)
                        except Exception:
                            log.exception("Fallback insert after converting null-only columns failed")
                            raise
                    else:
                        try:
                            cleaned_df = _normalize_for_avro(cleaned_df)
                            fg.insert(cleaned_df, write_options={"wait_for_job": False, "start_offline_materialization": True})
                            log.info("Inserted row into %s v%s after dropping null-only columns", name, version)
                        except Exception:
                            log.exception("Insert after dropping null-only columns failed")
                            raise
            except Exception as schema_exc:
                log.error("Failed to align DataFrame to feature group schema for %s: %s", name, schema_exc)
                # Do not attempt to insert the original unaligned DataFrame because it may contain
                # columns incompatible with the feature group (causes 'null' dtype errors). Raise instead.
                raise
        except Exception as e:
            log.error("Failed to store features for %s: %s", name, e)
            raise