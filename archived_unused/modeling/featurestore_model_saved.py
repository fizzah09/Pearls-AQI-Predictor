"""
Model Registry integration for storing trained models
"""
import hopsworks
import joblib
from pathlib import Path
import logging
from typing import Dict, Any, Optional

log = logging.getLogger(__name__)


def _flatten_metrics(metrics: Dict[str, Any], parent_key: str = "", sep: str = "_") -> Dict[str, float]:
    out: Dict[str, float] = {}

    def _rec(cur, prefix):
        if isinstance(cur, dict):
            for k, v in cur.items():
                new_key = f"{prefix}{sep}{k}" if prefix else k
                _rec(v, new_key)
        else:
            try:
                out[prefix] = float(cur)
            except Exception:
                log.warning("Skipping non-numeric metric '%s': %r", prefix, cur)

    _rec(metrics, parent_key)
    return out


def register_model_to_hopsworks(
    model_path: str,
    model_name: str,
    metrics: Dict[str, Any],
    feature_names: list,
    target_name: str,
    description: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Optional[object]:
    try:
        log.info("Connecting to Hopsworks...")

        if api_key:
            project = hopsworks.login(api_key_value=api_key)
        else:
            project = hopsworks.login()

        log.info("Connected to project: %s", project.name)

        mr = project.get_model_registry()

        model = joblib.load(model_path)

        # Try to build a schema but fall back if not possible
        model_schema = None
        try:
            from hsml.schema import Schema
            from hsml.model_schema import ModelSchema

            input_schema = Schema(feature_names)
            output_schema = Schema([target_name])
            model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)
        except Exception as e_schema_build:
            log.warning("Could not build hsml ModelSchema: %s. Registering without explicit schema.", e_schema_build)

        if description is None:
            description = f"Model for {target_name} prediction trained on {len(feature_names)} features"

        log.info("Registering model: %s", model_name)

        flat_metrics = _flatten_metrics(metrics) if metrics else {}

        registered_model = None
        if model_schema is not None:
            try:
                registered_model = mr.python.create_model(
                    name=model_name,
                    metrics=flat_metrics,
                    model_schema=model_schema,
                    description=description,
                    input_example=None,
                )
            except Exception as e_schema:
                log.warning("Schema rejected: %s. Retrying without schema.", e_schema)

        if registered_model is None:
            registered_model = mr.python.create_model(
                name=model_name, metrics=flat_metrics, description=description, input_example=None
            )

        model_dir = Path(model_path).parent
        registered_model.save(str(model_dir))

        log.info("Model registered successfully: %s v%s", model_name, getattr(registered_model, "version", "?"))
        return registered_model

    except Exception as e:
        log.error("Failed to register model: %s", e)
        raise


def load_model_from_registry(model_name: str, version: Optional[int] = None):
    try:
        log.info("Loading model from registry: %s", model_name)
        project = hopsworks.login()
        mr = project.get_model_registry()

        if version:
            model = mr.get_model(model_name, version=version)
        else:
            model = mr.get_model(model_name)

        model_dir = model.download()
        model_path = Path(model_dir) / f"{model_name}.pkl"
        loaded_model = joblib.load(model_path)
        log.info("Model loaded: %s v%s", model_name, getattr(model, "version", "?"))
        return loaded_model, model

    except Exception as e:
        log.error("Failed to load model: %s", e)
        raise


def list_registered_models():
    try:
        project = hopsworks.login()
        mr = project.get_model_registry()

        try:
            models = mr.get_models()
        except TypeError:
            log.warning("mr.get_models() requires args; returning empty list for this client version")
            models = []

        print("\n" + "=" * 70)
        print("REGISTERED MODELS")
        print("=" * 70)

        for model in models:
            try:
                print(f"\nModel: {model.name}")
                print(f"  Version: {model.version}")
                print(f"  Created: {model.created}")
                if hasattr(model, "metrics"):
                    print(f"  Metrics: {model.metrics}")
            except Exception:
                print("\nModel (unparsed):", repr(model))

        print("=" * 70)
        print("Model registry query complete. Models found:", len(models))
        return models

    except Exception as e:
        log.error("Failed to list models: %s", e)
        raise
