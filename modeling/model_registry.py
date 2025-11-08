"""
Model Registry integration for storing trained models in Hopsworks
"""
import hopsworks
import joblib
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Any, Optional
import json
import os
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

log = logging.getLogger(__name__)


def _flatten_metrics(metrics: Dict[str, Any], parent_key: str = "", sep: str = "_") -> Dict[str, float]:
    """Flatten nested metric dicts into a single-level dict with numeric values.

    Example: {"validation": {"rmse": 1.2}} -> {"validation_rmse": 1.2}
    Non-numeric leaf values are skipped with a warning.
    """
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
    api_key: Optional[str] = None
) -> Any:
    try:
        log.info(f"Connecting to Hopsworks...")
        
        if api_key is None:
            api_key = os.getenv('HOPSWORKS_API_KEY')
        
        project_name = os.getenv('HOPSWORKS_PROJECT_NAME')
        
        if api_key:
            if project_name:
                project = hopsworks.login(api_key_value=api_key, project=project_name)
            else:
                project = hopsworks.login(api_key_value=api_key)
        else:
            project = hopsworks.login()
        
        log.info(f"Connected to project: {project.name}")
        
        mr = project.get_model_registry()
        
        model = joblib.load(model_path)
        
                                                                         
                                                                           
                                                                     
                                                                           
                                                                           
                                                                   
        try:
            from hsml.schema import Schema
            from hsml.model_schema import ModelSchema

            input_schema = Schema(feature_names)
            output_schema = Schema([target_name])

            model_schema = ModelSchema(
                input_schema=input_schema,
                output_schema=output_schema
            )
        except Exception as e_schema_build:
            log.warning("Could not build hsml ModelSchema from feature names: %s. Will register model without explicit schema.", e_schema_build)
            model_schema = None
        
        if description is None:
            description = f"XGBoost model for {target_name} prediction trained on {len(feature_names)} features"
        
        log.info(f"Registering model: {model_name}")
        
                                                                             
                                                                         
                                                                         
                                                                              
                                                                  
                                                                              
                                                                        
                                                                         
        flat_metrics = {}
        if metrics:
            flat_metrics = _flatten_metrics(metrics)

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
                log.warning("Model schema rejected by Hopsworks client/registry: %s. Retrying without model_schema.", e_schema)
                registered_model = mr.python.create_model(
                    name=model_name,
                    metrics=flat_metrics,
                    description=description,
                    input_example=None,
                )
        else:
            registered_model = mr.python.create_model(
                name=model_name,
                metrics=flat_metrics,
                description=description,
                input_example=None,
            )
        
        model_dir = Path(model_path).parent
        registered_model.save(str(model_dir))
        
        log.info(f" Model registered successfully!")
        log.info(f"  Model name: {model_name}")
        log.info(f"  Version: {registered_model.version}")
        log.info(f"  Metrics: {metrics}")
        
        return registered_model
        
    except Exception as e:
        log.error(f"Failed to register model: {e}")
        raise


def load_model_from_registry(
    model_name: str,
    version: Optional[int] = None
):
    try:
        log.info(f"Loading model from registry: {model_name}")
        
        api_key = os.getenv('HOPSWORKS_API_KEY')
        project_name = os.getenv('HOPSWORKS_PROJECT_NAME')
        
        if api_key:
            if project_name:
                project = hopsworks.login(api_key_value=api_key, project=project_name)
            else:
                project = hopsworks.login(api_key_value=api_key)
        else:
            project = hopsworks.login()
        
        mr = project.get_model_registry()
        
        if version:
            model = mr.get_model(model_name, version=version)
        else:
            model = mr.get_model(model_name)
        
        model_dir = model.download()
        
        model_path = Path(model_dir) / f"{model_name}.pkl"
        loaded_model = joblib.load(model_path)
        
        log.info(f" Model loaded: {model_name} v{model.version}")
        
        return loaded_model, model
        
    except Exception as e:
        log.error(f"Failed to load model: {e}")
        raise


def list_registered_models():
    """List all models in the Model Registry.

    This function tries a few common ModelRegistry APIs to be compatible with
    different hopsworks client versions. It prints a human-readable list of
    models and returns the list (possibly empty).
    """
    api_key = os.getenv('HOPSWORKS_API_KEY')
    project_name = os.getenv('HOPSWORKS_PROJECT_NAME')

    if api_key:
        if project_name:
            project = hopsworks.login(api_key_value=api_key, project=project_name)
        else:
            project = hopsworks.login(api_key_value=api_key)
    else:
        project = hopsworks.login()

    mr = project.get_model_registry()

    models = []

                                        
    try:
        models = mr.get_models()
    except TypeError:
        log.warning("mr.get_models() requires arguments on this client version; trying fallbacks")
                                        
        for candidate in ("", "*"):
            try:
                models = mr.get_models(candidate)
                break
            except Exception:
                continue

                                                                   
        if not models:
            if hasattr(mr, 'get_model_names'):
                try:
                    names = mr.get_model_names()
                    for name in names:
                        try:
                            m = mr.get_model(name)
                            models.append(m)
                        except Exception:
                            log.exception("Failed to fetch model '%s'", name)
                except Exception:
                    log.exception("Failed to enumerate model names using get_model_names()")
            elif hasattr(mr, 'get_models_by_name'):
                try:
                    names = mr.get_models_by_name()
                    for name in names:
                        try:
                            m = mr.get_model(name)
                            models.append(m)
                        except Exception:
                            log.exception("Failed to fetch model '%s'", name)
                except Exception:
                    log.exception("Failed to enumerate model names using get_models_by_name()")
            else:
                log.exception("Unable to list models: mr.get_models requires a name and no fallback available")
    except Exception:
        log.exception("Unexpected error when calling mr.get_models()")

                   
    print("\n" + "="*70)
    print("REGISTERED MODELS")
    print("="*70)

    for model in models:
        try:
            print(f"\nModel: {model.name}")
            print(f"  Version: {model.version}")
            print(f"  Created: {model.created}")
            if hasattr(model, 'metrics'):
                print(f"  Metrics: {model.metrics}")
        except Exception:
            print("\nModel (unparsed object):", repr(model))

    print("="*70)

    return models
