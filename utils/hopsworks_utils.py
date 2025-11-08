import shutil
from pathlib import Path
import pandas as pd

def download_latest_model_from_hopsworks(model_name: str = "aqi_pollutant_aqi_regression", dest_dir: str | Path = "modeling/models"):
    """
    Attempts to download the latest model from Hopsworks Model Registry.
    Returns Path to downloaded .pkl or None.
    Requires HOPSWORKS credentials available in env / configuration.
    """
    try:
        import hopsworks
    except Exception:
        return None

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    project = hopsworks.login()
    mr = project.get_model_registry()
                                                                         
    models = mr.get_models(name=model_name)
    if not models:
        return None
                              
    model_obj = sorted(models, key=lambda m: getattr(m, "created", 0))[-1]
    model_dir = model_obj.download()                                      
    model_dir = Path(model_dir)
                                          
    pkls = list(model_dir.rglob("*.pkl"))
    if not pkls:
                                                                              
        for p in model_dir.iterdir():
            if p.is_file():
                shutil.copy(p, dest_dir / p.name)
        return None
    chosen = pkls[0]
    target = dest_dir / chosen.name
    shutil.copy(chosen, target)
    return target

def read_feature_group(fg_name: str, version: int | None = None):
    """
    Reads a feature group from Hopsworks Feature Store if available.
    Returns pandas.DataFrame or None.
    """
    try:
        import hopsworks
    except Exception:
        return None
    project = hopsworks.login()
    fs = project.get_feature_store()
    if version is None:
        fg = fs.get_feature_group(fg_name)
    else:
        fg = fs.get_feature_group(fg_name, version=version)
    df = fg.read()
    return df