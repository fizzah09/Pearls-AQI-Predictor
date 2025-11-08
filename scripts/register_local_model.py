
from pathlib import Path
import logging
import csv
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("register_local_model")


def main():
    project_root = Path(__file__).parent.parent
                                                                   
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    model_path = project_root / "modeling" / "models" / "pollutant_aqi_regression_xgboost.pkl"
    if not model_path.exists():
        log.error("Model file not found: %s", model_path)
        return

                                                                         
    metrics_file = project_root / "modeling" / "evaluation" / "metrics_summary.csv"
    metrics = {}
    if metrics_file.exists():
        with open(metrics_file, newline='', encoding='utf-8') as fh:
            reader = csv.reader(fh)
            header = next(reader)
                                                                               
            metric_names = header[1:]
            for row in reader:
                if not row:
                    continue
                idx = row[0]
                for i, col in enumerate(metric_names, start=1):
                    key = f"{idx}_{col}" if col != 'n_samples' else f"{idx}_samples"
                    val = row[i] if i < len(row) else ''
                    try:
                        metrics[key] = float(val)
                    except Exception:
                        metrics[key] = val
        log.info("Loaded metrics from %s: %s", metrics_file, list(metrics.keys()))
    else:
        log.warning("Metrics file not found: %s — proceeding with minimal metrics", metrics_file)
        metrics = {"note": "no_metrics_available"}

                                                               
    feat_file = project_root / "modeling" / "evaluation" / "feature_importance.csv"
    if feat_file.exists():
        feature_names = []
        with open(feat_file, newline='', encoding='utf-8') as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                if 'feature' in r:
                    feature_names.append(r['feature'])
        log.info("Loaded %d feature names from %s", len(feature_names), feat_file)
    else:
                                                                 
        feature_names = None
        log.warning("Feature importance file not found: %s — feature names will be omitted", feat_file)

                    
    try:
        from modeling.model_registry import register_model_to_hopsworks

        model_name = "aqi_pollutant_aqi_regression"
        log.info("Registering model %s -> %s", model_name, model_path)

        registered = register_model_to_hopsworks(
            model_path=str(model_path),
            model_name=model_name,
            metrics=metrics,
            feature_names=feature_names or [],
            target_name="pollutant_aqi",
            description="Manual registration of local pollutant_aqi_regression_xgboost model"
        )

        log.info("Registration result: %s v%s", getattr(registered, 'name', None), getattr(registered, 'version', None))
    except Exception as e:
        log.exception("Failed to register local model: %s", e)


if __name__ == '__main__':
    main()
