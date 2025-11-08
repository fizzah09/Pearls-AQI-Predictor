import sys
from pathlib import Path
import logging
import argparse

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from modeling.train_pipeline import main as train_main


def main():
    """Main training pipeline with optional Hopsworks integration"""
    parser = argparse.ArgumentParser(description="Train AQI prediction model")
    parser.add_argument(
        "--use-hopsworks",
        action="store_true",
        help="Load data from Hopsworks Feature Store instead of local CSV"
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=30,
        help="Number of days of data to load from Hopsworks (default: 30)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/ml_training_data_1year.csv",
        help="Path to local CSV file (used when not using Hopsworks)"
    )
    
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("AQI TRAINING PIPELINE")
    logger.info("="*70)
    
    if args.use_hopsworks:
        logger.info(" Loading data from Hopsworks Feature Store...")
        try:
            from modeling.data_loader_hopswork import load_training_data_from_hopsworks
            
            df = load_training_data_from_hopsworks(days_back=args.days_back)
            
            if df is None or len(df) == 0:
                logger.error(" No data loaded from Hopsworks")
                logger.info(" Falling back to local CSV")
                args.use_hopsworks = False
            else:
                data_path = project_root / "data" / "ml_training_data_latest.csv"
                data_path.parent.mkdir(exist_ok=True, parents=True)
                df.to_csv(data_path, index=False)
                logger.info(f" Saved Hopsworks data to {data_path}")
                
                args.data = str(data_path)
                
        except Exception as e:
            logger.error(f" Failed to load from Hopsworks: {e}")
            logger.info(" Falling back to local CSV")
            args.use_hopsworks = False
    
    logger.info(f" Starting training with data: {args.data}")
    sys.argv = [sys.argv[0], "--data", args.data]
    result = train_main()
    
                                                                                           
    Path("data").mkdir(parents=True, exist_ok=True)
    try:
        X_train = None

                                                     
        if result is not None and hasattr(result, "to_csv"):
            X_train = result
                                                                        
        elif isinstance(result, dict) and "X_train" in result:
            X_train = result["X_train"]

                                                                                                                 
        if X_train is None:
            try:
                from modeling import train_pipeline
                X_train = getattr(train_pipeline, "X_train", None)
            except Exception:
                X_train = None

        if X_train is not None and hasattr(X_train, "to_csv"):
            X_train.to_csv("data/training_features.csv", index=False)
            logger.info("Saved training feature matrix to data/training_features.csv")
        else:
            logger.info("No X_train available to save (train_main did not return it and no module-level X_train found)")
    except Exception as e:
        logger.error("Failed to save training features: %s", e)


if __name__ == "__main__":
    main()
