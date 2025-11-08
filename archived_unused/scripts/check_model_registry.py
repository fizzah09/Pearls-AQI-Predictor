
from pathlib import Path
from dotenv import load_dotenv

# load repository-level .env (same behaviour as modeling.model_registry)
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

if __name__ == "__main__":
    #!/usr/bin/env python3
    """Quick script to list models in Hopsworks Model Registry non-interactively.

    Run from the repository root or from the `scripts` directory. This script
    adds the repository root to `sys.path` so `import modeling` works.
    """
    from pathlib import Path
    from dotenv import load_dotenv
    import sys

    # Determine project root (two levels up from this file: scripts/ -> project root)
    project_root = Path(__file__).resolve().parent.parent

    # Ensure project root is on sys.path so imports like `modeling.*` work when
    # running the script from the `scripts/` folder.
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # load repository-level .env (same behaviour as modeling.model_registry)
    env_path = project_root / '.env'
    load_dotenv(dotenv_path=env_path)

    if __name__ == "__main__":
        try:
            from modeling.model_registry import list_registered_models
        except Exception as e:
            print("Failed to import model registry helper:", e)
            print("sys.path (first 6 entries):", sys.path[:6])
            print("project_root:", project_root)
            raise

        models = list_registered_models()
        try:
            count = len(models)
        except Exception:
            count = "unknown"

        print(f"\nModel registry query complete. Models found: {count}")
