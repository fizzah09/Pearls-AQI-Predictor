import os, re, yaml, logging
from pathlib import Path
from typing import Any, Dict

log = logging.getLogger(__name__)


def _sub_env(val: Any) -> Any:
    if isinstance(val, dict):
        return {k: _sub_env(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_sub_env(v) for v in val]
    if isinstance(val, str):
        for name in re.findall(r"\$\{([^}]+)\}", val):
            env = os.getenv(name)
            if env is None:
                log.warning(f"Environment variable {name} not set")
                continue
            val = val.replace(f"${{{name}}}", env)
        try:
            if val.replace(".", "", 1).lstrip("-").isdigit():
                return float(val) if "." in val else int(val)
        except Exception:
            pass
    return val


essential_env = [
    "OPENWEATHER_API_KEY",
    "HOPSWORKS_API_KEY",
    "HOPSWORKS_PROJECT_NAME",
    "LOCATION_LAT",
    "LOCATION_LON",
    "LOCATION_CITY",
]


def load_config(rel_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load YAML config and substitute ${VAR} from .env or process env."""
    root = Path(__file__).resolve().parents[2]
    
    env_path = root / ".env"
    if not env_path.exists():
        env_path = root.parent / ".env"
    
    if env_path.exists():
        loaded = False
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
            loaded = True
            log.info(f"Loaded environment from {env_path}")
        except Exception:
            try:
                _manual_load_env(env_path)
                loaded = True
                log.info(f"Loaded environment (manual) from {env_path}")
            except Exception as e:
                log.warning(f"Failed to load .env manually: {e}")
        if not loaded:
            log.info("Using process environment only")
    else:
        log.info(".env not found in project or parent; using process environment only")

    cfg_path = root / rel_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg = _sub_env(cfg)

    missing = [v for v in essential_env if v not in os.environ and _contains_placeholder(cfg, v)]
    if missing:
        log.warning("Missing environment variables: %s", ", ".join(missing))

    return cfg


def _contains_placeholder(obj: Any, var: str) -> bool:
    if isinstance(obj, dict):
        return any(_contains_placeholder(v, var) for v in obj.values())
    if isinstance(obj, list):
        return any(_contains_placeholder(v, var) for v in obj)
    if isinstance(obj, str):
        return f"${{{var}}}" in obj
    return False


def _manual_load_env(path: Path) -> None:
    """Minimal .env loader: KEY=VALUE lines, ignoring comments/blank lines."""
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "=" not in s:
            continue
        key, val = s.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val
