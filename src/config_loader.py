from pathlib import Path
import yaml

CONF_PATH = Path(__file__).resolve().parents[1] / "conf" / "config.yaml"

def load_config():
    with open(CONF_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
