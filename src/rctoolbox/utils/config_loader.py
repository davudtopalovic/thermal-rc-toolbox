import os
import json
from importlib.resources import files

def load_json_config(path, name):
    if path is None:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} config not found: {path}")
    with open(path, "r") as f:
        return json.load(f)
    
def load_default_building_config(name: str):
    try:
        config_path = files("rctoolbox.configs.buildings") / f"{name}.json"
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise ValueError(f"Default building config '{name}' not found")
    

def list_available_buildings():
    return sorted([
        p.stem for p in files("rctoolbox.configs.buildings").iterdir()
        if p.suffix == ".json"
    ])