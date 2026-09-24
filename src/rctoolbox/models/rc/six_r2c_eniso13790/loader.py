"""Load a fitted 6R2C model from a results.json produced by the pipeline."""
import json
from .model import SixR2C_ENISO13790


def load_best_6r2c(results_path, best_from_cluster=False):
    with open(results_path, "r") as f:
        results = json.load(f)

    max_hp_power = results["setup"]["building_config"]["max_HP_power"]
    init_config  = results["setup"]["identification_config"]["model_config"].get("init_config", {"method": "default"})

    model = SixR2C_ENISO13790()
    model.set_heat_power_range([0, max_hp_power])
    model.set_initialization(init_config=init_config)

    if best_from_cluster:
        c = results["results"]["clustering-results"]
        best_key = min(c, key=lambda k: c[k]["best_candidate"]["cl_eval_score"])
        params   = c[best_key]["best_candidate"]["params"]
    else:
        params = results["results"]["best_candidate"]["params"]

    model.set_params(params)
    return model
