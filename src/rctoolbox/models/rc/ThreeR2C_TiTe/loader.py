"""Load calibrated 3R2C-family models (3R2C, 3R2C-gains) from a results.json.

Dispatches through the pipeline registry on the model_type stored in the
results, so the model is rebuilt exactly as it was configured at calibration
time (heat-power range, initialization, integration)."""
import json

from rctoolbox.pipeline.registry import get_plugin


def load_best_model(results_path, best_from_cluster=False):
    with open(results_path, "r") as f:
        results = json.load(f)

    model_config    = results["setup"]["identification_config"]["model_config"]
    building_config = results["setup"]["building_config"]
    model = get_plugin(model_config["model_type"]).model_factory(building_config, model_config)

    if best_from_cluster:
        c = results["results"]["clustering-results"]
        best_key = min(c, key=lambda k: c[k]["best_candidate"]["cl_eval_score"])
        params = c[best_key]["best_candidate"]["params"]
    else:
        params = results["results"]["best_candidate"]["params"]

    model.set_params(params)
    return model
