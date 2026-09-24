"""Generic loader for calibrated RC models.

One function for every model class: pass the model class together with the
``results.json`` produced by the identification pipeline, and the model is
rebuilt with the configuration stored at calibration time — heat-power range,
initialization, and integration, each applied only if the class supports it —
plus the selected parameter vector.

    from rctoolbox.models.loader import load_best_model
    from rctoolbox.models.rc.two_r2c.model import TwoR2C

    model = load_best_model("results/results.json", TwoR2C)

The per-model loader modules (``five_r1c_eniso13790.loader``,
``two_r2c.loader``, ``three_r2c.loader``) remain available and unchanged;
this is the class-parameterized alternative so new models do not need a
loader module of their own. (For registry-registered model types,
``three_r2c.loader.load_best_model`` offers the complementary variant that
dispatches on the ``model_type`` stored in the results instead of taking a
class.)
"""
import json


def load_best_model(results_path, model_cls, best_from_cluster=False):
    """Rebuild the calibrated model from a pipeline ``results.json``.

    Parameters
    ----------
    results_path : str or Path
        Path to the ``results.json`` written by the identification pipeline.
    model_cls : type
        The model class to instantiate (e.g. ``TwoR2C``, ``ThreeR2C``,
        ``FiveR1C_ENISO13790``).
    best_from_cluster : bool
        If True, use the best candidate of the best-scoring cluster
        (minimum ``cl_eval_score``) instead of the overall best candidate.
    """
    with open(results_path, "r") as f:
        results = json.load(f)

    model_config    = results["setup"]["identification_config"]["model_config"]
    building_config = results["setup"]["building_config"]

    model = model_cls()

    # Reproduce the calibration-time configuration, feature-by-feature: each
    # setter is applied only if the class provides it and the config carries it.
    if hasattr(model, "set_heat_power_range") and "max_HP_power" in building_config:
        model.set_heat_power_range([0, building_config["max_HP_power"]])
    if hasattr(model, "set_initialization") and "init_config" in model_config:
        model.set_initialization(init_config=model_config["init_config"])
    if hasattr(model, "set_integration") and "integration" in model_config:
        # pass-through semantics: absent/None resolves to the model's fallback
        model.set_integration(model_config.get("integration"))

    if best_from_cluster:
        c = results["results"]["clustering-results"]
        best_key = min(c, key=lambda k: c[k]["best_candidate"]["cl_eval_score"])
        params = c[best_key]["best_candidate"]["params"]
    else:
        params = results["results"]["best_candidate"]["params"]

    model.set_params(params)
    return model
