"""Load a calibrated model back from a results folder written by ``rct-train``.

Works for registry models and for local models alike: for a local model the
code is imported from the copy the pipeline put into the results folder, so
the training folder is not needed.

    from rctoolbox.pipeline.run import load_run

    run = load_run("results")
    run.model                   # best candidate, ready to predict
    run.results["results"]      # the full results.json
    optimizer = run.load_optimizer()   # needs --save-optimizer
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from rctoolbox.pipeline.plugin import ModelPlugin, build_parameter_space, load_optimizer, resolve_plugin


@dataclass
class Run:
    """A calibrated run: the model with the selected parameters, its plugin and the saved results."""

    path: Path
    model: object
    params: dict                   # the selected parameter vector the model carries
    plugin: ModelPlugin
    module: ModuleType | None      # the local model module, None for registry models
    building_config: dict
    model_config: dict
    results: dict

    def build_model(self, params: dict | None = None):
        """A fresh model configured as at calibration time, with ``params`` if given."""
        model = self.plugin.model_factory(self.building_config, self.model_config)
        if params is not None:
            model.set_params(params)
        return model

    def parameter_space(self):
        """The parameter space of the calibration: the bounds the GA searched in."""
        return build_parameter_space(self.plugin, self.building_config, self.model_config)

    def load_optimizer(self):
        """The pickled ``GAOptimizer`` (``optimizer.pkl``, written with ``--save-optimizer``)."""
        return load_optimizer(self.path / "optimizer.pkl", self.module)


def selected_params(results: dict, best_from_cluster: bool = False) -> dict:
    """The overall best candidate, or the best candidate of the best-scoring cluster."""
    if best_from_cluster:
        c = results["results"]["clustering-results"]
        if not c:
            raise ValueError("results.json has no clustering results (stage 3 did not run)")
        best_key = min(c, key=lambda k: c[k]["best_candidate"]["cl_eval_score"])
        return c[best_key]["best_candidate"]["params"]
    return results["results"]["best_candidate"]["params"]


def load_run(results_dir, best_from_cluster: bool = False) -> Run:
    """Rebuild the calibrated model from ``results.json`` (and the model code copied next to it)."""
    results_dir = Path(results_dir)
    with open(results_dir / "results.json") as f:
        results = json.load(f)

    building_config = results["setup"]["building_config"]
    model_config = results["setup"]["identification_config"]["model_config"]
    plugin, module = resolve_plugin(model_config, base_dir=results_dir)

    params = selected_params(results, best_from_cluster)
    model = plugin.model_factory(building_config, model_config)
    model.set_params(params)
    return Run(
        path=results_dir,
        model=model,
        params=params,
        plugin=plugin,
        module=module,
        building_config=building_config,
        model_config=model_config,
        results=results,
    )
