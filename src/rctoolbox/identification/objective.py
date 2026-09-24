"""Objective functions for the GA identification.

An objective is any callable ``f(y_true, y_pred) -> float`` where both arrays
are whatever the problem's ``get_truth`` / ``predict`` return, e.g. ``(B, T)``
for a single supervised series or ``(B, T, n)`` for several states. The GA
minimizes it on the training windows and reports it on the validation ones.

It is chosen by ``optimization_config["objective_function"]``:

- a metric name (one of ``KNOWN_METRICS``, e.g. ``"MAE"``) selects the
  built-in objective: the metric's mean over windows plus ``std_weight``
  times its spread (``calculate_objective_function``);
- ``"<file.py>:<function>"`` (path relative to the config file or absolute)
  or ``"<package.module>:<function>"`` names a custom callable
  (``load_objective``). ``"__main__:<function>"`` reaches a function defined
  in the running notebook/script, at the price of not being reproducible from
  the command line.

The same string can be given on the command line as ``--objective-function``.
"""
import importlib
import importlib.util
import os

import numpy as np

from rctoolbox.evaluation.forecast_metrics import compute_forecast_metrics

def load_objective(ref: str, base_dir=None):
    """Resolve ``"<file.py>:<function>"`` or ``"<module>:<function>"`` to a
    callable.

    A ``.py`` path is loaded as a standalone module; a relative path is tried
    against ``base_dir`` (the identification config's directory) first, then
    the current working directory. Anything else is imported as a module.
    """
    if not isinstance(ref, str) or ":" not in ref:
        raise ValueError(
            f"objective reference {ref!r} must look like 'file.py:function' "
            "or 'package.module:function'"
        )
    target, func_name = ref.rsplit(":", 1)

    if target.endswith(".py"):
        candidates = [target]
        if not os.path.isabs(target) and base_dir:
            candidates.insert(0, os.path.join(base_dir, target))
        path = next((c for c in candidates if os.path.isfile(c)), None)
        if path is None:
            raise FileNotFoundError(
                f"objective file {target!r} not found (tried {candidates})"
            )
        spec = importlib.util.spec_from_file_location(
            "_rctoolbox_objective_" + os.path.splitext(os.path.basename(path))[0], path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)

    try:
        fn = getattr(module, func_name)
    except AttributeError:
        raise AttributeError(f"{target!r} has no attribute {func_name!r}") from None
    if not callable(fn):
        raise TypeError(f"{ref!r} resolved to a non-callable {type(fn).__name__}")
    return fn


KNOWN_METRICS = ("MAE", "MSE", "RMSE", "CV(RMSE)", "RE", "NMBE")


def resolve_objective(spec: str, std_weight=None, base_dir=None):
    """Turn an ``objective_function`` config value into a callable.

    ``spec`` is a metric name (built-in objective, ``std_weight`` applies) or
    a ``target:function`` reference (custom callable, ``std_weight`` ignored).
    """
    if not isinstance(spec, str) or not spec:
        raise ValueError(
            "optimization_config['objective_function'] must be a metric name "
            f"{KNOWN_METRICS} or a 'file.py:function' / 'package.module:function' "
            f"reference, got {spec!r}"
        )
    if ":" in spec:
        return load_objective(spec, base_dir=base_dir)
    if spec not in KNOWN_METRICS:
        raise ValueError(
            f"Unknown objective metric {spec!r}. Use one of {KNOWN_METRICS}, or a "
            "'file.py:function' / 'package.module:function' reference."
        )
    from functools import partial
    return partial(calculate_objective_function, metric=spec, std_weight=std_weight or 0.0)


def calculate_objective_function(y_true,
                                 y_pred,
                                 metric,
                                 std_weight):

    performance = compute_forecast_metrics(y_true, y_pred, axis=1)
    objective_value = np.nanmean(performance[metric]) + std_weight * np.nanstd(performance[metric])
    return objective_value


# def make_objective_function(metric, std_weight):
#     def objective(y_true, y_pred):
#         performance = compute_forecast_metrics(y_true, y_pred, axis=1)
#         return performance[metric].mean() + std_weight * performance[metric].std()
#     return objective

# class ObjectiveFunction:
#     def __init__(self, metric, std_weight):
#         self.metric = metric
#         self.std_weight = std_weight

#     def __call__(self, y_true, y_pred):
#         performance = compute_forecast_metrics(y_true, y_pred, axis=1)
#         return performance[self.metric].mean() + self.std_weight * performance[self.metric].std()

